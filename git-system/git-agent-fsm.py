#!/usr/bin/env python3
"""OpenAI-compatible FSM/planning agent for git repositories."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import openai

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.planning_modes import (
    NONBLOCKING_VIOLATION_POLICIES,
    PLANNING_BATCH,
    PLANNING_STEP,
    VIOLATION_IGNORE,
    VIOLATION_RETRY,
    normalize_planning_granularity,
    normalize_violation_policy,
    planning_mode_config,
    property_guidance_text as _property_guidance_text,
    property_policy_text as _property_policy_text,
)
from utils.property_catalog import (
    aps_from_properties,
    load_property_catalog,
    observe_ap_values,
    select_properties,
)
from utils.session import (
    accepted_nodes,
    accepted_plan_from_nodes,
    annotate_node_executed,
    append_result_notice as _shared_append_result_notice,
    append_node,
    build_tree_summary,
    checkpoint_result as _shared_checkpoint_result,
    make_action,
    make_attempt,
    make_planning_node,
    make_planning_tree,
    make_session,
    make_state_path_entry,
    make_verification,
    mark_feasible,
    set_node_outcome,
    start_result_session as _shared_start_result_session,
    write_result as _shared_write_result,
)
from utils.tla_verifier import (
    verify_fsm_trace,
)
from utils.chat_terminal import ChatCommand, ChatTerminal
from utils.git_learning_reset import reset_git_learning_lab
from utils.planning_terminal import (
    RuntimePlanningConfig,
    build_planning_commands,
    format_runtime_config,
    runtime_config_from_values,
)

from property_verifier import PropertyVerifier as GitPropertyVerifier
from git_shell_tools import (
    ALLOWED_GIT_SUBCOMMANDS, SHELL_COMMANDS, TOOL_IMPL, WORK_DIR, _SHELL_NAMES,
    is_git_repo, tool_git, tool_shell,
)
from utils.llm_client import llm as _llm_call, make_client

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_PLAN_STEPS   = 10
MAX_RETRIES      = 3
MAX_OUTPUT_CHARS = 4000
CMD_TIMEOUT      = 20

_RESOURCES_DIR      = Path(__file__).resolve().parent / "resources"
_PROPERTIES_FILE    = _RESOURCES_DIR / "GIT_PROPERTIES_AST.json"

WORK_DIR = Path.cwd().resolve()
DEFAULT_RESULT_DIR = str(Path(__file__).resolve().parents[2] / "playground-llm-agents-fsm" / "results")
_RESULT_DIR_ENV = os.environ.get("GIT_AGENT_FSM_RESULT_DIR")
RESULT_DIR = None if _RESULT_DIR_ENV == "" else Path(_RESULT_DIR_ENV or DEFAULT_RESULT_DIR)
_last_result_path: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs that let the FSM cover both former plan and FSM modes."""

    planning_granularity: str = PLANNING_BATCH
    violation_policy: str = VIOLATION_RETRY
    max_plan_steps: int = MAX_PLAN_STEPS
    max_retries: int = MAX_RETRIES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default




def default_config_from_env() -> AgentConfig:
    mode_config = planning_mode_config(
        os.environ.get("GIT_AGENT_FSM_PLANNING_MODE"),
        retry_default=MAX_RETRIES,
        invalid="raise",
    )
    default_granularity = str(mode_config["planning_granularity"])
    default_policy = str(mode_config["violation_policy"])
    default_retries = int(mode_config["max_retries"])
    return AgentConfig(
        planning_granularity=normalize_planning_granularity(
            os.environ.get("GIT_AGENT_FSM_PLANNING_GRANULARITY")
            or os.environ.get("GIT_AGENT_FSM_PLANNING"),
            default=default_granularity,
            invalid="raise",
        ),
        violation_policy=normalize_violation_policy(
            os.environ.get("GIT_AGENT_FSM_VIOLATION_POLICY")
            or os.environ.get("GIT_AGENT_FSM_VIOLATIONS"),
            default=default_policy,
            invalid="raise",
        ),
        max_plan_steps=_env_int("GIT_AGENT_FSM_MAX_PLAN_STEPS", MAX_PLAN_STEPS),
        max_retries=_env_int("GIT_AGENT_FSM_MAX_RETRIES", default_retries),
    )


def _start_result_session(record: dict[str, Any]) -> str | None:
    global _last_result_path
    result_path = _shared_start_result_session(record, RESULT_DIR)
    if result_path is not None:
        _last_result_path = result_path
    return result_path


def _checkpoint_result(record: dict[str, Any], result_path: str | None) -> str | None:
    return _shared_checkpoint_result(record, result_path)


def _write_result(record: dict[str, Any], result_path: str | None = None) -> str | None:
    global _last_result_path
    result_path = _shared_write_result(record, RESULT_DIR, result_path)
    if result_path is not None:
        _last_result_path = result_path
    return result_path



def last_result_path() -> str | None:
    return _last_result_path

PROPERTY_SAMPLE_SIZE: int | None = None

_ALL_PROPS: list[dict] = load_property_catalog(_PROPERTIES_FILE)
PROPERTIES = select_properties(_ALL_PROPS, sample_size=PROPERTY_SAMPLE_SIZE)

ALL_APS = sorted(set(ap for part in aps_from_properties(PROPERTIES) for ap in part))

_AP_SPEC_BY_NAME: dict[str, dict[str, Any]] = {
    spec.get("name", ""): spec
    for spec in json.loads(
        (_RESOURCES_DIR / "GIT_AP_CANDIDATES.json").read_text(encoding="utf-8")
    ).get("current_state_aps", [])
    if spec.get("name")
}

DEFAULT_OPENAI_MAX_TOKENS = 512
DEFAULT_OPENAI_PLANNING_MAX_TOKENS = 2048

_CLIENT = make_client()
_MAX_TOKENS = int(os.environ.get("GIT_AGENT_FSM_OPENAI_MAX_TOKENS", str(DEFAULT_OPENAI_MAX_TOKENS)))
_PLANNING_MAX_TOKENS = int(os.environ.get("GIT_AGENT_FSM_OPENAI_PLANNING_MAX_TOKENS", str(max(_MAX_TOKENS, DEFAULT_OPENAI_PLANNING_MAX_TOKENS))))



def _extract_json(text: str) -> Any:
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

def phase3_build_s0(model: str) -> dict[str, bool]:
    """
    Phase 3: observe every AP extracted from the selected property ASTs.
    """
    print(f"  \033[36m[Phase 3] Observing {len(ALL_APS)} APs extracted from properties...\033[0m")
    verifier = GitPropertyVerifier(str(WORK_DIR), model=model, client=_CLIENT)
    s0 = observe_ap_values(ALL_APS, verifier.observe_ap)
    for ap in ALL_APS:
        val = s0[ap]
        mark = "✓" if val else "✗"
        print(f"    {mark} {ap[:70]}")
    return s0

PROMPT_4A_SYSTEM = f"""\
You are proposing the next single git action toward a goal.
Working directory: {WORK_DIR}

Tool call signatures:
  git_cmd(command)         — subcommand + flags as one string, e.g. "add ." or "commit -m 'msg'"
                             Allowed: {", ".join(sorted(ALLOWED_GIT_SUBCOMMANDS))}
  shell_cmd(command, args) — bare program name + args list, e.g. command="ls", args=["-la"]
                             Allowed programs: {_SHELL_NAMES}

Rules: NO pipes (|), redirects (>/<), semicolons, or backticks in any argument.

Output ONLY valid JSON:
{{
  "action_label": "short_snake_case_label",
  "tool": "git_cmd | shell_cmd | none",
  "args": {{"command": "..."}},
  "rationale": "one sentence"
}}
If the goal is already achieved, output {{"action_label": "goal_satisfied", "tool": "none", "args": {{}}, "rationale": "done"}}"""

PROMPT_4A_BATCH_SYSTEM = f"""\
You are proposing a complete git action plan toward a goal.
Working directory: {WORK_DIR}

Tool call signatures:
  git_cmd(command)         — subcommand + flags as one string, e.g. "add ." or "commit -m 'msg'"
                             Allowed: {", ".join(sorted(ALLOWED_GIT_SUBCOMMANDS))}
  shell_cmd(command, args) — bare program name + args list, e.g. command="ls", args=["-la"]
                             Allowed programs: {_SHELL_NAMES}

Rules: NO pipes (|), redirects (>/<), semicolons, or backticks in any argument.
The plan will be checked before execution. Keep it short and include verification/readback steps when useful.

Output ONLY valid JSON:
{{
  "plan": [
    {{
      "action_label": "short_snake_case_label",
      "tool": "git_cmd | shell_cmd | none",
      "args": {{"command": "..."}},
      "rationale": "one sentence"
    }}
  ],
  "finish_response": "one sentence"
}}
If the goal is already achieved, output {{"plan": [], "finish_response": "done"}}"""


def _feedback_json(value: list[dict] | None) -> str:
    if not value:
        return "none"
    return json.dumps(value[-5:], indent=2, sort_keys=True)


def prompt4a_propose(goal: str, trace: list[dict], tried: list[str], model: str,
                     config: AgentConfig | None = None,
                     failed_attempts: list[dict] | None = None) -> dict | None:
    config = config or default_config_from_env()
    done_steps = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"

    content, _ = _llm_call(_CLIENT, [
        {"role": "system", "content": PROMPT_4A_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Steps done so far:\n{done_steps}\n\n"
            f"{_property_prompt_block(config.violation_policy)}"
            f"Failed plan attempts and backtrack feedback:\n{_feedback_json(failed_attempts)}\n\n"
            f"Banned first actions at this planning point: {tried or 'none'}\n\n"
            f"What is the next action?"},
    ], model, max_tokens=_PLANNING_MAX_TOKENS)

    result = _extract_json(content)
    return _normalise_proposal(result) if isinstance(result, dict) else None


def _normalise_proposal(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None
    action_label = str(item.get("action_label", "")).strip()
    tool = str(item.get("tool", "none")).strip() or "none"
    args = item.get("args") if isinstance(item.get("args"), dict) else {}
    if not action_label:
        command = args.get("command") if isinstance(args, dict) else None
        action_label = command.replace(" ", "_")[:40] if isinstance(command, str) else tool
    proposal = {
        "action_label": action_label,
        "tool": tool,
        "args": args,
        "rationale": str(item.get("rationale", "")),
    }
    if "finish_response" in item:
        proposal["finish_response"] = str(item.get("finish_response", ""))
    return proposal


def prompt4a_propose_batch(goal: str, trace: list[dict], tried: list[str],
                           max_actions: int, model: str,
                           config: AgentConfig | None = None,
                           failed_attempts: list[dict] | None = None) -> dict | None:
    config = config or default_config_from_env()
    done_steps = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"

    content, _ = _llm_call(_CLIENT, [
        {"role": "system", "content": PROMPT_4A_BATCH_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Steps already accepted:\n{done_steps}\n\n"
            f"{_property_prompt_block(config.violation_policy)}"
            f"Failed plan attempts and backtrack feedback:\n{_feedback_json(failed_attempts)}\n\n"
            f"Banned first actions at this planning point: {tried or 'none'}\n\n"
            f"Return at most {max_actions} remaining action(s)."},
    ], model, max_tokens=_PLANNING_MAX_TOKENS)

    result = _extract_json(content)
    if isinstance(result, list):
        raw_plan = result
        finish_response = ""
    elif isinstance(result, dict):
        raw_plan = result.get("plan", [])
        finish_response = str(result.get("finish_response", ""))
    else:
        return None

    if not isinstance(raw_plan, list):
        return None
    plan = []
    for item in raw_plan[:max_actions]:
        proposal = _normalise_proposal(item)
        if proposal is not None:
            plan.append(proposal)
    return {"plan": plan, "finish_response": finish_response}

PROMPT_4B_SYSTEM = """\
You are predicting the value of one atomic proposition after a git action is executed
by the local tool wrapper.

Predict the post-action value the live AP observer would read from the repository,
not the value that would hold if the user's intended workflow succeeded.

Important:
- The command may fail. Account for non-zero exits, rejected pushes, merge/rebase
  conflicts, missing editor state, detached HEAD, stale remotes, and tool-denied
  commands.
- shell_cmd can only run its allowed shell programs. It cannot run git.
- git_cmd runs git directly, but git itself can still reject an operation.
- A failed or denied command usually leaves repository-state APs unchanged, except
  APs about the most recent action or partially-started git operations.
- Consider false-to-true transitions. Do not assume APs that are currently FALSE
  stay FALSE.

Output ONLY valid JSON:
{"value": true, "reason": "one sentence"}
{"value": false, "reason": "one sentence"}"""


def _ap_definition(ap: str) -> str:
    spec = _AP_SPEC_BY_NAME.get(ap) or {}
    description = str(spec.get("description") or ap)
    commands = spec.get("git_commands") or []
    if not commands:
        return description
    command_text = "; ".join(str(command) for command in commands[:4])
    if len(commands) > 4:
        command_text += "; ..."
    return f"{description}\nObserver evidence commands: {command_text}"


def _format_ap_state_for_prompt(state: dict[str, bool]) -> str:
    true_aps = [ap for ap in ALL_APS if state.get(ap, False)]
    false_aps = [ap for ap in ALL_APS if not state.get(ap, False)]
    return (
        "TRUE APs:\n"
        + ("\n".join(f"- {ap}" for ap in true_aps) or "- (none)")
        + "\n\nFALSE APs:\n"
        + ("\n".join(f"- {ap}" for ap in false_aps) or "- (none)")
    )


def _command_prediction_notes(tool: str, args: dict) -> str:
    command = str(args.get("command", "") if isinstance(args, dict) else "").strip()
    notes: list[str] = []
    if tool == "shell_cmd":
        notes.append(
            "shell_cmd allowed programs are: %s. Any other command returns an error and does not mutate git state."
            % _SHELL_NAMES
        )
        if command == "git":
            notes.append(
                "This exact shell_cmd attempts to run git, which is denied by the shell tool; predict no git repository mutation."
            )
    elif tool == "git_cmd":
        notes.append(
            "git_cmd executes git subcommands directly; shell metacharacters are denied before git runs."
        )
        subcommand = command.split()[0] if command else ""
        if subcommand == "fetch":
            notes.append(
                "fetch may update remote-tracking refs and can change ahead/behind/divergence APs without changing the current branch tip."
            )
        elif subcommand == "pull":
            notes.append(
                "pull can fail on divergent branches unless merge/rebase/ff-only is configured; if it fails, unresolved divergence usually remains."
            )
        elif subcommand == "push":
            notes.append(
                "push can be rejected for non-fast-forward, stale force-with-lease, authentication, permissions, or protected branch policy."
            )
        elif subcommand == "rebase":
            notes.append(
                "rebase can fail with conflicts and leave an in-progress rebase until continue/abort/skip resolves it."
            )
            if "-i" in command.split():
                notes.append(
                    "interactive rebase may fail in this terminal when EDITOR is unset or the terminal is non-interactive."
                )
        elif subcommand in {"cherry-pick", "revert", "merge"}:
            notes.append(
                f"{subcommand} can stop with conflicts and leave an in-progress operation."
            )
        elif subcommand == "reset":
            notes.append(
                "reset --hard can move the current branch backward and discard working-tree changes."
            )
        elif subcommand in {"switch", "checkout"}:
            notes.append(
                f"{subcommand} can fail if the target ref does not exist or local changes would be overwritten."
            )
        elif subcommand == "commit":
            notes.append(
                "commit can fail if there is nothing staged, identity is missing, or an in-progress operation still needs resolution."
            )
        elif subcommand == "stash":
            notes.append(
                "stash push reports 'No local changes to save' and leaves stash-related state unchanged when the tree is clean."
            )
    elif tool == "none":
        notes.append("No command runs; repository-state APs should remain unchanged.")
    else:
        notes.append("Unknown tool; execution will produce an error and should not mutate git state.")
    return "\n".join(f"- {note}" for note in notes)

def prompt4b_predict_ap(ap: str, current_val: bool,
                        action_label: str, tool: str, args: dict,
                        model: str,
                        *,
                        current_state: dict[str, bool] | None = None,
                        trace: list[dict] | None = None) -> tuple[bool, str]:
    recent_steps = trace[-3:] if trace else []
    recent_text = "\n".join(
        f"- {step.get('action_label', 'unknown')}: "
        f"{step.get('tool', 'none')}({json.dumps(step.get('args') or {})})"
        for step in recent_steps
    ) or "- (none)"
    state_text = (
        _format_ap_state_for_prompt(current_state)
        if current_state is not None
        else f"- {ap}: {'TRUE' if current_val else 'FALSE'}"
    )
    content, _ = _llm_call(_CLIENT, [
        {"role": "system", "content": PROMPT_4B_SYSTEM},
        {"role": "user",   "content":
            f"Atomic proposition: {ap}\n"
            f"Definition: {_ap_definition(ap)}\n"
            f"Current value: {'TRUE' if current_val else 'FALSE'}\n\n"
            f"Full current AP state before this action:\n{state_text}\n\n"
            f"Recently accepted planning steps:\n{recent_text}\n\n"
            f"Action: {action_label}\n"
            f"Command: {tool}({json.dumps(args)})\n\n"
            f"Execution/tool notes:\n{_command_prediction_notes(tool, args)}"},
    ], model)
    result = _extract_json(content)
    if isinstance(result, dict) and "value" in result:
        return bool(result["value"]), result.get("reason", "")
    return current_val, "parse error — unchanged"


def prompt4b_predict(action_label: str, tool: str, args: dict,
                     s_current: dict[str, bool], model: str,
                     trace: list[dict] | None = None) -> dict[str, bool]:
    s_after = dict(s_current)

    for ap in ALL_APS:
        current_val = bool(s_current.get(ap, False))
        new_val, _ = prompt4b_predict_ap(
            ap,
            current_val,
            action_label,
            tool,
            args,
            model,
            current_state=s_current,
            trace=trace,
        )
        s_after[ap] = new_val

    s_after["last_action"] = action_label
    return s_after

def _observe_execution_ap_state(verifier: GitPropertyVerifier, action_label: str) -> dict[str, bool]:
    ap_state = observe_ap_values(ALL_APS, verifier.observe_ap)
    ap_state["last_action"] = action_label
    return ap_state


def _diff_ap_states(previous: dict[str, bool], current: dict[str, bool]) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for name in ALL_APS:
        prev_val = previous.get(name)
        curr_val = current.get(name)
        if prev_val is not None and prev_val != curr_val:
            changes.append({"name": name, "before": prev_val, "after": curr_val})
    return changes


def _run_verification(
    s0: dict[str, bool],
    trace: list[dict],
    *,
    is_complete_trace: bool = False,
) -> tuple[bool, str, str, dict]:
    """Build TLA+ spec for s0 + trace and run TLC.

    Returns (passed, tla_spec, violations_summary, shared_verification_result).
    trace entries: {action_label, state_after: dict[str,bool]}
    """
    action_labels = [str(step["action_label"]) for step in trace]
    states_after = [step["state_after"] for step in trace]
    result = verify_fsm_trace(
        s0,
        action_labels,
        states_after,
        ALL_APS,
        PROPERTIES,
        module_name="GitTrace",
        timeout=CMD_TIMEOUT,
        is_complete_trace=is_complete_trace,
    )
    tlc_result = result["tlc_result"]
    summary = "; ".join(tlc_result.get("violations", [])) or tlc_result.get("reason", "")
    return result["passed"], result["tla_spec"], summary, result

def _planning_action_from_step(step: dict) -> dict:
    return make_action(
        step.get("action_label"),
        step.get("tool", "none"),
        step.get("args") or {},
    )


def _property_prompt_block(policy: str) -> str:
    if policy == VIOLATION_IGNORE:
        return ""
    return (
        f"Property policy:\n{_property_policy_text(policy)}\n\n"
        f"Properties to avoid violating:\n{_property_guidance_text(PROPERTIES, bullet=True)}\n\n"
    )


def _state_path_from_trace(trace: list[dict]) -> list[dict]:
    return [
        make_state_path_entry(
            _planning_action_from_step(step),
            step.get("state_after") or {},
        )
        for step in trace
    ]


def _branch_retry_budget(config: AgentConfig) -> int:
    if config.violation_policy in NONBLOCKING_VIOLATION_POLICIES:
        return max(1, config.max_retries)
    return config.max_retries


def _proposal_finish_response(proposal: dict | None) -> str:
    if not isinstance(proposal, dict):
        return "done"
    return str(
        proposal.get("finish_response")
        or proposal.get("rationale")
        or "done"
    )


def _request_plan_bundle(
    *,
    goal: str,
    current_state: dict[str, bool],
    accepted_trace: list[dict],
    failed_attempts: list[dict],
    banned_first_actions: list[str],
    depth: int,
    model: str,
    config: AgentConfig,
) -> dict | None:
    if config.planning_granularity == PLANNING_BATCH:
        max_actions = max(1, config.max_plan_steps - depth)
        return prompt4a_propose_batch(
            goal,
            accepted_trace,
            banned_first_actions,
            max_actions,
            model,
            config,
            failed_attempts=failed_attempts,
        )

    proposal = prompt4a_propose(
        goal,
        accepted_trace,
        banned_first_actions,
        model,
        config,
        failed_attempts=failed_attempts,
    )
    if proposal is None:
        return None
    if str(proposal.get("action_label", "")) == "goal_satisfied":
        return {
            "plan": [],
            "finish_response": _proposal_finish_response(proposal),
        }
    return {
        "plan": [proposal],
        "finish_response": _proposal_finish_response(proposal),
    }


def _verify_candidate(
    *,
    proposal: dict,
    s0: dict[str, bool],
    s_current: dict[str, bool],
    trace: list[dict],
    model: str,
    is_complete_trace: bool = False,
) -> dict:
    action_label = proposal.get("action_label", "unknown")
    tool = proposal.get("tool", "none")
    args = proposal.get("args") or {}

    print(f"    -> {action_label}: {tool}({json.dumps(args)})")
    s_after = prompt4b_predict(action_label, tool, args, s_current, model, trace=trace)
    candidate = {
        "action_label": action_label,
        "tool": tool,
        "args": args,
        "state_before": dict(s_current),
        "state_after": s_after,
    }

    passed, tla_spec, violations_str, verification_result = _run_verification(
        s0,
        trace + [candidate],
        is_complete_trace=is_complete_trace,
    )
    violations_list = [v for v in violations_str.split(";") if v.strip()] if violations_str else []

    tlc_result = verification_result.get("tlc_result", {})
    skipped = bool(tlc_result.get("skipped"))
    verif = make_verification(
        passed=passed,
        properties_checked=verification_result.get("properties_checked", []),
        violations=violations_list,
        tla_spec=tla_spec,
        skipped=skipped,
    )
    failure_feedback = None
    if not passed:
        failure_feedback = {
            "type": "verification_skipped" if skipped else "tla_property_violation",
            "action": _planning_action_from_step(candidate),
            "violations": violations_list,
            "message": violations_str or (
                "TLC verification was skipped." if skipped else "TLC verification failed."
            ),
        }
    return {
        "candidate": candidate,
        "state_after": s_after,
        "passed": passed,
        "violations_str": violations_str,
        "verification": verif,
        "failure": failure_feedback,
    }


def _accept_node(node: dict, candidate: dict, check: dict) -> None:
    result = (
        "accepted_with_ignored_violations"
        if check.get("ignored_by_policy")
        else "accepted"
    )
    set_node_outcome(
        node,
        result=result,
        action_label=candidate.get("action_label", "unknown"),
        tool=candidate.get("tool", "none"),
        args=candidate.get("args") or {},
        state_after=check.get("state_after") or {},
        verification=check.get("verification"),
    )


def _search_plan(
    *,
    goal: str,
    s0: dict[str, bool],
    current_state: dict[str, bool],
    accepted_trace: list[dict],
    depth: int,
    planning_tree: dict,
    parent_node_id: int | None,
    inherited_failures: list[dict],
    hint_plan: list[dict] | None,
    model: str,
    config: AgentConfig,
    checkpoint: Callable[[], None],
) -> dict:
    """Search recursively using one planning-tree node per candidate action."""
    if len(planning_tree["nodes"]) >= config.max_plan_steps:
        return {
            "success": False,
            "failure": {
                "type": "max_tries",
                "depth": depth,
                "nodes_created": len(planning_tree["nodes"]),
                "message": (
                    "Planning exceeded the max node budget of %d."
                    % config.max_plan_steps
                ),
            },
            "finish_response": "No feasible property-satisfying plan found.",
        }

    node_id = len(planning_tree["nodes"])
    node = make_planning_node(
        node_id=node_id,
        parent_node_id=parent_node_id,
        depth=depth,
        state_before=dict(current_state),
        state_path=_state_path_from_trace(accepted_trace),
    )
    append_node(planning_tree, node, link_parent=True)
    checkpoint()

    failed_attempts = list(inherited_failures)
    banned_first_actions: list[str] = []
    current_hint = (
        copy.deepcopy(hint_plan)
        if hint_plan and config.planning_granularity == PLANNING_BATCH
        else []
    )

    for child_index in range(_branch_retry_budget(config)):
        if current_hint:
            print(f"  [depth {depth+1} attempt {child_index+1}] reusing planned suffix")
            plan_bundle = {
                "plan": copy.deepcopy(current_hint),
                "finish_response": "done",
            }
            attempt = make_attempt(
                retry_index=child_index,
                accepted=False,
                planner_decision=plan_bundle,
                plan_source="hint_plan",
                hint_plan_length=len(current_hint),
            )
            current_hint = []
        else:
            print(f"  [depth {depth+1} attempt {child_index+1}] proposing action...")
            try:
                plan_bundle = _request_plan_bundle(
                    goal=goal,
                    current_state=current_state,
                    accepted_trace=accepted_trace,
                    failed_attempts=failed_attempts,
                    banned_first_actions=banned_first_actions,
                    depth=depth,
                    model=model,
                    config=config,
                )
            except Exception as exc:
                failure = {
                    "type": "planning_error",
                    "depth": depth,
                    "child_index": child_index,
                    "message": str(exc),
                }
                node["attempts"].append(
                    make_attempt(
                        retry_index=child_index,
                        accepted=False,
                        error=str(exc),
                        failure_feedback=failure,
                    )
                )
                checkpoint()
                failed_attempts.append(failure)
                current_hint = []
                continue

            if plan_bundle is None:
                failure = {
                    "type": "planning_error",
                    "depth": depth,
                    "child_index": child_index,
                    "message": "Planner returned unparsable or invalid JSON.",
                }
                node["attempts"].append(
                    make_attempt(
                        retry_index=child_index,
                        accepted=False,
                        error=failure["message"],
                        failure_feedback=failure,
                    )
                )
                checkpoint()
                failed_attempts.append(failure)
                current_hint = []
                continue

            attempt = make_attempt(
                retry_index=child_index,
                accepted=False,
                planner_decision=plan_bundle,
            )

        plan = plan_bundle.get("plan", [])
        if not isinstance(plan, list):
            plan = []
        if config.planning_granularity == PLANNING_STEP and len(plan) > 1:
            attempt["truncated_to_single_step"] = True
            plan = plan[:1]
            plan_bundle = dict(plan_bundle)
            plan_bundle["plan"] = plan
            attempt["planner_decision"] = plan_bundle

        first_action = plan[0] if plan else None
        if isinstance(first_action, dict) and first_action.get("action_label") == "goal_satisfied":
            plan = []
            plan_bundle = dict(plan_bundle)
            plan_bundle["plan"] = []
            plan_bundle["finish_response"] = _proposal_finish_response(first_action)
            attempt["planner_decision"] = plan_bundle

        if not plan:
            finish_response = str(plan_bundle.get("finish_response") or "done")
            attempt["accepted"] = True
            attempt["finish"] = True
            node["attempts"].append(attempt)
            set_node_outcome(
                node,
                result="finish",
                finish_response=finish_response,
            )
            checkpoint()
            return {
                "success": True,
                "plan": [],
                "finish_response": finish_response,
                "node_id": node_id,
            }

        action = plan[0]
        tail = plan[1:] if config.planning_granularity == PLANNING_BATCH else []
        check = _verify_candidate(
            proposal=action,
            s0=s0,
            s_current=current_state,
            trace=accepted_trace,
            model=model,
            is_complete_trace=(
                not tail and config.planning_granularity == PLANNING_BATCH
            ),
        )
        candidate = check["candidate"]
        attempt["action"] = candidate
        attempt["proposal"] = action
        attempt["predicted_state_after"] = check["state_after"]
        attempt["verification"] = check["verification"]

        if not check["passed"]:
            failure = check["failure"]
            can_continue = (
                config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
                and failure is not None
            )
            if can_continue:
                attempt["ignored_property_violation"] = failure
                check["ignored_by_policy"] = True
            else:
                if check["violations_str"]:
                    print(f"    FAIL: {check['violations_str'][:120]}")
                attempt["failure_feedback"] = failure
                node["attempts"].append(attempt)
                checkpoint()
                if failure is not None:
                    failed_attempts.append(failure)
                banned_first_actions.append(str(candidate.get("action_label", "unknown")))
                current_hint = []
                continue

        if check["passed"]:
            print("    PASS")
        elif check["violations_str"]:
            print(f"    {config.violation_policy.upper()} VIOLATION: {check['violations_str'][:120]}")
        else:
            print(f"    {config.violation_policy.upper()} VERIFICATION FAILURE")

        if not tail and config.planning_granularity == PLANNING_BATCH:
            attempt["accepted"] = True
            node["attempts"].append(attempt)
            _accept_node(node, candidate, check)
            checkpoint()
            return {
                "success": True,
                "plan": [candidate],
                "finish_response": str(plan_bundle.get("finish_response") or "done"),
                "node_id": node_id,
            }

        child_result = _search_plan(
            goal=goal,
            s0=s0,
            current_state=check["state_after"],
            accepted_trace=accepted_trace + [candidate],
            depth=depth + 1,
            planning_tree=planning_tree,
            parent_node_id=node_id,
            inherited_failures=[],
            hint_plan=tail if config.planning_granularity == PLANNING_BATCH else None,
            model=model,
            config=config,
            checkpoint=checkpoint,
        )
        attempt["child_node_id"] = child_result.get("node_id")

        if child_result.get("success"):
            attempt["accepted"] = True
            node["attempts"].append(attempt)
            _accept_node(node, candidate, check)
            checkpoint()
            return {
                "success": True,
                "plan": [candidate] + child_result.get("plan", []),
                "finish_response": child_result.get(
                    "finish_response",
                    str(plan_bundle.get("finish_response") or "done"),
                ),
                "node_id": node_id,
            }

        child_failure = child_result.get("failure") or {
            "type": "child_failure",
            "depth": depth + 1,
            "message": "Child subtree exhausted.",
        }
        attempt["child_failure"] = child_failure
        node["attempts"].append(attempt)
        checkpoint()
        if child_failure.get("type") == "max_tries":
            set_node_outcome(node, result="backtracked", failure=child_failure)
            checkpoint()
            return {
                "success": False,
                "failure": child_failure,
                "finish_response": "No feasible property-satisfying plan found.",
                "node_id": node_id,
            }
        failed_attempts.append(child_failure)
        current_hint = []

    exhaustion_failure = {
        "type": "branch_exhausted",
        "depth": depth,
        "node_id": node_id,
        "failed_attempts": failed_attempts,
        "message": (
            "All %d action attempts at this node were exhausted."
            % _branch_retry_budget(config)
        ),
    }
    set_node_outcome(node, result="backtracked", failure=exhaustion_failure)
    checkpoint()
    return {
        "success": False,
        "failure": exhaustion_failure,
        "finish_response": "No feasible property-satisfying plan found.",
        "node_id": node_id,
    }


def phase4_plan(goal: str, s0: dict[str, bool],
                model: str,
                config: AgentConfig | None = None,
                planning_tree: dict | None = None,
                result_record: dict[str, Any] | None = None,
                result_path: str | None = None) -> tuple[list[dict], dict, bool]:
    """
    Returns (trace, planning_tree, feasible).
    trace         — accepted candidate dicts {action_label, tool, args, state_before, state_after}
    planning_tree — canonical planning-tree dict in utils.session schema
    """
    config = config or default_config_from_env()
    planning_tree = planning_tree or make_planning_tree(
        mode="%s_%s" % (config.planning_granularity, config.violation_policy),
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        max_branch_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=PROPERTIES,
        initial_state=s0,
    )
    planning_tree.setdefault("max_branch_retries", config.max_retries)

    def checkpoint() -> None:
        if result_record is not None:
            _checkpoint_result(result_record, result_path)

    print(
        "\n\033[35m[Phase 4] Planning with TLC verification "
        f"({config.planning_granularity}, violations={config.violation_policy}, "
        f"retries={config.max_retries})...\033[0m"
    )

    result = _search_plan(
        goal=goal,
        s0=s0,
        current_state=dict(s0),
        accepted_trace=[],
        depth=0,
        planning_tree=planning_tree,
        parent_node_id=None,
        inherited_failures=[],
        hint_plan=None,
        model=model,
        config=config,
        checkpoint=checkpoint,
    )
    feasible = bool(result.get("success"))
    if feasible:
        return result.get("plan", []), planning_tree, True

    if result.get("failure"):
        planning_tree["failure"] = result["failure"]
    return [], planning_tree, False


def _rejected_action_labels(planning_tree: dict) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: Any) -> None:
        text = str(label or "").strip()
        if text and text not in seen:
            labels.append(text)
            seen.add(text)

    for node in planning_tree.get("nodes", []):
        if not isinstance(node, dict):
            continue
        action = node.get("action") or {}
        if node.get("result") == "rejected" and isinstance(action, dict):
            add(action.get("label"))
        for attempt in node.get("attempts", []):
            if not isinstance(attempt, dict) or attempt.get("accepted"):
                continue
            proposal = attempt.get("proposal") or attempt.get("action") or {}
            if isinstance(proposal, dict):
                add(proposal.get("action_label") or proposal.get("label"))
    return labels


def phase5_execute(
    trace: list[dict],
    *,
    s0: dict[str, bool],
    model: str,
    planning_tree: dict | None = None,
    result_record: dict[str, Any] | None = None,
    result_path: str | None = None,
) -> list[str]:
    results = []
    if not trace:
        return results
    executed_nodes = accepted_nodes(planning_tree) if planning_tree is not None else []
    verifier = GitPropertyVerifier(str(WORK_DIR), model=model, client=_CLIENT)
    observed_trace: list[dict] = []
    previous_ap_state = s0

    print(f"\n\033[35m[Phase 5] Executing {len(trace)} verified step(s):\033[0m")
    for i, step in enumerate(trace, 1):
        tool = step.get("tool", "none")
        args = step.get("args") or {}
        action_label = str(step["action_label"])
        print(f"\n  \033[33m[exec {i}/{len(trace)}] {action_label}: {tool}({json.dumps(args)})\033[0m")

        if tool == "none" or not args:
            result = "(no-op)"
        else:
            impl = TOOL_IMPL.get(tool)
            if impl is None:
                result = f"[error] unknown tool '{tool}'"
            else:
                try:
                    result = impl(**args)
                except Exception as e:
                    result = f"[error] {e}"

            print(f"  \033[2m{result}\033[0m")
        results.append(result)
        ap_state = _observe_execution_ap_state(verifier, action_label)
        observed_step = dict(step)
        observed_step["state_after"] = ap_state
        observed_trace.append(observed_step)
        _, _, _, tla_result = _run_verification(
            s0,
            observed_trace,
            is_complete_trace=(i == len(trace)),
        )
        tlc_result = tla_result.get("tlc_result", {})
        if tlc_result.get("skipped"):
            reason = str(tlc_result.get("reason") or "TLC verification was skipped.")
            print(f"  \033[33mexecution verification SKIPPED: {reason[:120]}\033[0m")
        elif tla_result["passed"]:
            print("  \033[32mexecution verification PASS\033[0m")
        else:
            violations = "; ".join(tlc_result.get("violations", []))
            print(f"  \033[31mexecution verification FAIL: {violations[:120]}\033[0m")
        node_index = i - 1
        if node_index < len(executed_nodes):
            annotate_node_executed(
                executed_nodes[node_index],
                execution_step=node_index,
                execution_result=result,
                ap_state=ap_state,
                ap_changes=_diff_ap_states(previous_ap_state, ap_state),
                tla_verification=tla_result,
            )
        previous_ap_state = ap_state
        if result_record is not None:
            _checkpoint_result(result_record, result_path)

    return results

PROMPT_5_SYSTEM = """\
You are summarising the result of a verified git workflow for the user.
Be concise (3-5 sentences). Cover: what was done, what changed, current repo state."""

def prompt5_summary(goal: str, trace: list[dict],
                    exec_results: list[str], model: str) -> str:
    trace_text = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        f"\n     result: {exec_results[i-1][:200] if i <= len(exec_results) else 'N/A'}"
        for i, s in enumerate(trace, 1)
    )
    content, _ = _llm_call(_CLIENT, [
        {"role": "system", "content": PROMPT_5_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\nExecuted steps:\n{trace_text}"},
    ], model)
    return content

PROMPT_7_SYSTEM = """\
You are explaining to a user why their git request could not be safely executed.
The plan was blocked — either every candidate action violated a safety property,
or retries were exhausted.
Be direct and specific: name the property that blocked the plan, and suggest a safe alternative if one exists."""

def prompt7_blocked(goal: str, s0: dict[str, bool],
                    tried_actions: list[str], model: str) -> str:
    s0_text = "\n".join(f"  {'T' if v else 'F'}  {ap}" for ap, v in s0.items())
    content, _ = _llm_call(_CLIENT, [
        {"role": "system", "content": PROMPT_7_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Initial state:\n{s0_text}\n\n"
            f"Actions tried and rejected: {tried_actions}"},
    ], model)
    return content

def handle_query(goal: str, model: str,
                 config: AgentConfig | None = None) -> tuple[str, dict]:
    """Run one query. Returns (response_text, session_turn_dict)."""
    config = config or default_config_from_env()

    planning_mode = "%s_%s" % (config.planning_granularity, config.violation_policy)
    turn = make_session(
        agent="git-agent-fsm",
        model=model,
        domain="git",
        request=goal,
        work_dir=str(WORK_DIR),
        properties=PROPERTIES,
        planning_config={
            "planning_mode":        planning_mode,
            "planning_granularity": config.planning_granularity,
            "violation_policy":     config.violation_policy,
            "max_plan_steps":       config.max_plan_steps,
            "max_retries":          config.max_retries,
            "max_branch_retries":   config.max_retries,
        },
    )
    turn["planning_tree"] = make_planning_tree(
        mode=planning_mode,
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        max_branch_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=PROPERTIES,
    )
    turn["status"] = "planning"
    result_path = _start_result_session(turn)

    print(f"\n\033[35m[Phase 3] Observing initial state s0...\033[0m")
    s0 = phase3_build_s0(model)
    turn["planning_tree"]["initial_state"] = s0
    _checkpoint_result(turn, result_path)

    trace, planning_tree, feasible = phase4_plan(
        goal,
        s0,
        model,
        config,
        planning_tree=turn["planning_tree"],
        result_record=turn,
        result_path=result_path,
    )
    if feasible:
        mark_feasible(planning_tree, accepted_plan_from_nodes(planning_tree))
    else:
        planning_tree["feasible"] = False
        planning_tree["accepted_plan"] = []

    executed_trace = trace if feasible else []
    if feasible and executed_trace:
        turn["status"] = "executing"
        _checkpoint_result(turn, result_path)
        exec_results = phase5_execute(
            executed_trace,
            s0=s0,
            model=model,
            planning_tree=planning_tree,
            result_record=turn,
            result_path=result_path,
        )
    else:
        exec_results = []

    if feasible:
        if trace:
            response = prompt5_summary(goal, trace, exec_results, model)
        else:
            response = "The request appears already satisfied; no git action was executed."
        turn["status"] = "finished"
    else:
        tried_all = _rejected_action_labels(planning_tree)
        response   = prompt7_blocked(goal, s0, tried_all, model)
        turn["status"] = "infeasible"

    planning_tree["tree_summary"] = build_tree_summary(planning_tree)
    turn["final_message"] = response
    result_path = _write_result(turn, result_path)
    return _shared_append_result_notice(response, result_path), turn

def save_session(turns: list[dict], model: str,
                 config: AgentConfig | None = None) -> Path:
    config = config or default_config_from_env()
    latest_turn = turns[-1] if turns else None
    planning_mode = "%s_%s" % (config.planning_granularity, config.violation_policy)
    if latest_turn is not None:
        session = dict(latest_turn)
    else:
        session = make_session(
            agent="git-agent-fsm",
            model=model,
            domain="git",
            request="(multi-turn session)",
            work_dir=str(WORK_DIR),
            properties=PROPERTIES,
            planning_config={
                "planning_mode": planning_mode,
                "planning_granularity": config.planning_granularity,
                "violation_policy": config.violation_policy,
                "max_plan_steps": config.max_plan_steps,
                "max_retries": config.max_retries,
                "max_branch_retries": config.max_retries,
            },
        )
    if isinstance(session.get("planning_tree"), dict):
        session["planning_tree"]["tree_summary"] = build_tree_summary(session["planning_tree"])
    session["turn_count"] = len(turns)
    session["latest_turn_index"] = len(turns) - 1 if turns else None
    session["turns"] = turns
    path = _write_result(session)
    if path is None:
        raise RuntimeError("Result saving is disabled")
    return Path(path)

def repl() -> None:
    model = os.environ.get("SHRDLU_OPENAI_MODEL", DEFAULT_MODEL)
    config = default_config_from_env()
    runtime_config = runtime_config_from_values(
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        max_retries=config.max_retries,
        retry_default=MAX_RETRIES,
        max_steps=config.max_plan_steps,
    )

    in_repo = is_git_repo()
    repo_notice = "" if in_repo else "  \033[33m(not a git repo)\033[0m"
    sample_note = (f"sampled {len(PROPERTIES)}/{len(_ALL_PROPS)}"
                   if PROPERTY_SAMPLE_SIZE else f"all {len(PROPERTIES)}")

    def intro_lines() -> list[str]:
        return [
            f"\033[1mgit-agent-fsm\033[0m  model={model}  cwd={WORK_DIR}{repo_notice}",
            f"Properties: {sample_note} | {len(ALL_APS)} APs",
            "Planning: %s" % format_runtime_config(runtime_config),
            "Type /help for commands, /exit to quit.",
        ]

    def help_title() -> str:
        return "git-agent-fsm commands (model=%s):" % model

    def show_props(_args: str) -> str:
        return "%d properties | %d APs" % (len(PROPERTIES), len(ALL_APS))

    def show_cwd(_args: str) -> str:
        return str(WORK_DIR)

    def reset_lab(_args: str) -> str:
        return reset_git_learning_lab(WORK_DIR).message

    def set_model(args: str) -> str:
        nonlocal model
        if args:
            model = args.strip()
        return "Model: %s" % model

    def get_planning_config() -> RuntimePlanningConfig:
        return runtime_config

    def set_planning_config(next_config: RuntimePlanningConfig) -> RuntimePlanningConfig:
        nonlocal config, runtime_config
        runtime_config = next_config
        config = replace(
            config,
            planning_granularity=next_config.planning_granularity,
            violation_policy=next_config.violation_policy,
            max_retries=next_config.max_retries,
        )
        return runtime_config

    def handle_message(user_input: str) -> str:
        try:
            answer, _detail = handle_query(user_input, model, config)
        except openai.OpenAIError as e:
            answer = f"[openai error] {e}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer = f"[error] {e}"
        return answer

    ChatTerminal(
        name="git-agent-fsm",
        message_handler=handle_message,
        intro=intro_lines,
        help_title=help_title,
        help_footer="Each query runs: Phase 3 (observe s0) -> Phase 4 (plan+TLC) -> Phase 5 (execute).",
        commands=[
            ChatCommand(("/reset", "reset"), "reset git-learning-lab from parent installer", reset_lab),
            ChatCommand(("/props",), "show property counts", show_props),
            ChatCommand(("/model",), "switch OpenAI model", set_model, "<name>"),
            *build_planning_commands(
                get_config=get_planning_config,
                set_config=set_planning_config,
                retry_default=MAX_RETRIES,
            ),
            ChatCommand(("/cwd",), "show working directory", show_cwd),
        ],
    ).run()

if __name__ == "__main__":
    repl()
