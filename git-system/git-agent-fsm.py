#!/usr/bin/env python3
"""OpenAI-compatible FSM/planning agent for git repositories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

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
    append_node,
    build_tree_summary,
    make_action,
    make_attempt,
    make_planning_node,
    make_planning_tree,
    make_session,
    make_state_path_entry,
    make_verification,
    mark_accepted_branch_backtracked,
    mark_feasible,
)
from utils.tla_verifier import (
    verify_fsm_trace,
)

from property_verifier import TransitionPropertyVerifier as GitPropertyVerifier

DEFAULT_MODEL    = "gpt-4o-mini"
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


def _normalize_planning_granularity(
    value: str | None,
    default: str = PLANNING_BATCH,
    *,
    invalid: Literal["default", "raise"] = "default",
) -> str:
    return normalize_planning_granularity(
        value,
        default=default,
        invalid=invalid,
    )


def _normalize_violation_policy(
    value: str | None,
    default: str = VIOLATION_RETRY,
    *,
    invalid: Literal["default", "raise"] = "default",
) -> str:
    return normalize_violation_policy(value, default=default, invalid=invalid)


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
        planning_granularity=_normalize_planning_granularity(
            os.environ.get("GIT_AGENT_FSM_PLANNING_GRANULARITY")
            or os.environ.get("GIT_AGENT_FSM_PLANNING"),
            default_granularity,
            invalid="raise",
        ),
        violation_policy=_normalize_violation_policy(
            os.environ.get("GIT_AGENT_FSM_VIOLATION_POLICY")
            or os.environ.get("GIT_AGENT_FSM_VIOLATIONS"),
            default_policy,
            invalid="raise",
        ),
        max_plan_steps=_env_int("GIT_AGENT_FSM_MAX_PLAN_STEPS", MAX_PLAN_STEPS),
        max_retries=_env_int("GIT_AGENT_FSM_MAX_RETRIES", default_retries),
    )


def _start_result_session(record: dict[str, Any]) -> str | None:
    global _last_result_path
    if RESULT_DIR is None:
        return None
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_path = RESULT_DIR / ("result_%s.json" % timestamp)
    record["_live"] = True
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _last_result_path = str(result_path)
    return _last_result_path


def _checkpoint_result(record: dict[str, Any], result_path: str | None) -> str | None:
    if not result_path:
        return result_path
    record["_live"] = True
    Path(result_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result_path


def _write_result(record: dict[str, Any], result_path: str | None = None) -> str | None:
    global _last_result_path
    if RESULT_DIR is None and not result_path:
        return None
    if result_path is None:
        result_path = _start_result_session(record)
    if not result_path:
        return None
    record.pop("_live", None)
    Path(result_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    _last_result_path = str(result_path)
    return _last_result_path


def _append_result_notice(message: str, result_path: str | None) -> str:
    if not result_path:
        return message
    return message + "\n\nResult saved to %s" % result_path


def last_result_path() -> str | None:
    return _last_result_path

ALLOWED_GIT = {
    "status", "log", "diff", "show", "branch", "checkout", "switch",
    "add", "commit", "restore", "reset", "rebase", "merge", "fetch",
    "pull", "push", "remote", "rev-parse", "stash", "tag", "blame",
    "shortlog", "describe", "reflog", "cherry-pick", "revert", "clean",
    "ls-files", "ls-remote", "submodule", "config", "init", "clone",
}

SHELL_BINS: dict[str, str] = {
    "ls":   "/bin/ls",    "pwd":  "/bin/pwd",   "cat":  "/bin/cat",
    "echo": "/bin/echo",  "find": "/usr/bin/find", "grep": "/usr/bin/grep",
    "wc":   "/usr/bin/wc", "head": "/usr/bin/head", "tail": "/usr/bin/tail",
    "stat": "/usr/bin/stat",
}
_SHELL_NAMES = ", ".join(sorted(SHELL_BINS))

PROPERTY_SAMPLE_SIZE: int | None = None

_ALL_PROPS: list[dict] = load_property_catalog(_PROPERTIES_FILE)
PROPERTIES = select_properties(_ALL_PROPS, sample_size=PROPERTY_SAMPLE_SIZE)

STATE_APS, TRANS_APS = aps_from_properties(PROPERTIES)
ALL_APS = STATE_APS + TRANS_APS

def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT_CHARS else s[:MAX_OUTPUT_CHARS] + "\n...[truncated]"

def _run(argv: list[str]) -> str:
    try:
        p = subprocess.run(argv, cwd=str(WORK_DIR), capture_output=True,
                           text=True, timeout=CMD_TIMEOUT, check=False)
        merged = "\n".join(filter(None, [(p.stdout or "").strip(), (p.stderr or "").strip()]))
        return _clip(merged or "(no output)") + f"\n[exit {p.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[error] timed out after {CMD_TIMEOUT}s"
    except Exception as e:
        return f"[error] {e}"

def _has_shell_meta(s: str) -> bool:
    return any(c in s for c in ("|", ";", "&", ">", "<", "`", "$"))

def tool_git(command: str) -> str:
    if _has_shell_meta(command):
        return "[error] shell metacharacters (|;&><`$) not allowed in git_cmd"
    try:
        parts = shlex.split(command.strip())
    except ValueError as e:
        return f"[error] {e}"
    if not parts or parts[0] not in ALLOWED_GIT:
        return f"[error] '{parts[0] if parts else ''}' not allowed"
    return _run(["git"] + parts)

def tool_shell(command: str, args: list[str] | None = None) -> str:
    if _has_shell_meta(command):
        return "[error] shell metacharacters not allowed in command name — pass separate args list"
    binary = SHELL_BINS.get(command)
    if not binary:
        return f"[error] '{command}' not allowed. Allowed: {_SHELL_NAMES}"
    return _run([binary] + [str(a) for a in (args or [])])

TOOL_IMPL = {"git_cmd": tool_git, "shell_cmd": tool_shell}

# These APs describe organizational/remote policies that have no local git
# equivalent. They are still observed for s0 through property_verifier, but
# the planner does not try to predict their changes from local git actions.

UNOBSERVABLE_APS: frozenset[str] = frozenset({
    "A maintainer-level role.",
    "A non-fast-forward update approval token is present for this action.",
    "A protected-branch classification.",
    "A protected branch requires linear history.",
    "A queue-drain or operator-intervention state is required before execution resumes.",
    "A remote write-protection or permission-denied status is active.",
    "A required immediate force push was missed.",
    "A workflow state in which an immediate force push is required.",
    "An authorized/authenticated status allowing the guarded action.",
    "The action is executed under an approved protected-branch override policy.",
    "The actor is authenticated and authorized for force-push on the target branch.",
    "The current depth of pending git workflow operations in the processing queue exceeds 64.",
    "The number of retry attempts used by the workflow operation exceeds 3.",
    "The repository workflow state indicates that a force push is required before normal publishing can continue.",
    "A network status indicating remote connectivity is available.",
    "Authentication credentials for remote write are valid.",
})

# These APs are structural invariants guaranteed by git's object model or
# desired liveness goals. The planner does not try to predict their changes
# from local git actions.
ALWAYS_TRUE_APS: frozenset[str] = frozenset({
    "The commit graph remains acyclic and does not introduce reference cycles.",
    "The repository repeatedly returns to a clean synchronized state over time.",
    "The workflow repeatedly reaches a state with no unpublished local commit debt.",
    "Those detached commits are eventually anchored to a named branch reference.",
})

OBSERVABLE_APS  = [ap for ap in ALL_APS
                   if ap not in UNOBSERVABLE_APS and ap not in ALWAYS_TRUE_APS]

_LLM_LOG: list[dict] = []

def _llm_log_reset() -> None:
    _LLM_LOG.clear()

def _llm_log_snapshot() -> list[dict]:
    return list(_LLM_LOG)

_CLIENT: openai.OpenAI | None = None

def _get_client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
    return _CLIENT

def _make_git_property_verifier(model: str) -> GitPropertyVerifier:
    return GitPropertyVerifier(
        str(WORK_DIR),
        model=model,
        client=_get_client(),
    )

def _llm(messages: list[dict], model: str,
         tools: list | None = None, tag: str = "") -> tuple[str, list]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    response = _get_client().chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = (msg.content or "").strip()

    tool_calls = []
    for tc in msg.tool_calls or []:
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append({
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": arguments,
            },
        })

    _LLM_LOG.append({
        "tag":             tag,
        "messages_in":     messages,
        "content_out":     content,
        "tool_calls_out":  tool_calls,
    })
    return content, tool_calls

def _extract_json(text: str) -> Any:
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

PROMPT_6_SYSTEM = f"""\
You are a scope guard for a git assistant agent.
The agent can only handle: git repository operations and these shell commands: {_SHELL_NAMES}.
Decide if the user's request is in scope.

Output ONLY valid JSON: {{"in_scope": true}} or {{"in_scope": false, "reason": "one sentence"}}"""

def prompt6_guard(query: str, model: str) -> tuple[bool, str]:
    content, _ = _llm([
        {"role": "system", "content": PROMPT_6_SYSTEM},
        {"role": "user",   "content": query},
    ], model, tag="6_guard")
    result = _extract_json(content)
    if isinstance(result, dict) and "in_scope" in result:
        return result["in_scope"], result.get("reason", "")
    return True, ""

def phase3_build_s0(model: str) -> dict[str, bool]:
    """
    Phase 3: observe every AP extracted from the selected property ASTs.
    """
    print(f"  \033[36m[Phase 3] Observing {len(ALL_APS)} APs extracted from properties...\033[0m")
    verifier = _make_git_property_verifier(model)
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
                             Allowed: {", ".join(sorted(ALLOWED_GIT))}
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
                             Allowed: {", ".join(sorted(ALLOWED_GIT))}
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

def prompt4a_propose(goal: str, s_current: dict[str, bool],
                     trace: list[dict], tried: list[str], model: str,
                     config: AgentConfig | None = None) -> dict | None:
    config = config or default_config_from_env()
    done_steps = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"

    content, _ = _llm([
        {"role": "system", "content": PROMPT_4A_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Steps done so far:\n{done_steps}\n\n"
            f"{_property_prompt_block(config.violation_policy)}"
            f"Already tried at this step (rejected): {tried or 'none'}\n\n"
            f"What is the next action?"},
    ], model, tag="4A_propose")

    result = _extract_json(content)
    return result if isinstance(result, dict) else None


def _normalise_proposal(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None
    action_label = str(item.get("action_label", "")).strip()
    tool = str(item.get("tool", "none")).strip() or "none"
    args = item.get("args") if isinstance(item.get("args"), dict) else {}
    if not action_label:
        command = args.get("command") if isinstance(args, dict) else None
        action_label = command.replace(" ", "_")[:40] if isinstance(command, str) else tool
    return {
        "action_label": action_label,
        "tool": tool,
        "args": args,
        "rationale": str(item.get("rationale", "")),
    }


def prompt4a_propose_batch(goal: str, s_current: dict[str, bool],
                           trace: list[dict], tried: list[str],
                           max_actions: int, model: str,
                           config: AgentConfig | None = None) -> dict | None:
    del s_current
    config = config or default_config_from_env()
    done_steps = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"

    content, _ = _llm([
        {"role": "system", "content": PROMPT_4A_BATCH_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Steps already accepted:\n{done_steps}\n\n"
            f"{_property_prompt_block(config.violation_policy)}"
            f"Rejected first actions for this planning point: {tried or 'none'}\n\n"
            f"Return at most {max_actions} remaining action(s)."},
    ], model, tag="4A_propose_batch")

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
You are predicting the value of one atomic proposition after a git action completes.
You are given the proposition, its current value, and the exact git command being run.

Reason about what the command does and whether it would change this proposition.

Output ONLY valid JSON:
{"value": true, "reason": "one sentence"}
{"value": false, "reason": "one sentence"}"""

def prompt4b_predict_ap(ap: str, current_val: bool,
                        action_label: str, tool: str, args: dict,
                        model: str) -> tuple[bool, str]:
    content, _ = _llm([
        {"role": "system", "content": PROMPT_4B_SYSTEM},
        {"role": "user",   "content":
            f"Atomic proposition: {ap}\n"
            f"Current value: {'TRUE' if current_val else 'FALSE'}\n\n"
            f"Action: {action_label}\n"
            f"Command: {tool}({json.dumps(args)})"},
    ], model, tag="4B_predict_ap")
    result = _extract_json(content)
    if isinstance(result, dict) and "value" in result:
        return bool(result["value"]), result.get("reason", "")
    return current_val, "parse error — unchanged"


def _label_is(label: str, *words: str) -> bool:
    return any(w in label.split("_") for w in words)


def _set_transition_ap_values(ap_state: dict[str, bool], action_label: str) -> None:
    al = action_label.lower()
    for ap in TRANS_APS:
        apl = ap.lower()
        if "force" in apl and "push" in apl:
            ap_state[ap] = _label_is(al, "force", "forcepush", "force_push")
        elif "rebase" in apl:
            ap_state[ap] = _label_is(al, "rebase")
        elif "merge" in apl:
            ap_state[ap] = _label_is(al, "merge")
        elif "push" in apl:
            ap_state[ap] = _label_is(al, "push") and "force" not in al
        elif "direct commit" in apl or ("commit" in apl and "rebase" not in apl and "push" not in apl):
            ap_state[ap] = _label_is(al, "commit", "stage", "add")
        elif "destructive" in apl or "rewrite" in apl:
            ap_state[ap] = _label_is(al, "force", "rebase", "reset", "amend")
        elif "mutating" in apl:
            ap_state[ap] = _label_is(al, "commit", "push", "merge", "rebase", "reset", "force")
        else:
            ap_state[ap] = False


def prompt4b_predict(action_label: str, tool: str, args: dict,
                     s_current: dict[str, bool], model: str) -> dict[str, bool]:
    """
    Predict state_after by evaluating each observable state AP independently.
    Only APs currently TRUE are checked (they might flip to FALSE).
    Transition APs are set deterministically by Python from action_label.
    """
    s_after = dict(s_current)

    for ap in OBSERVABLE_APS:
        if ap.startswith("(transition)"):
            continue
        if s_current.get(ap, False):
            new_val, _ = prompt4b_predict_ap(ap, True, action_label, tool, args, model)
            s_after[ap] = new_val

    _set_transition_ap_values(s_after, action_label)
    s_after["last_action"] = action_label
    return s_after

def _observe_execution_ap_state(verifier: GitPropertyVerifier, action_label: str) -> dict[str, bool]:
    ap_state = observe_ap_values(ALL_APS, verifier.observe_ap)
    _set_transition_ap_values(ap_state, action_label)
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


def _append_finish_node(
    planning_tree: dict,
    *,
    parent_node_id: int | None,
    depth: int,
    s_current: dict[str, bool],
    trace: list[dict],
    retry_index: int,
    proposal: dict | None = None,
) -> dict:
    node_id = len(planning_tree["nodes"])
    node = make_planning_node(
        node_id=node_id,
        parent_node_id=parent_node_id,
        depth=depth,
        state_before=dict(s_current),
        state_path=_state_path_from_trace(trace),
        action_label="goal_satisfied",
        tool="none",
        args={},
        state_after=dict(s_current),
        verification=make_verification(
            passed=True,
            properties_checked=[],
            violations=["[skipped] goal already satisfied"],
            skipped=True,
        ),
        attempts=[
            make_attempt(
                retry_index=retry_index,
                accepted=True,
                proposal=proposal or {
                    "action_label": "goal_satisfied",
                    "tool": "none",
                    "args": {},
                },
                finish=True,
            )
        ],
        result="finish",
    )
    node["outcome"] = {"finish_response": "goal already satisfied"}
    append_node(planning_tree, node, link_parent=True)
    return node


def _mark_branch_backtracked(nodes: list[dict], failure: dict) -> None:
    mark_accepted_branch_backtracked(nodes, failure)


def _check_candidate(
    *,
    proposal: dict,
    s0: dict[str, bool],
    s_current: dict[str, bool],
    trace: list[dict],
    planning_tree: dict,
    parent_node_id: int | None,
    depth: int,
    retry_index: int,
    model: str,
    config: AgentConfig,
    is_complete_trace: bool = False,
) -> tuple[dict, dict[str, bool], bool, str, dict]:
    action_label = proposal.get("action_label", "unknown")
    tool = proposal.get("tool", "none")
    args = proposal.get("args") or {}

    print(f"    → {action_label}: {tool}({json.dumps(args)})")
    s_after = prompt4b_predict(action_label, tool, args, s_current, model)
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

    verif = make_verification(
        passed=passed,
        properties_checked=verification_result.get("properties_checked", []),
        violations=violations_list,
        tla_spec=tla_spec,
        skipped=bool(verification_result.get("tlc_result", {}).get("skipped")),
    )
    node_result = "accepted" if passed else "rejected"
    if not passed and config.violation_policy in NONBLOCKING_VIOLATION_POLICIES:
        node_result = "accepted_with_ignored_violations"
    failure_feedback = None
    if not passed:
        failure_feedback = {
            "type": "tla_property_violation",
            "violations": violations_list,
        }
    accepted_by_policy = passed or config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
    node = make_planning_node(
        node_id=len(planning_tree["nodes"]),
        parent_node_id=parent_node_id,
        depth=depth,
        state_path=_state_path_from_trace(trace),
        action_label=action_label,
        tool=tool,
        args=args,
        state_before=dict(s_current),
        state_after=s_after,
        verification=verif,
        attempts=[
            make_attempt(
                retry_index=retry_index,
                accepted=accepted_by_policy,
                proposal=proposal,
                predicted_state_after=s_after,
                verification=verif,
                violation_policy=config.violation_policy,
                failure_feedback=failure_feedback,
                ignored_property_violation=(
                    failure_feedback
                    if failure_feedback is not None
                    and config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
                    else None
                ),
            )
        ],
        result=node_result,
    )
    append_node(planning_tree, node, link_parent=True)
    return candidate, s_after, passed, violations_str, node


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
    trace: list[dict] = []
    s_current = dict(s0)
    tried_per_step: list[list[str]] = []
    planning_tree = planning_tree or make_planning_tree(
        mode="%s_%s" % (config.planning_granularity, config.violation_policy),
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=PROPERTIES,
        initial_state=s0,
    )
    current_parent_node_id: int | None = None

    def checkpoint() -> None:
        if result_record is not None:
            _checkpoint_result(result_record, result_path)

    print(
        "\n\033[35m[Phase 4] Planning with TLC verification "
        f"({config.planning_granularity}, violations={config.violation_policy}, "
        f"retries={config.max_retries})...\033[0m"
    )

    if config.planning_granularity == PLANNING_BATCH:
        return _phase4_plan_batch(
            goal,
            s0,
            model,
            config,
            trace,
            s_current,
            planning_tree,
            current_parent_node_id,
            result_record=result_record,
            result_path=result_path,
        )

    goal_done = False
    blocked = False
    attempt_budget = (
        max(1, config.max_retries)
        if config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
        else config.max_retries
    )

    for step_idx in range(config.max_plan_steps):
        while len(tried_per_step) <= step_idx:
            tried_per_step.append([])
        tried = tried_per_step[step_idx]

        if goal_done:
            print(f"  [step {step_idx+1}] skipped (goal already satisfied)")
            continue

        attempt = 0
        proposal_misses = 0
        MAX_PROPOSAL_MISSES = 6
        accepted_this_step = False

        while attempt < attempt_budget:
            if proposal_misses >= MAX_PROPOSAL_MISSES:
                print(f"  [step {step_idx+1}] Too many repeated/invalid proposals — stopping")
                planning_tree["failure"] = {
                    "type": "proposal_exhausted",
                    "depth": step_idx,
                    "message": "Too many repeated or invalid proposals before verification.",
                }
                checkpoint()
                blocked = True
                break

            print(f"  [step {step_idx+1} attempt {attempt+1}] proposing action...")
            proposal = prompt4a_propose(goal, s_current, trace, tried, model, config)
            if proposal is None:
                print(f"    4A parse error, skipping")
                proposal_misses += 1
                continue

            action_label = proposal.get("action_label", "unknown")
            tool         = proposal.get("tool", "none")
            args         = proposal.get("args") or {}

            if action_label == "goal_satisfied":
                print(f"  [step {step_idx+1}] Goal satisfied — remaining steps will be skipped")
                _append_finish_node(
                    planning_tree,
                    parent_node_id=current_parent_node_id,
                    depth=step_idx,
                    s_current=s_current,
                    trace=trace,
                    retry_index=attempt,
                    proposal=proposal,
                )
                checkpoint()
                goal_done = True
                accepted_this_step = True
                break

            if action_label in tried:
                print(f"    {action_label} already tried, re-asking LLM")
                proposal_misses += 1
                continue

            proposal_misses = 0
            attempt += 1
            candidate, s_after, passed, violations_str, node = _check_candidate(
                proposal=proposal,
                s0=s0,
                s_current=s_current,
                trace=trace,
                planning_tree=planning_tree,
                parent_node_id=current_parent_node_id,
                depth=step_idx,
                retry_index=attempt - 1,
                model=model,
                config=config,
            )
            checkpoint()

            if passed or config.violation_policy in NONBLOCKING_VIOLATION_POLICIES:
                if passed:
                    print(f"    \033[32mPASS\033[0m")
                elif violations_str:
                    print(f"    \033[33m{config.violation_policy.upper()} VIOLATION: {violations_str[:120]}\033[0m")
                else:
                    print(f"    \033[33m{config.violation_policy.upper()} VERIFICATION FAILURE\033[0m")
                trace.append(candidate)
                s_current = s_after
                current_parent_node_id = node["node_id"]
                accepted_this_step = True
                break

            if violations_str:
                print(f"    \033[31mFAIL: {violations_str[:120]}\033[0m")
            tried.append(action_label)

        if blocked:
            break
        if not accepted_this_step and not goal_done:
            print(f"  [step {step_idx+1}] Exhausted retries — blocking execution")
            planning_tree["failure"] = {
                "type": "max_retries",
                "depth": step_idx,
                "tried_actions": list(tried),
                "message": "Exhausted action retries while planning this step.",
            }
            checkpoint()
            blocked = True
            break

    feasible = (goal_done or bool(trace)) and not blocked
    return trace, planning_tree, feasible


def _phase4_plan_batch(
    goal: str,
    s0: dict[str, bool],
    model: str,
    config: AgentConfig,
    trace: list[dict],
    s_current: dict[str, bool],
    planning_tree: dict,
    current_parent_node_id: int | None,
    *,
    result_record: dict[str, Any] | None = None,
    result_path: str | None = None,
) -> tuple[list[dict], dict, bool]:
    tried: list[str] = []
    attempt_budget = (
        max(1, config.max_retries)
        if config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
        else config.max_retries
    )

    def checkpoint() -> None:
        if result_record is not None:
            _checkpoint_result(result_record, result_path)

    for attempt in range(attempt_budget):
        remaining = config.max_plan_steps - len(trace)
        if remaining <= 0:
            return trace, planning_tree, bool(trace)

        print(f"  [batch attempt {attempt+1}] proposing up to {remaining} action(s)...")
        bundle = prompt4a_propose_batch(goal, s_current, trace, tried, remaining, model, config)
        if bundle is None:
            print("    4A batch parse error")
            continue

        plan = bundle.get("plan", [])
        if not plan:
            print("  [batch] Goal satisfied or empty plan returned")
            _append_finish_node(
                planning_tree,
                parent_node_id=current_parent_node_id,
                depth=len(trace),
                s_current=s_current,
                trace=trace,
                retry_index=attempt,
                proposal={
                    "action_label": "goal_satisfied",
                    "tool": "none",
                    "args": {},
                    "finish_response": bundle.get("finish_response", ""),
                },
            )
            checkpoint()
            return trace, planning_tree, True

        batch_trace = list(trace)
        batch_state = dict(s_current)
        batch_parent_node_id = current_parent_node_id
        batch_nodes: list[dict] = []
        batch_ok = True
        failed_label = ""

        for offset, proposal in enumerate(plan):
            action_label = proposal.get("action_label", "unknown")
            if action_label == "goal_satisfied":
                print("  [batch] Goal satisfied")
                _append_finish_node(
                    planning_tree,
                    parent_node_id=batch_parent_node_id,
                    depth=len(batch_trace),
                    s_current=batch_state,
                    trace=batch_trace,
                    retry_index=attempt,
                    proposal=proposal,
                )
                checkpoint()
                return batch_trace, planning_tree, True

            candidate, s_after, passed, violations_str, node = _check_candidate(
                proposal=proposal,
                s0=s0,
                s_current=batch_state,
                trace=batch_trace,
                planning_tree=planning_tree,
                parent_node_id=batch_parent_node_id,
                depth=len(batch_trace),
                retry_index=attempt,
                model=model,
                config=config,
                is_complete_trace=(offset == len(plan) - 1),
            )
            batch_nodes.append(node)
            checkpoint()

            if passed or config.violation_policy in NONBLOCKING_VIOLATION_POLICIES:
                if passed:
                    print(f"    \033[32mPASS\033[0m")
                elif violations_str:
                    print(f"    \033[33m{config.violation_policy.upper()} VIOLATION: {violations_str[:120]}\033[0m")
                else:
                    print(f"    \033[33m{config.violation_policy.upper()} VERIFICATION FAILURE\033[0m")
                batch_trace.append(candidate)
                batch_state = s_after
                batch_parent_node_id = node["node_id"]
                continue

            if violations_str:
                print(f"    \033[31mFAIL: {violations_str[:120]}\033[0m")
            batch_ok = False
            failed_label = action_label
            break

        if batch_ok:
            return batch_trace, planning_tree, True

        if failed_label:
            tried.append(failed_label)
            _mark_branch_backtracked(
                batch_nodes,
                {
                    "type": "batch_suffix_failed",
                    "batch_attempt": attempt,
                    "failed_action": failed_label,
                    "message": "A later action in this batch failed verification.",
                },
            )
            checkpoint()

    print("  [batch] Exhausted retries — blocking execution")
    planning_tree["failure"] = {
        "type": "max_retries",
        "tried_actions": list(tried),
        "message": "Exhausted batch planning retries.",
    }
    checkpoint()
    return trace, planning_tree, False

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
    verifier = _make_git_property_verifier(model)
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
        if tla_result["passed"]:
            print("  \033[32mexecution verification PASS\033[0m")
        else:
            violations = "; ".join(tla_result.get("tlc_result", {}).get("violations", []))
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
    content, _ = _llm([
        {"role": "system", "content": PROMPT_5_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\nExecuted steps:\n{trace_text}"},
    ], model, tag="5_summary")
    return content

PROMPT_7_SYSTEM = """\
You are explaining to a user why their git request could not be safely executed.
The plan was blocked — either every candidate action violated a safety property,
or retries were exhausted.
Be direct and specific: name the property that blocked the plan, and suggest a safe alternative if one exists."""

def prompt7_blocked(goal: str, s0: dict[str, bool],
                    tried_actions: list[str], model: str) -> str:
    s0_text = "\n".join(f"  {'T' if v else 'F'}  {ap}" for ap, v in s0.items())
    content, _ = _llm([
        {"role": "system", "content": PROMPT_7_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Initial state:\n{s0_text}\n\n"
            f"Actions tried and rejected: {tried_actions}"},
    ], model, tag="7_blocked")
    return content

def handle_query(goal: str, model: str,
                 config: AgentConfig | None = None) -> tuple[str, dict]:
    """Run one query. Returns (response_text, session_turn_dict)."""
    _llm_log_reset()
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
        },
    )
    turn["planning_tree"] = make_planning_tree(
        mode=planning_mode,
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=PROPERTIES,
    )
    turn["status"] = "planning"
    result_path = _start_result_session(turn)

    in_scope, reason = prompt6_guard(goal, model)
    if not in_scope:
        response = f"Out of scope: {reason}"
        turn["status"]        = "out_of_scope"
        turn["final_message"] = response
        turn["llm_log"]       = _llm_log_snapshot()
        turn["planning_tree"]["tree_summary"] = build_tree_summary(turn["planning_tree"])
        result_path = _write_result(turn, result_path)
        return _append_result_notice(response, result_path), turn

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
        tried_all = [
            n["action"]["label"]
            for n in planning_tree["nodes"]
            if n.get("result") == "rejected"
        ]
        response   = prompt7_blocked(goal, s0, tried_all, model)
        turn["status"] = "infeasible"

    planning_tree["tree_summary"] = build_tree_summary(planning_tree)
    turn["final_message"] = response
    turn["llm_log"]       = _llm_log_snapshot()
    result_path = _write_result(turn, result_path)
    return _append_result_notice(response, result_path), turn

def save_session(turns: list[dict], model: str,
                 config: AgentConfig | None = None) -> Path:
    """Compatibility wrapper: persist the latest turn using SHRDLU result files."""
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

HELP_TEXT = """\
git-agent-fsm commands:
  /help          show this message
  /props         show property counts
  /model <name>  switch OpenAI model (current: {model})
  /config        show planning config
  /planning-mode <fsm|plan|advisory>
  /planning <step|batch>
  /violations <retry|ignore|advisory>
  /retries <n>
  /cwd           show working directory
  /exit  /quit   exit

Each query runs: guard → Phase 3 (observe s0) → Phase 4 (plan+TLC) → Phase 5 (execute).
"""

def _is_git_repo() -> bool:
    r = subprocess.run(["git", "rev-parse", "--git-dir"],
                       cwd=str(WORK_DIR), capture_output=True, text=True)
    return r.returncode == 0

def repl() -> None:
    model   = DEFAULT_MODEL
    config  = default_config_from_env()

    in_repo     = _is_git_repo()
    repo_notice = "" if in_repo else "  \033[33m(not a git repo)\033[0m"
    sample_note = (f"sampled {len(PROPERTIES)}/{len(_ALL_PROPS)}"
                   if PROPERTY_SAMPLE_SIZE else f"all {len(PROPERTIES)}")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    print(f"\033[1mgit-agent-fsm\033[0m  model={model}  base_url={base_url}  cwd={WORK_DIR}{repo_notice}")
    print(f"Properties: {sample_note} | {len(STATE_APS)} state APs | {len(TRANS_APS)} transition APs")
    print(
        "Planning: %s | violations=%s | retries=%d | max_steps=%d"
        % (config.planning_granularity, config.violation_policy, config.max_retries, config.max_plan_steps)
    )
    print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("\033[1mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("Bye.")
            return
        if user_input == "/help":
            print(HELP_TEXT.format(model=model))
            continue
        if user_input == "/config":
            print(
                "planning=%s | violations=%s | retries=%d | max_steps=%d\n"
                % (
                    config.planning_granularity,
                    config.violation_policy,
                    config.max_retries,
                    config.max_plan_steps,
                )
            )
            continue
        if user_input == "/props":
            print(f"{len(PROPERTIES)} properties | {len(STATE_APS)} state APs | {len(TRANS_APS)} transition APs\n")
            continue
        if user_input == "/cwd":
            print(f"{WORK_DIR}\n")
            continue
        if user_input.startswith("/planning-mode"):
            parts = user_input.split(maxsplit=1)
            planning_mode = parts[1].strip().lower() if len(parts) == 2 else ""
            try:
                mode_config = planning_mode_config(
                    planning_mode,
                    retry_default=MAX_RETRIES,
                    invalid="raise",
                )
            except ValueError:
                print("Usage: /planning-mode <fsm|plan|advisory>\n")
                continue
            config = AgentConfig(
                planning_granularity=str(mode_config["planning_granularity"]),
                violation_policy=str(mode_config["violation_policy"]),
                max_plan_steps=config.max_plan_steps,
                max_retries=int(mode_config["max_retries"]),
            )
            print(
                "Planning: %s | violations=%s | retries=%d\n"
                % (config.planning_granularity, config.violation_policy, config.max_retries)
            )
            continue
        if user_input.startswith("/planning"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Planning: {config.planning_granularity}\n")
                continue
            try:
                planning = _normalize_planning_granularity(
                    parts[1],
                    config.planning_granularity,
                    invalid="raise",
                )
            except ValueError as exc:
                print("%s\n" % exc)
                continue
            config = replace(config, planning_granularity=planning)
            print(f"Planning: {config.planning_granularity}\n")
            continue
        if user_input.startswith("/violations"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Violations: {config.violation_policy}\n")
                continue
            try:
                policy = _normalize_violation_policy(
                    parts[1],
                    config.violation_policy,
                    invalid="raise",
                )
            except ValueError as exc:
                print("%s\n" % exc)
                continue
            config = replace(config, violation_policy=policy)
            print(f"Violations: {config.violation_policy}\n")
            continue
        if user_input.startswith("/retries"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Retries: {config.max_retries}\n")
                continue
            try:
                retries = max(0, int(parts[1]))
            except ValueError:
                print("Usage: /retries <n>\n")
                continue
            config = replace(config, max_retries=retries)
            print(f"Retries: {config.max_retries}\n")
            continue
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            model = parts[1].strip() if len(parts) == 2 else model
            print(f"Model: {model}\n")
            continue
        if user_input.startswith("/"):
            print("Unknown command. Type /help.\n")
            continue

        try:
            answer, detail = handle_query(user_input, model, config)
        except openai.OpenAIError as e:
            answer  = f"[openai error] {e}"
            detail  = {"query": user_input, "response": answer, "llm_log": _llm_log_snapshot()}
        except Exception as e:
            import traceback; traceback.print_exc()
            answer  = f"[error] {e}"
            detail  = {"query": user_input, "response": answer, "llm_log": _llm_log_snapshot()}

        print(f"\n\033[1mAgent>\033[0m {answer}\n")

if __name__ == "__main__":
    repl()
