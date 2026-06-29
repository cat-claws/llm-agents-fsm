#!/usr/bin/env python3
"""OpenAI-compatible FSM/planning agent for git repositories."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import openai

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.property_catalog import (
    aps_from_properties,
    load_property_catalog,
    observe_ap_values,
    select_properties,
)
from utils.agent_planning import (
    AgentFlowSpec,
    default_config_from_env as planning_config_from_env,
    execute_tool_step,
    run_agent_flow,
)
from utils.chat_terminal import ChatCommand, ChatTerminal
from utils.git_learning_reset import reset_git_learning_lab
from utils.planning_terminal import (
    build_planning_commands,
    format_runtime_config,
    runtime_config_from_values,
)

from property_verifier import PropertyVerifier as GitPropertyVerifier
from git_shell_tools import (
    ALLOWED_GIT_SUBCOMMANDS, CMD_TIMEOUT, TOOL_IMPL, WORK_DIR, _SHELL_NAMES, is_git_repo,
)
from utils.llm_client import llm as _llm_call, make_client, tool_arguments

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_PLAN_STEPS   = 10
MAX_RETRIES      = 3

_RESOURCES_DIR      = Path(__file__).resolve().parent / "resources"

RESULT_DIR = (
    None
    if os.environ.get("GIT_AGENT_FSM_RESULT_DIR") == ""
    else Path(
        os.environ.get("GIT_AGENT_FSM_RESULT_DIR")
        or (_REPO_ROOT.parent / "playground-llm-agents-fsm" / "results")
    )
)


PROPERTY_SAMPLE_SIZE: int | None = None

_ALL_PROPS: list[dict] = load_property_catalog(_RESOURCES_DIR / "GIT_PROPERTIES_AST.json")
PROPERTIES = select_properties(_ALL_PROPS, sample_size=PROPERTY_SAMPLE_SIZE)

ALL_APS = sorted(set(ap for part in aps_from_properties(PROPERTIES) for ap in part))

_AP_CATALOG: dict[str, Any] = json.loads(
    (_RESOURCES_DIR / "GIT_AP_CANDIDATES.json").read_text(encoding="utf-8")
)
_AP_CATALOG_METADATA: dict[str, Any] = _AP_CATALOG.get("metadata", {})
_AP_SPEC_BY_NAME: dict[str, dict[str, Any]] = {
    spec.get("name", ""): spec
    for spec in _AP_CATALOG.get("current_state_aps", [])
    if spec.get("name")
}

DEFAULT_OPENAI_PLANNING_MAX_TOKENS = 2048

_CLIENT = make_client()



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

Call propose_git_action with:
{{
  "action_label": "short_snake_case_label",
  "tool": "git_cmd | shell_cmd | none",
  "args": {{"command": "..."}},
  "rationale": "one sentence"
}}
If the goal is already achieved, call propose_git_action with {{"action_label": "goal_satisfied", "tool": "none", "args": {{}}, "rationale": "done"}}"""

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

Call propose_git_plan with:
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
If the goal is already achieved, call propose_git_plan with {{"plan": [], "finish_response": "done"}}"""

_ACTION_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_label": {"type": "string"},
        "tool": {"type": "string", "enum": ["git_cmd", "shell_cmd", "none"]},
        "args": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "command": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
        "finish_response": {"type": "string"},
    },
    "required": ["action_label", "tool", "args", "rationale"],
    "additionalProperties": False,
}

_ACTION_PROPOSAL_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_git_action",
            "description": "Return the next single git workflow action.",
            "parameters": _ACTION_PROPOSAL_SCHEMA,
        },
    }
]

_PLAN_PROPOSAL_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_git_plan",
            "description": "Return the remaining git workflow plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": _ACTION_PROPOSAL_SCHEMA,
                    },
                    "finish_response": {"type": "string"},
                },
                "required": ["plan", "finish_response"],
                "additionalProperties": False,
            },
        },
    }
]

_AP_PREDICTION_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_ap_value",
            "description": "Predict one atomic proposition value after a candidate git action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["value", "reason"],
                "additionalProperties": False,
            },
        },
    }
]

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

Call predict_ap_value with:
{"value": true, "reason": "one sentence"}
{"value": false, "reason": "one sentence"}"""

FINAL_RESPONSE_SYSTEM = """\
You are writing the final response for a git workflow agent.
If the workflow succeeded, concisely summarize what was done, what changed, and the current state.
If the workflow was blocked or infeasible, explain why it could not be safely executed, name the blocking property or failed condition when available, and suggest a safe alternative if one exists.
Be direct, specific, and concise."""


def _render_prediction_note(template: Any, **values: object) -> str:
    if not isinstance(template, str):
        return ""
    return template.format(**values)


def _command_prediction_notes(tool: str, args: dict) -> str:
    command = str(args.get("command", "") if isinstance(args, dict) else "").strip()
    note_specs = _AP_CATALOG_METADATA.get("tool_prediction_notes", {})
    notes: list[str] = []
    if tool == "shell_cmd":
        shell_notes = note_specs.get("shell_cmd", {})
        notes.append(_render_prediction_note(shell_notes.get("base"), allowed_programs=_SHELL_NAMES))
        if command == "git":
            notes.append(_render_prediction_note(shell_notes.get("git_denied")))
    elif tool == "git_cmd":
        git_notes = note_specs.get("git_cmd", {})
        notes.append(_render_prediction_note(git_notes.get("base")))
        subcommand_notes = git_notes.get("subcommands", {})
        subcommand = command.split()[0] if command else ""
        subcommand_note = _render_prediction_note(
            subcommand_notes.get(subcommand),
            subcommand=subcommand,
        )
        if subcommand_note:
            notes.append(subcommand_note)
        if subcommand == "rebase":
            if "-i" in command.split():
                notes.append(_render_prediction_note(subcommand_notes.get("rebase_interactive")))
    elif tool == "none":
        notes.append(_render_prediction_note(note_specs.get("none")))
    else:
        notes.append(_render_prediction_note(note_specs.get("unknown")))
    return "\n".join(f"- {note}" for note in notes if note)


def _format_final_trace(trace: list[dict], exec_results: list[str]) -> str:
    if not trace:
        return "(none)"
    lines = []
    for i, step in enumerate(trace, 1):
        result = exec_results[i - 1][:200] if i <= len(exec_results) else "N/A"
        lines.append(
            f"  {i}. {step['action_label']}: {step['tool']}({json.dumps(step['args'])})"
            f"\n     result: {result}"
        )
    return "\n".join(lines)


def _format_final_state(state: dict[str, bool] | None) -> str:
    if not state:
        return "(not provided)"
    return "\n".join(
        f"  {'T' if value else 'F'}  {ap}"
        for ap, value in state.items()
    )


def _final_response_text(
    *,
    goal: str,
    status: str,
    model: str,
    trace_text: str = "(none)",
    exec_results_text: str = "(none)",
    initial_state_text: str = "(not provided)",
    tried_actions_text: str = "(none)",
    blocking_feedback: str = "(none)",
    llm_call,
    client,
) -> str:
    content, _tool_calls = llm_call(client, [
        {"role": "system", "content": FINAL_RESPONSE_SYSTEM},
        {"role": "user", "content":
            f"Goal: {goal}\n"
            f"Status: {status}\n\n"
            f"Executed steps:\n{trace_text}\n\n"
            f"Execution results:\n{exec_results_text}\n\n"
            f"Initial state:\n{initial_state_text}\n\n"
            f"Actions tried and rejected:\n{tried_actions_text}\n\n"
            f"Blocking feedback:\n{blocking_feedback}"},
    ], model)
    return content.strip()


def summarize_final_response(
    goal: str,
    trace: list[dict],
    exec_results: list[str],
    model: str,
    *,
    llm_call,
    client,
) -> str:
    exec_results_text = "\n".join(
        f"  {i}. {result[:200]}"
        for i, result in enumerate(exec_results, 1)
    ) or "(none)"
    return _final_response_text(
        goal=goal,
        status="success",
        model=model,
        trace_text=_format_final_trace(trace, exec_results),
        exec_results_text=exec_results_text,
        llm_call=llm_call,
        client=client,
    )


def explain_blocked_final_response(
    goal: str,
    initial_state: dict[str, bool],
    tried_actions: list[str],
    model: str,
    *,
    llm_call,
    client,
) -> str:
    return _final_response_text(
        goal=goal,
        status="blocked",
        model=model,
        initial_state_text=_format_final_state(initial_state),
        tried_actions_text=json.dumps(tried_actions),
        blocking_feedback=(
            "No feasible property-satisfying plan was found after the listed actions "
            "were rejected or retries were exhausted."
        ),
        llm_call=llm_call,
        client=client,
    ) or "No feasible property-satisfying plan found."


def _set_runtime_config(state: dict[str, Any], next_config):
    state["runtime_config"] = next_config
    state["config"] = replace(
        state["config"],
        planning_granularity=next_config.planning_granularity,
        violation_policy=next_config.violation_policy,
        max_retries=next_config.max_retries,
    )
    return next_config


def _handle_message(state: dict[str, Any], spec: AgentFlowSpec, user_input: str) -> str:
    try:
        answer, _detail = run_agent_flow(
            goal=user_input,
            model=str(state["model"]),
            config=state["config"],
            spec=spec,
        )
    except openai.OpenAIError as e:
        answer = f"[openai error] {e}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = f"[error] {e}"
    return answer


def _set_model(state: dict[str, Any], args: str) -> str:
    if args:
        state["model"] = args.strip()
    return "Model: %s" % state["model"]


def repl() -> None:
    config = planning_config_from_env(
        env_var="GIT_AGENT_FSM_PLANNING_MODE",
        retry_default=MAX_RETRIES,
        max_plan_steps=MAX_PLAN_STEPS,
    )
    spec = AgentFlowSpec(
        agent="git-agent-fsm",
        domain="git",
        work_dir=str(WORK_DIR),
        properties=PROPERTIES,
        aps=ALL_APS,
        result_dir=RESULT_DIR,
        verification_module_name="GitTrace",
        verification_timeout=CMD_TIMEOUT,
        observe_initial_state=phase3_build_s0,
        execute_step=lambda step: execute_tool_step(step, tool_impl=TOOL_IMPL),
        summarize_result=partial(
            summarize_final_response,
            llm_call=_llm_call,
            client=_CLIENT,
        ),
        explain_blocked=partial(
            explain_blocked_final_response,
            llm_call=_llm_call,
            client=_CLIENT,
        ),
        llm_call=_llm_call,
        client=_CLIENT,
        tool_arguments=tool_arguments,
        max_planning_tokens=DEFAULT_OPENAI_PLANNING_MAX_TOKENS,
        propose_step_prompt=PROMPT_4A_SYSTEM,
        propose_batch_prompt=PROMPT_4A_BATCH_SYSTEM,
        predict_ap_prompt=PROMPT_4B_SYSTEM,
        action_proposal_tool=_ACTION_PROPOSAL_TOOL,
        action_proposal_tool_name="propose_git_action",
        plan_proposal_tool=_PLAN_PROPOSAL_TOOL,
        plan_proposal_tool_name="propose_git_plan",
        ap_prediction_tool=_AP_PREDICTION_TOOL,
        ap_prediction_tool_name="predict_ap_value",
        ap_spec_by_name=_AP_SPEC_BY_NAME,
        ap_catalog_metadata=_AP_CATALOG_METADATA,
        ap_evidence_field="git_commands",
        action_prediction_notes=_command_prediction_notes,
        observe_execution_ap=lambda model: GitPropertyVerifier(
            str(WORK_DIR),
            model=model,
            client=_CLIENT,
        ).observe_ap,
        already_satisfied_response=(
            "The request appears already satisfied; no git action was executed."
        ),
    )
    state = {
        "model": os.environ.get("SHRDLU_OPENAI_MODEL", DEFAULT_MODEL),
        "config": config,
        "runtime_config": runtime_config_from_values(
            planning_granularity=config.planning_granularity,
            violation_policy=config.violation_policy,
            max_retries=config.max_retries,
            retry_default=MAX_RETRIES,
            max_steps=config.max_plan_steps,
        ),
    }
    repo_notice = "" if is_git_repo() else "  \033[33m(not a git repo)\033[0m"
    sample_note = (f"sampled {len(PROPERTIES)}/{len(_ALL_PROPS)}"
                   if PROPERTY_SAMPLE_SIZE else f"all {len(PROPERTIES)}")

    ChatTerminal(
        name="git-agent-fsm",
        message_handler=partial(_handle_message, state, spec),
        intro=lambda: [
            f"\033[1mgit-agent-fsm\033[0m  model={state['model']}  cwd={WORK_DIR}{repo_notice}",
            f"Properties: {sample_note} | {len(ALL_APS)} APs",
            "Planning: %s" % format_runtime_config(state["runtime_config"]),
            "Type /help for commands, /exit to quit.",
        ],
        help_title=lambda: "git-agent-fsm commands (model=%s):" % state["model"],
        help_footer="Each query runs: Phase 3 (observe s0) -> Phase 4 (plan+TLC) -> Phase 5 (execute).",
        commands=[
            ChatCommand(
                ("/reset", "reset"),
                "reset git-learning-lab from parent installer",
                lambda _args: reset_git_learning_lab(WORK_DIR).message,
            ),
            ChatCommand(
                ("/props",),
                "show property counts",
                lambda _args: "%d properties | %d APs" % (len(PROPERTIES), len(ALL_APS)),
            ),
            ChatCommand(("/model",), "switch OpenAI model", partial(_set_model, state), "<name>"),
            *build_planning_commands(
                get_config=lambda: state["runtime_config"],
                set_config=partial(_set_runtime_config, state),
                retry_default=MAX_RETRIES,
            ),
            ChatCommand(("/cwd",), "show working directory", lambda _args: str(WORK_DIR)),
        ],
    ).run()

if __name__ == "__main__":
    repl()
