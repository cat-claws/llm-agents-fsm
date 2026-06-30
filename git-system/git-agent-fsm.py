#!/usr/bin/env python3
"""OpenAI-compatible FSM/planning agent for git repositories."""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

_script_dir = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils.property_catalog import (
    aps_from_properties,
    load_property_catalog,
)
from utils.agent_planning import (
    AgentFlowSpec,
    default_config_from_env as planning_config_from_env,
    execute_tool_step,
    handle_agent_flow_message,
    make_final_response_handlers,
    render_template_note,
    set_runtime_config_in_state,
)
from utils.chat_terminal import ChatCommand, ChatTerminal, set_state_model
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

_resources_dir = Path(__file__).resolve().parent / "resources"
_result_dir = _repo_root.parent / "playground-llm-agents-fsm" / "results"


properties: list[dict] = load_property_catalog(_resources_dir / "GIT_PROPERTIES_AST.json")

all_aps = sorted(set(ap for part in aps_from_properties(properties) for ap in part))

_ap_catalog: dict[str, Any] = json.loads(
    (_resources_dir / "GIT_AP_CANDIDATES.json").read_text(encoding="utf-8")
)
_ap_catalog_metadata: dict[str, Any] = _ap_catalog.get("metadata", {})
_ap_spec_by_name: dict[str, dict[str, Any]] = {
    spec.get("name", ""): spec
    for spec in _ap_catalog.get("current_state_aps", [])
    if spec.get("name")
}

DEFAULT_OPENAI_PLANNING_MAX_TOKENS = 2048

_client = make_client()

SINGLE_ACTION_PROPOSAL_SYSTEM_PROMPT = f"""\
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

ACTION_PLAN_PROPOSAL_SYSTEM_PROMPT = f"""\
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

_action_proposal_schema: dict[str, Any] = {
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

_action_proposal_tool: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_git_action",
            "description": "Return the next single git workflow action.",
            "parameters": _action_proposal_schema,
        },
    }
]

_plan_proposal_tool: list[dict[str, Any]] = [
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
                        "items": _action_proposal_schema,
                    },
                    "finish_response": {"type": "string"},
                },
                "required": ["plan", "finish_response"],
                "additionalProperties": False,
            },
        },
    }
]

_ap_prediction_tool: list[dict[str, Any]] = [
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

AP_PREDICTION_AFTER_ACTION_SYSTEM_PROMPT = """\
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


def _command_prediction_notes(tool: str, args: dict) -> str:
    command = str(args.get("command", "") if isinstance(args, dict) else "").strip()
    note_specs = _ap_catalog_metadata.get("tool_prediction_notes", {})
    notes: list[str] = []
    if tool == "shell_cmd":
        shell_notes = note_specs.get("shell_cmd", {})
        notes.append(render_template_note(shell_notes.get("base"), allowed_programs=_SHELL_NAMES))
        if command == "git":
            notes.append(render_template_note(shell_notes.get("git_denied")))
    elif tool == "git_cmd":
        git_notes = note_specs.get("git_cmd", {})
        notes.append(render_template_note(git_notes.get("base")))
        subcommand_notes = git_notes.get("subcommands", {})
        subcommand = command.split()[0] if command else ""
        subcommand_note = render_template_note(
            subcommand_notes.get(subcommand),
            subcommand=subcommand,
        )
        if subcommand_note:
            notes.append(subcommand_note)
        if subcommand == "rebase":
            if "-i" in command.split():
                notes.append(render_template_note(subcommand_notes.get("rebase_interactive")))
    elif tool == "none":
        notes.append(render_template_note(note_specs.get("none")))
    else:
        notes.append(render_template_note(note_specs.get("unknown")))
    return "\n".join(f"- {note}" for note in notes if note)


def repl() -> None:
    config = planning_config_from_env(
        env_var="GIT_AGENT_FSM_PLANNING_MODE",
        retry_default=MAX_RETRIES,
        max_plan_steps=MAX_PLAN_STEPS,
    )
    summarize_result, explain_blocked = make_final_response_handlers(
        system_prompt=FINAL_RESPONSE_SYSTEM,
        llm_call=_llm_call,
        client=_client,
    )
    spec = AgentFlowSpec(
        agent="git-agent-fsm",
        domain="git",
        work_dir=str(WORK_DIR),
        properties=properties,
        aps=all_aps,
        result_dir=_result_dir,
        verification_module_name="GitTrace",
        verification_timeout=CMD_TIMEOUT,
        observe_ap_for_model=lambda model: GitPropertyVerifier(
            str(WORK_DIR),
            model=model,
            client=_client,
        ).observe_ap,
        execute_step=lambda step: execute_tool_step(step, tool_impl=TOOL_IMPL),
        summarize_result=summarize_result,
        explain_blocked=explain_blocked,
        llm_call=_llm_call,
        client=_client,
        tool_arguments=tool_arguments,
        max_planning_tokens=DEFAULT_OPENAI_PLANNING_MAX_TOKENS,
        propose_step_prompt=SINGLE_ACTION_PROPOSAL_SYSTEM_PROMPT,
        propose_batch_prompt=ACTION_PLAN_PROPOSAL_SYSTEM_PROMPT,
        predict_ap_prompt=AP_PREDICTION_AFTER_ACTION_SYSTEM_PROMPT,
        action_proposal_tool=_action_proposal_tool,
        action_proposal_tool_name="propose_git_action",
        plan_proposal_tool=_plan_proposal_tool,
        plan_proposal_tool_name="propose_git_plan",
        ap_prediction_tool=_ap_prediction_tool,
        ap_prediction_tool_name="predict_ap_value",
        ap_spec_by_name=_ap_spec_by_name,
        ap_catalog_metadata=_ap_catalog_metadata,
        ap_evidence_field="git_commands",
        action_prediction_notes=_command_prediction_notes,
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
    sample_note = f"all {len(properties)}"

    ChatTerminal(
        name="git-agent-fsm",
        message_handler=partial(handle_agent_flow_message, state, spec),
        intro=lambda: [
            f"\033[1mgit-agent-fsm\033[0m  model={state['model']}  cwd={WORK_DIR}{repo_notice}",
            f"Properties: {sample_note} | {len(all_aps)} APs",
            "Planning: %s" % format_runtime_config(state["runtime_config"]),
            "Type /help for commands, /exit to quit.",
        ],
        help_title=lambda: "git-agent-fsm commands (model=%s):" % state["model"],
        help_footer="Each query runs: observe s0 -> plan+TLC -> execute.",
        commands=[
            ChatCommand(
                ("/reset", "reset"),
                "reset git-learning-lab from parent installer",
                lambda _args: reset_git_learning_lab(WORK_DIR).message,
            ),
            ChatCommand(
                ("/props",),
                "show property counts",
                lambda _args: "%d properties | %d APs" % (len(properties), len(all_aps)),
            ),
            ChatCommand(("/model",), "switch OpenAI model", partial(set_state_model, state), "<name>"),
            *build_planning_commands(
                get_config=lambda: state["runtime_config"],
                set_config=partial(set_runtime_config_in_state, state),
                retry_default=MAX_RETRIES,
            ),
            ChatCommand(("/cwd",), "show working directory", lambda _args: str(WORK_DIR)),
        ],
    ).run()

if __name__ == "__main__":
    repl()
