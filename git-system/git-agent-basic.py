#!/usr/bin/env python3
"""
git-agent — a terminal chat agent for git operations.

Launch it anywhere inside (or outside) a git repo:
    python3 /path/to/git-agent.py
    git-agent

The agent works on the directory where it is launched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import openai

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.chat_terminal import ChatCommand, ChatTerminal
from utils.git_learning_reset import reset_git_learning_lab
from utils.llm_client import extra_body, make_client
from utils.session import make_session, save_session as _save_session_util
from git_shell_tools import (
    SHELL_COMMANDS, TOOL_IMPL, WORK_DIR, _SHELL_NAMES,
    is_git_repo, tool_git, tool_shell,
)

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_STEPS = 15
MAX_OUTPUT_CHARS = 6000
CMD_TIMEOUT = 20

WORK_DIR = Path.cwd().resolve()

SYSTEM = f"""You are a git assistant agent running in a terminal.
Working directory: {WORK_DIR}

You have two tools: git_cmd and shell_cmd.

shell_cmd is restricted to exactly these commands: {_SHELL_NAMES}
No other commands exist. Do not attempt any command not in that list.

CRITICAL RULES — never break these:
1. ALWAYS call a tool to answer questions. Never answer from memory or prior tool results — state changes between questions.
2. ALWAYS call a tool to perform actions. Never tell the user to run a command themselves — execute it directly.
3. After performing an action, verify the result with a follow-up tool call, then report what changed.
5. If the user asks for anything outside git operations or the allowed shell commands ({_SHELL_NAMES}), immediately refuse without calling any tool. Say clearly: "That is outside my scope. I can only help with git operations and these shell commands: {_SHELL_NAMES}."
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "git_cmd",
            "description": (
                "Run a git subcommand in the working directory. "
                "Pass the subcommand and its arguments as a single string, "
                "e.g. 'status -sb' or 'log --oneline -10'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Git subcommand + args (no leading 'git'), e.g. 'log --oneline -5'",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_cmd",
            "description": (
                f"Run one of the allowed shell commands: {_SHELL_NAMES}. "
                "Pass the command name and an optional list of argument strings. "
                "No other commands are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": f"Command name — must be one of: {_SHELL_NAMES}",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argument list, e.g. [\"-la\"] for ls or [\"-n\", \"20\", \"file.txt\"] for head",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

def run_turn(messages: list[dict], user_query: str, model: str, verbose: bool,
             client: openai.OpenAI) -> str:
    messages.append({"role": "user", "content": user_query})

    for _ in range(MAX_STEPS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0.2,
                max_tokens=512,
                extra_body=extra_body(),
            )
        except Exception as exc:
            raise RuntimeError("OpenAI-compatible chat error: %s" % exc) from exc
        try:
            msg = resp.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenAI-compatible response: %r" % resp) from exc
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            answer = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": answer})
            return answer

        messages.append(msg.model_dump(exclude_unset=False))

        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if verbose:
                print(f"\033[33m[tool] {fn}({json.dumps(args, ensure_ascii=False)})\033[0m")

            impl = TOOL_IMPL.get(fn)
            if impl is None:
                result = f"[error] unknown tool '{fn}'"
            else:
                try:
                    result = impl(**args)
                except TypeError as e:
                    result = f"[error] bad arguments for {fn}: {e}"
                except Exception as e:
                    result = f"[error] {fn} raised: {e}"

            if verbose:
                preview = result[:300].replace("\n", " ")
                print(f"\033[2m  → {preview}{'...' if len(result) > 300 else ''}\033[0m")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return f"[agent] stopped after {MAX_STEPS} steps without a final answer."

SESSIONS_DIR = WORK_DIR / ".git-agent-sessions"

def save_session(messages: list[dict], model: str) -> Path:
    session = make_session(
        agent="git-agent",
        model=model,
        domain="git",
        request=next((m["content"] for m in messages if m["role"] == "user"), ""),
        work_dir=str(WORK_DIR),
    )
    session["status"]   = "finished"
    session["messages"] = messages
    return _save_session_util(session, SESSIONS_DIR, filename_prefix="session")


def repl() -> None:
    model = os.environ.get("SHRDLU_OPENAI_MODEL", DEFAULT_MODEL)
    verbose = False
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    client = make_client()

    in_repo = is_git_repo(WORK_DIR)
    repo_notice = "" if in_repo else "  \033[33m(not a git repo)\033[0m"

    def intro_lines() -> list[str]:
        return [
            f"\033[1mgit-agent\033[0m  model={model}  cwd={WORK_DIR}{repo_notice}",
            "Type /help for commands, /exit to quit.",
        ]

    def help_title() -> str:
        return "git-agent commands (model=%s, verbose=%s):" % (model, verbose)

    def save_if_needed() -> str | None:
        if len(messages) <= 1:
            return None
        path = save_session(messages, model)
        return "Session saved -> %s" % path

    def save_now(_args: str) -> str:
        if len(messages) <= 1:
            return "Nothing to save yet."
        path = save_session(messages, model)
        return "Saved -> %s" % path

    def reset_conversation(_args: str) -> str:
        nonlocal messages
        lines = []
        if len(messages) > 1:
            path = save_session(messages, model)
            lines.append("Session saved -> %s" % path)
        messages = [{"role": "system", "content": SYSTEM}]
        lines.append("Conversation reset.")
        return "\n".join(lines)

    def reset_lab(_args: str) -> str:
        nonlocal messages
        result = reset_git_learning_lab(WORK_DIR)
        if result.ok:
            messages = [{"role": "system", "content": SYSTEM}]
            return result.message + "\nConversation reset."
        return result.message

    def show_cwd(_args: str) -> str:
        return str(WORK_DIR)

    def toggle_verbose(_args: str) -> str:
        nonlocal verbose
        verbose = not verbose
        return "Verbose: %s" % ("on" if verbose else "off")

    def set_model(args: str) -> str:
        nonlocal model
        if args:
            model = args.strip()
            return "Model set to: %s" % model
        return "Current model: %s" % model

    def handle_message(user_input: str) -> str:
        try:
            answer = run_turn(messages, user_input, model, verbose, client)
        except openai.OpenAIError as e:
            return f"\033[31m[openai error] {e}\033[0m"
        except Exception as e:
            return f"\033[31m[error] {e}\033[0m"
        return answer

    ChatTerminal(
        name="git-agent",
        message_handler=handle_message,
        intro=intro_lines,
        help_title=help_title,
        help_footer="Everything else is sent to the agent.",
        before_exit=save_if_needed,
        commands=[
            ChatCommand(("/reset", "reset"), "reset git-learning-lab from parent installer", reset_lab),
            ChatCommand(("/chat-reset",), "clear conversation history", reset_conversation),
            ChatCommand(("/save",), "save session to .git-agent-sessions/", save_now),
            ChatCommand(("/model",), "switch OpenAI model", set_model, "<name>"),
            ChatCommand(("/verbose",), "toggle verbose tool logging", toggle_verbose),
            ChatCommand(("/cwd",), "show working directory", show_cwd),
        ],
    ).run()


if __name__ == "__main__":
    repl()
