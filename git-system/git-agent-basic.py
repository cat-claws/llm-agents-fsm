#!/usr/bin/env python3
"""
git-agent — a terminal chat agent for git operations.

Launch it anywhere inside (or outside) a git repo:
    python3 /path/to/git-agent.py
    git-agent

The agent works on the directory where it is launched.

Environment variables:
    SHRDLU_OPENAI_API_KEY      API key (default: "EMPTY" for local servers)
    SHRDLU_OPENAI_BASE_URL     Base URL (default: http://127.0.0.1:30000/v1/)
    SHRDLU_OPENAI_MODEL        model name
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import openai

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.chat_terminal import ChatCommand, ChatTerminal
from utils.session import make_session, save_session as _save_session_util

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:30000/v1/"
DEFAULT_OPENAI_API_KEY = "EMPTY"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_OPENAI_TEMPERATURE = 0.2
DEFAULT_OPENAI_MAX_TOKENS = 512
DEFAULT_OPENAI_ENABLE_THINKING = True
DEFAULT_OPENAI_SEPARATE_REASONING = True

DEFAULT_MODEL = DEFAULT_OPENAI_MODEL
MAX_STEPS = 15
MAX_OUTPUT_CHARS = 6000
CMD_TIMEOUT = 20

WORK_DIR = Path.cwd().resolve()

ALLOWED_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "checkout", "switch",
    "add", "commit", "restore", "reset", "rebase", "merge", "fetch",
    "pull", "push", "remote", "rev-parse", "stash", "tag", "blame",
    "shortlog", "describe", "reflog", "cherry-pick", "revert", "clean",
    "ls-files", "ls-remote", "submodule", "config", "init", "clone",
}

SHELL_COMMANDS: dict[str, str] = {
    "ls":    "/bin/ls",
    "pwd":   "/bin/pwd",
    "cat":   "/bin/cat",
    "echo":  "/bin/echo",
    "find":  "/usr/bin/find",
    "grep":  "/usr/bin/grep",
    "wc":    "/usr/bin/wc",
    "head":  "/usr/bin/head",
    "tail":  "/usr/bin/tail",
    "stat":  "/usr/bin/stat",
}

_SHELL_NAMES = ", ".join(sorted(SHELL_COMMANDS))

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

def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    return s[:MAX_OUTPUT_CHARS] + f"\n... [truncated: {len(s) - MAX_OUTPUT_CHARS} more chars]"


def _run(argv: list[str]) -> str:
    try:
        p = subprocess.run(
            argv,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
            check=False,
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        merged = "\n".join(filter(None, [out, err]))
        return _clip(merged or "(no output)") + f"\n[exit {p.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {CMD_TIMEOUT}s"
    except Exception as e:
        return f"[error] {e}"


def tool_git(command: str) -> str:
    command = command.strip()
    if not command:
        return "[error] git command cannot be empty"
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"[error] could not parse command: {e}"
    if not parts:
        return "[error] empty git command"
    sub = parts[0]
    if sub not in ALLOWED_GIT_SUBCOMMANDS:
        allowed = ", ".join(sorted(ALLOWED_GIT_SUBCOMMANDS))
        return f"[error] subcommand '{sub}' not allowed.\nAllowed: {allowed}"
    return _run(["git"] + parts)


def tool_shell(command: str, args: list[str] | None = None) -> str:
    binary = SHELL_COMMANDS.get(command)
    if binary is None:
        allowed = ", ".join(sorted(SHELL_COMMANDS))
        return f"[error] '{command}' is not allowed. Allowed commands: {allowed}"
    argv = [binary] + [str(a) for a in (args or [])]
    return _run(argv)


TOOL_IMPL: dict[str, Any] = {
    "git_cmd": tool_git,
    "shell_cmd": tool_shell,
}

def _openai_base_url() -> str:
    return os.environ.get("SHRDLU_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def _openai_api_key() -> str:
    return os.environ.get("SHRDLU_OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)


def _openai_model() -> str:
    return os.environ.get("SHRDLU_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def _openai_temperature() -> float:
    return float(os.environ.get("SHRDLU_OPENAI_TEMPERATURE", str(DEFAULT_OPENAI_TEMPERATURE)))


def _openai_max_tokens() -> int:
    return int(os.environ.get("SHRDLU_OPENAI_MAX_TOKENS", str(DEFAULT_OPENAI_MAX_TOKENS)))


def _openai_extra_body() -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"enable_thinking": DEFAULT_OPENAI_ENABLE_THINKING},
        "separate_reasoning": DEFAULT_OPENAI_SEPARATE_REASONING,
    }


def _make_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=_openai_api_key(),
        base_url=_openai_base_url(),
    )

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
                temperature=_openai_temperature(),
                max_tokens=_openai_max_tokens(),
                extra_body=_openai_extra_body(),
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenAI-compatible chat error at %s: %s" % (_openai_base_url(), exc)
            ) from exc
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


def _is_git_repo(path: Path) -> bool:
    p = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


def repl() -> None:
    model = _openai_model()
    verbose = False
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    client = _make_client()

    in_repo = _is_git_repo(WORK_DIR)
    repo_notice = "" if in_repo else "  \033[33m(not a git repo)\033[0m"

    base_url = _openai_base_url()

    def intro_lines() -> list[str]:
        return [
            f"\033[1mgit-agent\033[0m  model={model}  base_url={base_url}  cwd={WORK_DIR}{repo_notice}",
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
            ChatCommand(("/reset",), "clear conversation history", reset_conversation),
            ChatCommand(("/save",), "save session to .git-agent-sessions/", save_now),
            ChatCommand(("/model",), "switch OpenAI model", set_model, "<name>"),
            ChatCommand(("/verbose",), "toggle verbose tool logging", toggle_verbose),
            ChatCommand(("/cwd",), "show working directory", show_cwd),
        ],
    ).run()


if __name__ == "__main__":
    repl()
