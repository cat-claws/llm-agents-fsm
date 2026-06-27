"""Shared subprocess helpers for git-agent scripts."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 4000
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
    "ls":   "/bin/ls",
    "pwd":  "/bin/pwd",
    "cat":  "/bin/cat",
    "echo": "/bin/echo",
    "find": "/usr/bin/find",
    "grep": "/usr/bin/grep",
    "wc":   "/usr/bin/wc",
    "head": "/usr/bin/head",
    "tail": "/usr/bin/tail",
    "stat": "/usr/bin/stat",
}

_SHELL_NAMES = ", ".join(sorted(SHELL_COMMANDS))


def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    return s[:MAX_OUTPUT_CHARS] + f"\n... [truncated: {len(s) - MAX_OUTPUT_CHARS} more chars]"


def _run(argv: list[str], cwd: Path | None = None) -> str:
    work = cwd or WORK_DIR
    try:
        p = subprocess.run(
            argv,
            cwd=str(work),
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


def _has_shell_meta(s: str) -> bool:
    return any(c in s for c in ("|", ";", "&", ">", "<", "`", "$"))


def tool_git(command: str, cwd: Path | None = None) -> str:
    command = command.strip()
    if not command:
        return "[error] git command cannot be empty"
    if _has_shell_meta(command):
        return "[error] shell metacharacters (|;&><`$) not allowed in git_cmd"
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
    return _run(["git"] + parts, cwd=cwd)


def tool_shell(command: str, args: list[str] | None = None, cwd: Path | None = None) -> str:
    if _has_shell_meta(command):
        return "[error] shell metacharacters not allowed in command name — pass separate args list"
    binary = SHELL_COMMANDS.get(command)
    if binary is None:
        return f"[error] '{command}' is not allowed. Allowed commands: {_SHELL_NAMES}"
    return _run([binary] + [str(a) for a in (args or [])], cwd=cwd)


def is_git_repo(path: Path | None = None) -> bool:
    p = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path or WORK_DIR),
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


TOOL_IMPL: dict[str, Any] = {
    "git_cmd": tool_git,
    "shell_cmd": tool_shell,
}
