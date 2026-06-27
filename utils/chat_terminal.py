"""Shared interactive chat terminal for llm-agents-fsm agents."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

CommandHandler = Callable[[str], str | None]
MessageHandler = Callable[[str], str | None]
TextProvider = str | Sequence[str] | Callable[[], str | Sequence[str]]


@dataclass(frozen=True)
class ChatCommand:
    """A terminal command handled before user text is sent to an agent."""

    names: tuple[str, ...]
    description: str
    handler: CommandHandler
    argument_hint: str = ""
    show_in_help: bool = True

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("ChatCommand requires at least one name")

    @property
    def help_name(self) -> str:
        primary = self.names[0]
        if self.argument_hint:
            primary = "%s %s" % (primary, self.argument_hint)
        if len(self.names) == 1:
            return primary
        return "%s (%s)" % (primary, ", ".join(self.names[1:]))


class ChatTerminal:
    """Small reusable REPL for terminal chat agents.

    The terminal owns the mechanics that should be consistent across agents:
    prompt handling, help and exit commands, interrupt handling, command
    dispatch, response formatting, and optional per-turn hooks.
    """

    def __init__(
        self,
        *,
        name: str,
        prompt: str = "\033[1mYou>\033[0m ",
        message_handler: MessageHandler,
        commands: Iterable[ChatCommand] = (),
        intro: TextProvider | None = None,
        help_title: TextProvider | None = None,
        help_footer: TextProvider | None = None,
        response_label: str | None = "\033[1mAgent>\033[0m",
        thinking_message: TextProvider | None = None,
        after_turn: Callable[[str, str | None], None] | None = None,
        before_exit: Callable[[], str | None] | None = None,
        enable_readline: bool = True,
        history_path: str | Path | None = None,
        history_limit: int = 1000,
        exit_names: tuple[str, ...] = ("/exit", "/quit", "exit", "quit"),
        help_names: tuple[str, ...] = ("/help", "help"),
        unknown_command_message: str = "Unknown command. Type /help.",
        goodbye: str | None = "Bye.",
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.message_handler = message_handler
        self.commands = list(commands)
        self.intro = intro
        self.help_title = help_title or ("%s commands:" % name)
        self.help_footer = help_footer
        self.response_label = response_label
        self.thinking_message = thinking_message
        self.after_turn = after_turn
        self.before_exit = before_exit
        self.enable_readline = enable_readline
        self.history_path = Path(history_path) if history_path is not None else self._default_history_path(name)
        self.history_limit = max(0, int(history_limit))
        self._readline = None
        self.exit_names = tuple(name.lower() for name in exit_names)
        self.help_names = tuple(name.lower() for name in help_names)
        self.unknown_command_message = unknown_command_message
        self.goodbye = goodbye

        self._command_map: dict[str, ChatCommand] = {}
        for command in self.commands:
            for command_name in command.names:
                key = command_name.lower()
                if key in self._command_map:
                    raise ValueError("Duplicate terminal command: %s" % command_name)
                self._command_map[key] = command

    def run(self) -> None:
        """Run the interactive terminal until the user exits."""
        self._setup_readline()
        self._print_intro()
        try:
            while True:
                try:
                    raw_text = input(self.prompt)
                except EOFError:
                    self._finish(prefix="\n")
                    return
                except KeyboardInterrupt:
                    self._finish(prefix="\n")
                    return

                request = raw_text.strip()
                if not request:
                    continue

                command_name, args = self._split_command(request)
                if command_name in self.exit_names:
                    self._remove_latest_history(request)
                    self._finish()
                    return
                self._add_history(request)
                if command_name in self.help_names:
                    print(self.render_help())
                    print("")
                    continue

                command = self._command_map.get(command_name)
                if command is not None:
                    self._print_command_result(command.handler(args))
                    continue

                if request.startswith("/"):
                    print("%s\n" % self.unknown_command_message)
                    continue

                self._print_thinking_message()
                response = self.message_handler(request)
                self._print_response(response)
                if self.after_turn is not None:
                    self.after_turn(request, response)
        finally:
            self._save_readline_history()

    def render_help(self) -> str:
        rows = [
            ("/help", "show this message"),
            ("/exit, /quit, exit, quit", "exit"),
        ]
        rows.extend(
            (command.help_name, command.description)
            for command in self.commands
            if command.show_in_help
        )
        width = max(len(name) for name, _description in rows)
        lines = [*self._as_lines(self.help_title)]
        lines.extend("  %-*s  %s" % (width, name, description) for name, description in rows)
        if self.help_footer:
            lines.append("")
            lines.extend(self._as_lines(self.help_footer))
        return "\n".join(lines)

    def _print_intro(self) -> None:
        if not self.intro:
            return
        lines = list(self._as_lines(self.intro))
        for line in lines:
            print(line)
        if lines:
            print("")

    def _print_thinking_message(self) -> None:
        if not self.thinking_message:
            return
        lines = list(self._as_lines(self.thinking_message))
        if not lines:
            return
        print("")
        for line in lines:
            print(line)

    def _print_response(self, response: str | None) -> None:
        if response is None:
            return
        text = str(response).rstrip()
        if not text:
            return
        if self.response_label:
            print("\n%s %s\n" % (self.response_label, text))
        else:
            print("\n%s\n" % text)

    @staticmethod
    def _print_command_result(result: str | None) -> None:
        if result is None:
            return
        text = str(result).rstrip()
        if text:
            print(text)
        print("")

    def _finish(self, prefix: str = "") -> None:
        if prefix:
            print(prefix, end="")
        if self.before_exit is not None:
            result = self.before_exit()
            if result:
                print(str(result).rstrip())
        self._print_goodbye()

    def _print_goodbye(self) -> None:
        if self.goodbye:
            print(self.goodbye)

    def _setup_readline(self) -> None:
        if not self.enable_readline or not sys.stdin.isatty():
            return
        try:
            import readline
        except ImportError:
            return
        self._readline = readline
        try:
            readline.parse_and_bind("set editing-mode emacs")
            readline.parse_and_bind("set enable-keypad on")
            if self.history_limit:
                readline.set_history_length(self.history_limit)
            if self.history_path is not None:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                if self.history_path.exists():
                    readline.read_history_file(str(self.history_path))
        except (OSError, ValueError):
            return

    def _add_history(self, request: str) -> None:
        if self._readline is None:
            return
        try:
            current_length = self._readline.get_current_history_length()
            if current_length:
                previous = self._readline.get_history_item(current_length)
                if previous == request:
                    return
            self._readline.add_history(request)
        except (OSError, ValueError):
            return

    def _remove_latest_history(self, request: str) -> None:
        if self._readline is None:
            return
        try:
            current_length = self._readline.get_current_history_length()
            if not current_length:
                return
            latest = self._readline.get_history_item(current_length)
            if latest == request:
                self._readline.remove_history_item(current_length - 1)
        except (AttributeError, OSError, ValueError):
            return

    def _save_readline_history(self) -> None:
        if self._readline is None or self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._readline.write_history_file(str(self.history_path))
        except OSError:
            return

    @staticmethod
    def _default_history_path(name: str) -> Path:
        state_root = os.environ.get("XDG_STATE_HOME")
        root = Path(state_root) if state_root else Path.home() / ".local" / "state"
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        return root / "llm-agents-fsm" / ("%s_history" % safe_name)

    @staticmethod
    def _split_command(request: str) -> tuple[str, str]:
        parts = request.split(maxsplit=1)
        command_name = parts[0].lower()
        args = parts[1].strip() if len(parts) == 2 else ""
        return command_name, args

    @staticmethod
    def _as_lines(provider: TextProvider) -> list[str]:
        value = provider() if callable(provider) else provider
        if isinstance(value, str):
            return value.splitlines() or [""]
        return [str(line) for line in value]
