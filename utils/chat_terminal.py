"""Shared interactive chat terminal for llm-agents-fsm agents."""

from __future__ import annotations

from dataclasses import dataclass
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
        self._print_intro()
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
                self._finish()
                return
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
