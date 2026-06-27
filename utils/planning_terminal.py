"""Shared planning-mode terminal commands."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from utils.chat_terminal import ChatCommand
from utils.planning_modes import (
    PLANNING_MODE_CHOICES_TEXT,
    PLANNING_MODE_CUSTOM,
    format_planning_config,
    infer_planning_mode,
    normalize_planning_granularity,
    normalize_violation_policy,
    planning_mode_config,
)


@dataclass(frozen=True)
class RuntimePlanningConfig:
    """Planning settings exposed by interactive FSM terminals."""

    mode: str
    planning_granularity: str
    violation_policy: str
    max_retries: int
    max_steps: int | None = None


def runtime_config_from_values(
    *,
    planning_granularity: str,
    violation_policy: str,
    max_retries: int,
    retry_default: int,
    max_steps: int | None = None,
) -> RuntimePlanningConfig:
    mode = infer_planning_mode(
        planning_granularity,
        violation_policy,
        max_retries,
        retry_default=retry_default,
    )
    return RuntimePlanningConfig(
        mode=mode,
        planning_granularity=planning_granularity,
        violation_policy=violation_policy,
        max_retries=int(max_retries),
        max_steps=max_steps,
    )


def format_runtime_config(config: RuntimePlanningConfig) -> str:
    return format_planning_config(
        mode=config.mode,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        max_retries=config.max_retries,
        max_steps=config.max_steps,
    )


def build_planning_commands(
    *,
    get_config: Callable[[], RuntimePlanningConfig],
    set_config: Callable[[RuntimePlanningConfig], RuntimePlanningConfig],
    retry_default: Callable[[], int] | int,
) -> list[ChatCommand]:
    """Build common runtime planning commands for FSM terminals."""

    def default_retries() -> int:
        return int(retry_default() if callable(retry_default) else retry_default)

    def update(next_config: RuntimePlanningConfig) -> str:
        return format_runtime_config(set_config(next_config))

    def show_config(_args: str) -> str:
        return format_runtime_config(get_config())

    def set_mode(args: str) -> str:
        mode = args.strip().lower()
        if not mode:
            return format_runtime_config(get_config())
        try:
            mode_config = planning_mode_config(
                mode,
                retry_default=default_retries(),
                invalid="raise",
            )
        except ValueError as exc:
            return str(exc)
        current = get_config()
        return update(
            RuntimePlanningConfig(
                mode=str(mode_config["mode"]),
                planning_granularity=str(mode_config["planning_granularity"]),
                violation_policy=str(mode_config["violation_policy"]),
                max_retries=int(mode_config["max_retries"]),
                max_steps=current.max_steps,
            )
        )

    def set_granularity(args: str) -> str:
        current = get_config()
        if not args:
            return format_runtime_config(current)
        try:
            granularity = normalize_planning_granularity(
                args,
                default=current.planning_granularity,
                invalid="raise",
            )
        except ValueError as exc:
            return str(exc)
        next_config = replace(
            current,
            mode=PLANNING_MODE_CUSTOM,
            planning_granularity=granularity,
        )
        return update(next_config)

    def set_violations(args: str) -> str:
        current = get_config()
        if not args:
            return format_runtime_config(current)
        try:
            policy = normalize_violation_policy(
                args,
                default=current.violation_policy,
                invalid="raise",
            )
        except ValueError as exc:
            return str(exc)
        next_config = replace(
            current,
            mode=PLANNING_MODE_CUSTOM,
            violation_policy=policy,
        )
        return update(next_config)

    def set_retries(args: str) -> str:
        current = get_config()
        if not args:
            return format_runtime_config(current)
        try:
            retries = max(0, int(args))
        except ValueError:
            return "retries must be an integer"
        next_config = replace(
            current,
            mode=PLANNING_MODE_CUSTOM,
            max_retries=retries,
        )
        return update(next_config)

    return [
        ChatCommand(("/config",), "show runtime planning config", show_config),
        ChatCommand(("/mode", "/planning-mode"), "show or set planning mode", set_mode, PLANNING_MODE_CHOICES_TEXT),
        ChatCommand(("/granularity", "/planning"), "show or set planning granularity", set_granularity, "<step|batch>"),
        ChatCommand(("/violations",), "show or set violation policy", set_violations, "<retry|ignore|advisory>"),
        ChatCommand(("/retries",), "show or set planning retries", set_retries, "<n>"),
    ]
