"""Terminal runners for agents that operate a live simulator service."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.chat_terminal import ChatCommand, ChatTerminal
from utils.planning_terminal import build_planning_commands, format_runtime_config

__all__ = [
    'run_agent_against_simulator',
    'run_agent_terminal',
    'print_result_conditions',
]


def run_agent_against_simulator(agent, simulator, launch_lines=None,
                                planning_retry_default: int | None = None) -> None:
    """Read agent goals from the terminal for an already running simulator."""
    intro_lines = list(launch_lines or [])
    if _has_planning_controls(agent):
        intro_lines.append(
            'Planning: %s' % format_runtime_config(
                agent.get_runtime_planning_config(retry_default=planning_retry_default)
            )
        )
    if getattr(simulator, 'base_url', None):
        intro_lines.append('Simulator URL: %s' % simulator.base_url)
    intro_lines.append('The block-world simulator must already be running.')
    try:
        simulator.snapshot()
    except Exception as exc:
        print('')
        for line in intro_lines:
            print(line)
        print('Could not connect to simulator: %s' % exc)
        return
    intro_lines.extend([
        'Enter natural-language agent goals here; watch execution in the viewer if one is running.',
        'Type /help for commands, /quit to quit.',
    ])
    run_agent_terminal(
        agent,
        simulator,
        intro_lines=intro_lines,
        planning_retry_default=planning_retry_default,
    )


def run_agent_terminal(agent, env, prompt: str = 'goal> ',
                       intro_lines=None,
                       planning_retry_default: int | None = None) -> None:
    """Run an interactive terminal loop for an agent."""
    def show_state(_args: str) -> str:
        return env.snapshot_text()

    def reset_environment(_args: str) -> str:
        env.reset()
        return 'Environment reset.'

    def show_events(_args: str) -> str:
        events = list(env.event_log())
        if not events:
            return '(no events)'
        return '\n'.join(_format_event(event) for event in events)

    def handle_goal(request: str) -> str:
        return agent.handle_user_input(request)

    def after_turn(_request: str, _response: str | None) -> None:
        result_path = (
            getattr(agent, 'last_result_path', None)
            or getattr(agent, 'last_trace_path', None)
        )
        if result_path:
            print_result_conditions(result_path)
            print('')

    commands = []
    if _has_planning_controls(agent):
        commands.extend(
            build_planning_commands(
                get_config=lambda: agent.get_runtime_planning_config(
                    retry_default=planning_retry_default,
                ),
                set_config=lambda config: agent.set_runtime_planning_config(
                    config,
                    retry_default=planning_retry_default,
                ),
                retry_default=(
                    planning_retry_default
                    if planning_retry_default is not None
                    else lambda: agent.get_runtime_planning_config().max_retries
                ),
            )
        )
    commands.extend([
        ChatCommand(('/state', 'state'), 'show simulator state', show_state),
        ChatCommand(('/reset', 'reset'), 'reset the simulator environment', reset_environment),
        ChatCommand(('/events', 'events'), 'show simulator event log', show_events),
    ])

    ChatTerminal(
        name='shrdlu-agent',
        prompt=prompt,
        message_handler=handle_goal,
        intro=intro_lines,
        help_title='shrdlu-agent commands:',
        help_footer='Everything else is sent to the agent as a natural-language goal.',
        thinking_message='Agent planning...',
        after_turn=after_turn,
        commands=commands,
    ).run()


def _has_planning_controls(agent) -> bool:
    return (
        hasattr(agent, 'get_runtime_planning_config')
        and hasattr(agent, 'set_runtime_planning_config')
    )


def _format_event(event) -> str:
    return '[{revision}] {kind}: {label} ({status}) -> {result}'.format(
        revision=event.get('revision'),
        kind=event.get('kind'),
        label=event.get('label'),
        status='OK' if event.get('ok') else 'ERROR',
        result=event.get('result'),
    )


def print_result_conditions(result_path: str, max_nodes: int = 12) -> None:
    """Print a compact summary of the latest saved result's planning tree."""
    path = Path(result_path)
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print('Result summary unavailable: %s' % exc)
        return

    print('Result: %s' % path)
    print('Status: %s' % record.get('status', 'unknown'))
    planning_tree = record.get('planning_tree')
    if isinstance(planning_tree, dict):
        _print_planning_tree(planning_tree, max_nodes=max_nodes)
        return
    _print_step_verification(record)


def print_trace_conditions(trace_path: str, max_nodes: int = 12) -> None:
    """Backward-compatible alias for older callers."""
    print_result_conditions(trace_path, max_nodes=max_nodes)


def _print_planning_tree(tree, max_nodes: int) -> None:
    print('Planning mode: %s' % tree.get('mode', 'unknown'))
    print('Feasible: %s' % tree.get('feasible'))
    plan = tree.get('accepted_plan') or []
    print('Accepted actions: %d' % len(plan))
    for index, action in enumerate(plan, start=1):
        print('  %d. %s' % (index, _format_action(action)))
    nodes = tree.get('nodes') or []
    print('Nodes explored: %d' % len(nodes))
    for node in nodes[:max_nodes]:
        attempts = node.get('attempts') or []
        print(
            '  node=%s depth=%s result=%s attempts=%d'
            % (node.get('node_id'), node.get('depth'), node.get('result'), len(attempts))
        )
        for attempt in attempts[:4]:
            _print_attempt(attempt)
    if len(nodes) > max_nodes:
        print('  ... %d more nodes omitted' % (len(nodes) - max_nodes))
    failure = tree.get('failure')
    if failure:
        print('Failure: %s' % _short_json(failure))


def _print_attempt(attempt) -> None:
    action = attempt.get('action') or attempt.get('planned_action')
    accepted = attempt.get('accepted')
    if action:
        print('    action=%s accepted=%s' % (_format_action(action), accepted))
    else:
        print('    attempt accepted=%s' % accepted)

    verification = (
        attempt.get('property_verification')
        or attempt.get('verification')
        or attempt.get('tla_verification')
    )
    if isinstance(verification, dict):
        _print_verification(verification, indent='      ')

    feedback = (
        attempt.get('failure_feedback')
        or attempt.get('failure')
        or attempt.get('child_failure')
    )
    if feedback:
        print('      feedback=%s' % _short_json(feedback))


def _print_step_verification(record) -> None:
    steps = record.get('steps') or []
    print('Executed steps: %d' % len(steps))
    for step in steps:
        action = step.get('executed_action') or step.get('planned_action')
        if action:
            print('  step=%s action=%s' % (step.get('step_index'), _format_action(action)))
        verification = step.get('property_verification')
        if isinstance(verification, dict):
            _print_verification(verification, indent='    ')


def _print_verification(verification, indent: str) -> None:
    if 'all_satisfied' in verification:
        print('%sproperties_all_satisfied=%s' % (indent, verification.get('all_satisfied')))
    if 'success' in verification:
        print('%stla_success=%s' % (indent, verification.get('success')))
    violations = verification.get('violations') or []
    if violations:
        print('%sviolations=%s' % (indent, _short_json(violations)))
    reason = verification.get('reason')
    if reason:
        print('%sreason=%s' % (indent, reason))


def _format_action(action) -> str:
    return json.dumps(action, sort_keys=True)


def _short_json(value, limit: int = 240) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit - 3] + '...'
