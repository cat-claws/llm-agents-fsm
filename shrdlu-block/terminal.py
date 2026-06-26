"""Terminal runners for agents that operate a live simulator service."""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    'run_agent_against_simulator',
    'run_agent_terminal',
    'print_result_conditions',
]


def run_agent_against_simulator(agent, simulator) -> None:
    """Read agent goals from the terminal for an already running simulator."""
    print('')
    if getattr(simulator, 'base_url', None):
        print('Simulator URL: %s' % simulator.base_url)
    print('The block-world simulator must already be running.')
    try:
        simulator.snapshot()
    except Exception as exc:
        print('Could not connect to simulator: %s' % exc)
        return
    print('Enter natural-language agent goals here; watch execution in the viewer if one is running.')
    print('Terminal commands: /state, /reset, /events, /quit')
    print('')
    run_agent_terminal(agent, simulator)


def run_agent_terminal(agent, env, prompt: str = 'goal> ') -> None:
    """Run an interactive terminal loop for an agent."""
    while True:
        try:
            text = input(prompt)
        except EOFError:
            print('')
            return
        request = text.strip()
        if not request:
            continue
        lower = request.lower()
        if lower in {'/quit', 'quit', 'exit'}:
            return
        if lower in {'/state', 'state'}:
            print(env.snapshot_text())
            continue
        if lower in {'/reset', 'reset'}:
            env.reset()
            print('Environment reset.')
            continue
        if lower in {'/events', 'events'}:
            for event in env.event_log():
                print('[{revision}] {kind}: {label} ({status}) -> {result}'.format(
                    revision=event.get('revision'),
                    kind=event.get('kind'),
                    label=event.get('label'),
                    status='OK' if event.get('ok') else 'ERROR',
                    result=event.get('result'),
                ))
            continue

        print('')
        print('Agent planning...')
        result = agent.handle_user_input(request)
        print('')
        print(result)
        result_path = (
            getattr(agent, 'last_result_path', None)
            or getattr(agent, 'last_trace_path', None)
        )
        if result_path:
            print('')
            print_result_conditions(result_path)
        print('')


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
