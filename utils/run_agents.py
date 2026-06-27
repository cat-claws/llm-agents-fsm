"""Unified launcher for the llm-agents-fsm terminal agents."""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.planning_modes import (
    PLANNING_MODES,
    PLANNING_MODE_ADVISORY,
    PLANNING_MODE_CHOICES_TEXT,
    PLANNING_MODE_FSM,
    PLANNING_MODE_PLAN,
    normalize_planning_granularity,
    normalize_violation_policy,
    planning_mode_config,
)

_PLANNING_MODE_TARGET_MODES = (PLANNING_MODE_PLAN, PLANNING_MODE_ADVISORY)

_TARGETS = {
    'git-basic': ('git', 'basic'),
    'git-fsm': ('git', PLANNING_MODE_FSM),
    'shrdlu-basic': ('shrdlu', 'basic'),
    'shrdlu-fsm': ('shrdlu', PLANNING_MODE_FSM),
}
_TARGETS.update(
    {
        '%s-%s' % (domain, mode): (domain, mode)
        for domain in ('git', 'shrdlu')
        for mode in _PLANNING_MODE_TARGET_MODES
    }
)

_FSM_AGENT_IMPL = {
    PLANNING_MODE_ADVISORY: PLANNING_MODE_FSM,
    PLANNING_MODE_PLAN: PLANNING_MODE_FSM,
}


def _agent_impl(agent: str) -> str:
    return _FSM_AGENT_IMPL.get(agent, agent)


def _planning_mode_agent(agent: str) -> str | None:
    return agent if agent in PLANNING_MODES else None


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _set_git_planning_mode_for_agent(agent: str) -> str | None:
    planning_mode = _planning_mode_agent(agent)
    if planning_mode in {PLANNING_MODE_PLAN, PLANNING_MODE_ADVISORY}:
        previous = os.environ.get('GIT_AGENT_FSM_PLANNING_MODE')
        os.environ['GIT_AGENT_FSM_PLANNING_MODE'] = planning_mode
        return previous
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Launch one of the Git or SHRDLU agents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  run-agents git-basic
  run-agents git-fsm
  run-agents git-plan
  run-agents shrdlu-basic -- --result-dir "$PWD/results"
  run-agents shrdlu-fsm -- --result-dir "$PWD/results"
  run-agents shrdlu-plan -- --result-dir "$PWD/results"

Equivalent option form:
  run-agents --domain shrdlu --agent fsm -- --max-steps 20
  run-agents --domain shrdlu --agent advisory
""",
    )
    parser.add_argument(
        'target',
        nargs='?',
        choices=sorted(_TARGETS),
        help='combined target name, e.g. git-fsm or shrdlu-fsm',
    )
    parser.add_argument(
        '--domain',
        choices=['git', 'shrdlu'],
        help='agent domain; use with --agent when target is omitted',
    )
    parser.add_argument(
        '--agent',
        choices=['basic', *PLANNING_MODES],
        help='agent kind or planning mode; planning modes run the FSM agent',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='print available targets and exit',
    )
    return parser


def _split_passthrough(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if '--' not in raw_args:
        return raw_args, []
    split_at = raw_args.index('--')
    return raw_args[:split_at], raw_args[split_at + 1:]


def _print_targets() -> None:
    print('Targets:')
    for target in sorted(_TARGETS):
        domain, requested = _TARGETS[target]
        agent = _agent_impl(requested)
        if requested in PLANNING_MODES:
            print('  %-17s domain=%-6s agent=%-5s mode=%s' % (target, domain, agent, requested))
        else:
            print('  %-17s domain=%-6s agent=%s' % (target, domain, agent))


def _resolve_target(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, str]:
    if args.target:
        return _TARGETS[args.target]

    if not args.domain or not args.agent:
        parser.error('provide a target, or provide both --domain and --agent')

    domain = args.domain
    agent = args.agent

    if agent == 'basic':
        return domain, 'basic'
    if agent in PLANNING_MODES:
        return domain, agent

    parser.error('%s agents support --agent basic, fsm, plan, or advisory' % domain)
    raise AssertionError('unreachable')


def _run_git(agent: str, passthrough: Sequence[str]) -> int:
    if passthrough:
        raise SystemExit('git agents do not accept launcher passthrough args')

    impl_agent = _agent_impl(agent)
    previous_planning_mode = _set_git_planning_mode_for_agent(agent)
    script_name = 'git-agent-basic.py' if impl_agent == 'basic' else 'git-agent-fsm.py'
    script_path = _REPO_ROOT / 'git-system' / script_name
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)]
    try:
        runpy.run_path(str(script_path), run_name='__main__')
    finally:
        sys.argv = old_argv
        if previous_planning_mode is not None or agent in {PLANNING_MODE_PLAN, PLANNING_MODE_ADVISORY}:
            _restore_env('GIT_AGENT_FSM_PLANNING_MODE', previous_planning_mode)
    return 0


def _run_shrdlu(agent: str, passthrough: Sequence[str]) -> int:
    return main_shrdlu(['--agent', agent, *passthrough])


def _build_shrdlu_parser() -> argparse.ArgumentParser:
    from shrdlu_agents.shrdlu_agent_basic import (
        DEFAULT_MAX_STEPS,
        DEFAULT_OPENAI_API_KEY,
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_OPENAI_MODEL,
        DEFAULT_RESULT_DIR,
    )
    from shrdlu_agents.simulator_api import DEFAULT_SIMULATOR_URL

    parser = argparse.ArgumentParser(
        prog='run-agents shrdlu-<agent> --',
        description='Run a SHRDLU block-world agent against a standalone simulator.',
    )
    parser.add_argument(
        '--agent',
        default=os.environ.get('SHRDLU_AGENT_TYPE', 'fsm'),
        choices=['basic', *PLANNING_MODES],
        help='agent kind or planning mode; plan/advisory run the FSM agent',
    )
    parser.add_argument(
        '--simulator-url',
        default=os.environ.get('SHRDLU_SIMULATOR_URL', DEFAULT_SIMULATOR_URL),
        help='base URL for the already-running simulator service',
    )
    parser.add_argument(
        '--base-url',
        default=os.environ.get('SHRDLU_OPENAI_BASE_URL', DEFAULT_OPENAI_BASE_URL),
        help='OpenAI-compatible chat completions base URL',
    )
    parser.add_argument(
        '--api-key',
        default=os.environ.get('SHRDLU_OPENAI_API_KEY', DEFAULT_OPENAI_API_KEY),
        help='OpenAI-compatible API key',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('SHRDLU_OPENAI_MODEL', DEFAULT_OPENAI_MODEL),
        help='OpenAI-compatible model name',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=float(os.environ.get('SHRDLU_OPENAI_TEMPERATURE', '0.2')),
        help='chat sampling temperature',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=int(os.environ.get('SHRDLU_OPENAI_MAX_TOKENS', '512')),
        help='maximum tokens per chat call',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=int(os.environ.get('SHRDLU_AGENT_MAX_STEPS', DEFAULT_MAX_STEPS)),
        help='maximum executed/planned simulator actions per request',
    )
    parser.add_argument(
        '--max-branch-retries',
        type=int,
        default=None,
        help='planning retries per branch',
    )
    parser.add_argument(
        '--planning-mode',
        choices=list(PLANNING_MODES),
        default=os.environ.get('SHRDLU_AGENT_FSM_PLANNING_MODE'),
        help='initial FSM planning mode %s' % PLANNING_MODE_CHOICES_TEXT,
    )
    overrides = parser.add_argument_group('advanced planning overrides')
    overrides.add_argument(
        '--planning-granularity',
        default=(
            os.environ.get('SHRDLU_AGENT_FSM_PLANNING_GRANULARITY')
            or os.environ.get('SHRDLU_AGENT_FSM_PLANNING')
        ),
        help='advanced override: step or batch; normally set by --planning-mode',
    )
    overrides.add_argument(
        '--violation-policy',
        default=(
            os.environ.get('SHRDLU_AGENT_FSM_VIOLATION_POLICY')
            or os.environ.get('SHRDLU_AGENT_FSM_VIOLATIONS')
        ),
        help='advanced override: retry, ignore, or advisory; normally set by --planning-mode',
    )
    parser.add_argument(
        '--result-dir',
        dest='result_dir',
        default=(
            os.environ.get('SHRDLU_AGENT_RESULT_DIR')
            or os.environ.get('SHRDLU_AGENT_TRACE_DIR')
            or DEFAULT_RESULT_DIR
        ),
        help='directory for saved agent result records; use an empty string to disable',
    )
    parser.add_argument(
        '--trace-dir',
        dest='result_dir',
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_shrdlu_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_shrdlu_parser()
    args = parser.parse_args(argv)
    args.requested_agent = args.agent
    if (
        args.requested_agent in {PLANNING_MODE_PLAN, PLANNING_MODE_ADVISORY}
        and args.planning_mode
        and args.planning_mode != args.requested_agent
    ):
        parser.error(
            'planning mode supplied twice: --agent/target %r conflicts with --planning-mode %r'
            % (args.requested_agent, args.planning_mode)
        )
    if args.requested_agent in PLANNING_MODES and not args.planning_mode:
        args.planning_mode = args.requested_agent
    args.agent = _agent_impl(args.agent)
    if args.agent not in {'basic', 'fsm'}:
        parser.error(
            '--agent must be one of advisory, basic, fsm, plan; got %r'
            % args.requested_agent
        )
    try:
        if args.planning_granularity:
            args.planning_granularity = normalize_planning_granularity(
                args.planning_granularity,
                invalid='raise',
            )
        if args.violation_policy:
            args.violation_policy = normalize_violation_policy(
                args.violation_policy,
                invalid='raise',
            )
        if args.planning_mode:
            planning_mode_config(
                args.planning_mode,
                retry_default=int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
                invalid='raise',
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.result_dir == '':
        args.result_dir = None
    return args


def _shrdlu_planning_mode(args: argparse.Namespace) -> str:
    if args.planning_mode:
        return args.planning_mode
    return PLANNING_MODE_FSM


def _build_shrdlu_agent(args: argparse.Namespace):
    from shrdlu_agents.shrdlu_agent_basic import OpenAICompatibleShrdluAgent
    from shrdlu_agents.shrdlu_agent_fsm import FsmOpenAICompatibleShrdluAgent
    from shrdlu_agents.simulator_api import HttpSimulatorClient

    agent_types = {
        'basic': OpenAICompatibleShrdluAgent,
        'fsm': FsmOpenAICompatibleShrdluAgent,
    }
    simulator = HttpSimulatorClient(args.simulator_url)
    default_branch_retries = int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3'))
    args.runtime_retry_default = (
        args.max_branch_retries
        if args.max_branch_retries is not None
        else default_branch_retries
    )
    kwargs = {
        'model': args.model,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'max_steps': args.max_steps,
        'result_dir': args.result_dir,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }
    if args.agent == 'fsm':
        planning_mode = _shrdlu_planning_mode(args)
        mode_config = planning_mode_config(
            planning_mode,
            retry_default=default_branch_retries,
            invalid='raise',
        )
        kwargs['max_branch_retries'] = (
            args.max_branch_retries
            if args.max_branch_retries is not None
            else int(mode_config['max_retries'])
        )
        kwargs['planning_granularity'] = (
            args.planning_granularity
            or str(mode_config['planning_granularity'])
        )
        kwargs['violation_policy'] = (
            args.violation_policy
            or str(mode_config['violation_policy'])
        )
    agent_obj = agent_types[args.agent](simulator, **kwargs)
    return agent_obj, simulator


def _shrdlu_launch_lines(args: argparse.Namespace) -> list[str]:
    requested_agent = getattr(args, 'requested_agent', args.agent)
    if requested_agent == args.agent:
        return ['Launch: domain=shrdlu | agent=%s' % args.agent]
    return [
        'Launch: domain=shrdlu | agent=%s | requested_mode=%s'
        % (args.agent, requested_agent)
    ]


def main_shrdlu(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = _parse_shrdlu_args(argv)
    agent_obj, simulator = _build_shrdlu_agent(args)

    from shrdlu_agents.terminal import run_agent_against_simulator

    launch_lines = _shrdlu_launch_lines(args)
    run_agent_against_simulator(
        agent_obj,
        simulator,
        launch_lines=launch_lines,
        planning_retry_default=(
            args.runtime_retry_default
            if args.agent == 'fsm'
            else None
        ),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    launcher_args, explicit_passthrough = _split_passthrough(argv)
    args, implicit_passthrough = parser.parse_known_args(launcher_args)
    passthrough = [*implicit_passthrough, *explicit_passthrough]

    if args.list:
        _print_targets()
        return 0

    domain, agent = _resolve_target(args, parser)
    if domain == 'git':
        return _run_git(agent, passthrough)
    return _run_shrdlu(agent, passthrough)


if __name__ == '__main__':
    raise SystemExit(main())
