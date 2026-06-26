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

_TARGETS = {
    'git-basic': ('git', 'basic'),
    'git-fsm': ('git', 'fsm'),
    'shrdlu-reactive': ('shrdlu', 'reactive'),
    'shrdlu-fsm': ('shrdlu', 'fsm'),
    # Compatibility names for the old SHRDLU modes. These still route through
    # the merged FSM implementation using the presets below.
    'shrdlu-preplanned': ('shrdlu', 'preplanned'),
    'shrdlu-predictive': ('shrdlu', 'predictive'),
    'shrdlu-suffix': ('shrdlu', 'suffix'),
}

_CANONICAL_TARGETS = ('git-basic', 'git-fsm', 'shrdlu-reactive', 'shrdlu-fsm')
_SHRDLU_ALIAS_TARGETS = ('shrdlu-preplanned', 'shrdlu-predictive', 'shrdlu-suffix')
_SHRDLU_AGENT_ALIASES = {
    'preplanned': 'fsm',
    'predictive': 'fsm',
    'suffix': 'fsm',
}
_SHRDLU_ALIAS_PRESETS = {
    'preplanned': {
        'planning_granularity': 'batch',
        'violation_policy': 'ignore',
        'max_branch_retries': 1,
    },
    'predictive': {
        'planning_granularity': 'step',
        'violation_policy': 'retry',
    },
    'suffix': {
        'planning_granularity': 'batch',
        'violation_policy': 'retry',
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Launch one of the Git or SHRDLU agents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  run-agents git-basic
  run-agents git-fsm
  run-agents shrdlu-reactive -- --result-dir "$PWD/results"
  run-agents shrdlu-fsm -- --result-dir "$PWD/results"

Equivalent option form:
  run-agents --domain shrdlu --agent fsm -- --max-steps 20
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
        choices=['basic', 'reactive', 'fsm', 'preplanned', 'predictive', 'suffix'],
        help='agent mode; basic/reactive are aliases across domains',
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
    print('Canonical targets:')
    for target in _CANONICAL_TARGETS:
        domain, agent = _TARGETS[target]
        print('  %-17s domain=%-6s agent=%s' % (target, domain, agent))
    print('\nSHRDLU compatibility aliases:')
    for target in _SHRDLU_ALIAS_TARGETS:
        domain, agent = _TARGETS[target]
        print('  %-17s domain=%-6s preset=%s' % (target, domain, agent))


def _resolve_target(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, str]:
    if args.target:
        return _TARGETS[args.target]

    if not args.domain or not args.agent:
        parser.error('provide a target, or provide both --domain and --agent')

    domain = args.domain
    agent = args.agent

    if domain == 'git':
        if agent in {'basic', 'reactive'}:
            return 'git', 'basic'
        if agent == 'fsm':
            return 'git', 'fsm'
        parser.error("git agents support --agent basic/reactive or fsm")

    if agent in {'basic', 'reactive'}:
        return 'shrdlu', 'reactive'
    if agent in {'fsm', 'preplanned', 'predictive', 'suffix'}:
        return 'shrdlu', agent

    parser.error("unsupported domain/agent combination")
    raise AssertionError('unreachable')


def _run_git(agent: str, passthrough: Sequence[str]) -> int:
    if passthrough:
        raise SystemExit('git agents do not accept launcher passthrough args')

    script_name = 'git-agent-basic.py' if agent == 'basic' else 'git-agent-fsm.py'
    script_path = _REPO_ROOT / 'git-system' / script_name
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)]
    try:
        runpy.run_path(str(script_path), run_name='__main__')
    finally:
        sys.argv = old_argv
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
        help='agent strategy to run: reactive or fsm; legacy aliases preplanned, predictive, suffix are accepted',
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
        help='planning retries per predictive branch',
    )
    parser.add_argument(
        '--planning-granularity',
        choices=['step', 'batch'],
        default=(
            os.environ.get('SHRDLU_AGENT_FSM_PLANNING_GRANULARITY')
            or os.environ.get('SHRDLU_AGENT_FSM_PLANNING')
        ),
        help='FSM planning granularity: step plans one action at a time; batch plans a remaining suffix',
    )
    parser.add_argument(
        '--violation-policy',
        choices=['retry', 'ignore'],
        default=(
            os.environ.get('SHRDLU_AGENT_FSM_VIOLATION_POLICY')
            or os.environ.get('SHRDLU_AGENT_FSM_VIOLATIONS')
        ),
        help='FSM property behavior: retry blocks/replans on violations; ignore records and continues',
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
    args.agent_alias = args.agent
    args.agent = _SHRDLU_AGENT_ALIASES.get(args.agent, args.agent)
    if args.agent not in {'reactive', 'fsm'}:
        parser.error(
            '--agent must be one of fsm, preplanned, predictive, reactive, suffix; got %r'
            % args.agent_alias
        )
    if args.result_dir == '':
        args.result_dir = None
    return args


def _build_shrdlu_agent(args: argparse.Namespace):
    from shrdlu_agents.shrdlu_agent_basic import OpenAICompatibleShrdluAgent
    from shrdlu_agents.shrdlu_agent_fsm import FsmOpenAICompatibleShrdluAgent
    from shrdlu_agents.simulator_api import HttpSimulatorClient

    agent_types = {
        'reactive': OpenAICompatibleShrdluAgent,
        'fsm': FsmOpenAICompatibleShrdluAgent,
    }
    simulator = HttpSimulatorClient(args.simulator_url)
    original_agent = getattr(args, 'agent_alias', args.agent)
    default_branch_retries = int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3'))
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
        preset = _SHRDLU_ALIAS_PRESETS.get(original_agent, {})
        kwargs['max_branch_retries'] = (
            args.max_branch_retries
            if args.max_branch_retries is not None
            else preset.get('max_branch_retries', default_branch_retries)
        )
        kwargs['planning_granularity'] = (
            args.planning_granularity
            or preset.get('planning_granularity')
            or 'batch'
        )
        kwargs['violation_policy'] = (
            args.violation_policy
            or preset.get('violation_policy')
            or 'retry'
        )
    agent_obj = agent_types[args.agent](simulator, **kwargs)
    return agent_obj, simulator


def _print_shrdlu_launch(args: argparse.Namespace) -> None:
    label = args.agent if args.agent_alias == args.agent else '%s (alias for %s)' % (
        args.agent_alias,
        args.agent,
    )
    print('Agent type: %s' % label)
    if args.agent == 'fsm':
        preset = _SHRDLU_ALIAS_PRESETS.get(args.agent_alias, {})
        retries = (
            args.max_branch_retries
            if args.max_branch_retries is not None
            else preset.get(
                'max_branch_retries',
                int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
            )
        )
        print(
            'FSM config: planning=%s violations=%s retries=%d'
            % (
                args.planning_granularity or preset.get('planning_granularity') or 'batch',
                args.violation_policy or preset.get('violation_policy') or 'retry',
                retries,
            )
        )


def main_shrdlu(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = _parse_shrdlu_args(argv)
    agent_obj, simulator = _build_shrdlu_agent(args)
    _print_shrdlu_launch(args)

    from shrdlu_agents.terminal import run_agent_against_simulator

    run_agent_against_simulator(agent_obj, simulator)
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
