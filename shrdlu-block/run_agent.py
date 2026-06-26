"""Run one SHRDLU block-world agent against an existing simulator service."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Type

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrdlu_agents.shrdlu_agent_basic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACE_DIR,
    OpenAICompatibleShrdluAgent,
)
from shrdlu_agents.simulator_api import DEFAULT_SIMULATOR_URL, HttpSimulatorClient
from shrdlu_agents.shrdlu_agent_fsm import FsmOpenAICompatibleShrdluAgent
from shrdlu_agents.terminal import run_agent_against_simulator

AGENT_TYPES: Dict[str, Type[OpenAICompatibleShrdluAgent]] = {
    'reactive': OpenAICompatibleShrdluAgent,
    'fsm': FsmOpenAICompatibleShrdluAgent,
}

AGENT_ALIASES = {
    'preplanned': 'fsm',
    'predictive': 'fsm',
    'suffix': 'fsm',
}

ALIAS_PRESETS = {
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


def build_agent(args):
    simulator = HttpSimulatorClient(args.simulator_url)
    original_agent = getattr(args, 'agent_alias', args.agent)
    default_branch_retries = int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3'))
    kwargs = {
        'model': args.model,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'max_steps': args.max_steps,
        'trace_dir': args.trace_dir,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }
    if args.agent == 'fsm':
        preset = ALIAS_PRESETS.get(original_agent, {})
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
    agent = AGENT_TYPES[args.agent](simulator, **kwargs)
    return agent, simulator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
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
        '--trace-dir',
        default=os.environ.get('SHRDLU_AGENT_TRACE_DIR', DEFAULT_TRACE_DIR),
        help='directory for saved agent traces; use an empty string to disable',
    )
    args = parser.parse_args(argv)
    args.agent_alias = args.agent
    args.agent = AGENT_ALIASES.get(args.agent, args.agent)
    if args.agent not in AGENT_TYPES:
        parser.error(
            '--agent must be one of %s; got %r'
            % (', '.join(sorted(AGENT_TYPES)), args.agent_alias)
        )
    if args.trace_dir == '':
        args.trace_dir = None
    return args


def main(argv=None):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(argv)
    agent, simulator = build_agent(args)
    label = args.agent if args.agent_alias == args.agent else '%s (alias for %s)' % (args.agent_alias, args.agent)
    print('Agent type: %s' % label)
    if args.agent == 'fsm':
        print(
            'FSM config: planning=%s violations=%s retries=%d'
            % (
                args.planning_granularity or ALIAS_PRESETS.get(args.agent_alias, {}).get('planning_granularity') or 'batch',
                args.violation_policy or ALIAS_PRESETS.get(args.agent_alias, {}).get('violation_policy') or 'retry',
                (
                    args.max_branch_retries
                    if args.max_branch_retries is not None
                    else ALIAS_PRESETS.get(args.agent_alias, {}).get(
                        'max_branch_retries',
                        int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
                    )
                ),
            )
        )
    run_agent_against_simulator(agent, simulator)


if __name__ == '__main__':
    main()
