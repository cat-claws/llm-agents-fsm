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
    PreplannedOpenAICompatibleShrdluAgent,
)
from shrdlu_agents.shrdlu_agent_plan import (
    PredictivePreplannedOpenAICompatibleShrdluAgent,
)
from shrdlu_agents.simulator_api import DEFAULT_SIMULATOR_URL, HttpSimulatorClient
from shrdlu_agents.shrdlu_agent_fsm import (
    SuffixPredictivePreplannedOpenAICompatibleShrdluAgent,
)
from shrdlu_agents.terminal import run_agent_against_simulator

AGENT_TYPES: Dict[str, Type[OpenAICompatibleShrdluAgent]] = {
    'reactive': OpenAICompatibleShrdluAgent,
    'preplanned': PreplannedOpenAICompatibleShrdluAgent,
    'predictive': PredictivePreplannedOpenAICompatibleShrdluAgent,
    'suffix': SuffixPredictivePreplannedOpenAICompatibleShrdluAgent,
}


def build_agent(args):
    simulator = HttpSimulatorClient(args.simulator_url)
    kwargs = {
        'model': args.model,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'max_steps': args.max_steps,
        'trace_dir': args.trace_dir,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }
    if args.agent in {'predictive', 'suffix'}:
        kwargs['max_branch_retries'] = args.max_branch_retries
    agent = AGENT_TYPES[args.agent](simulator, **kwargs)
    return agent, simulator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run a SHRDLU block-world agent against a standalone simulator.',
    )
    parser.add_argument(
        '--agent',
        choices=sorted(AGENT_TYPES),
        default=os.environ.get('SHRDLU_AGENT_TYPE', 'preplanned'),
        help='agent strategy to run',
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
        default=int(os.environ.get('SHRDLU_AGENT_MAX_BRANCH_RETRIES', '3')),
        help='planning retries per predictive branch',
    )
    parser.add_argument(
        '--trace-dir',
        default=os.environ.get('SHRDLU_AGENT_TRACE_DIR', DEFAULT_TRACE_DIR),
        help='directory for saved agent traces; use an empty string to disable',
    )
    args = parser.parse_args(argv)
    if args.agent not in AGENT_TYPES:
        parser.error(
            '--agent must be one of %s; got %r'
            % (', '.join(sorted(AGENT_TYPES)), args.agent)
        )
    if args.trace_dir == '':
        args.trace_dir = None
    return args


def main(argv=None):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(argv)
    agent, simulator = build_agent(args)
    print('Agent type: %s' % args.agent)
    run_agent_against_simulator(agent, simulator)


if __name__ == '__main__':
    main()
