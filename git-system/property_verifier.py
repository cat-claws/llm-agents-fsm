"""AP and property verification utilities for git-agent-fsm traces.

AP evaluation strategy
----------------------
Every AP in GIT_AP_CANDIDATES.json carries a list of ``git_commands`` that
gather raw text evidence from the repository.  That evidence is fed to an
OpenAI-compatible LLM with a yes/no question to determine the boolean truth
value of the AP.  APs starting with "The most recent action" describe the
immediately preceding step's action type and are evaluated by the LLM like
all other APs.

Property evaluation
-------------------
Properties are evaluated by walking the LTL AST from GIT_PROPERTIES_AST.json
recursively.  All remaining properties use only the node types:
  globally, next, implies, and, or, not, ap

verify_trace  — evaluates full finite-trace LTL semantics over a sequence
                of steps, including the ``next`` operator.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    'PROPERTY_FILE',
    'AP_FILE',
    'PropertyVerifier',
]

PROPERTY_FILE = Path(__file__).resolve().parent / 'resources' / 'GIT_PROPERTIES_AST.json'
AP_FILE = Path(__file__).resolve().parent / 'resources' / 'GIT_AP_CANDIDATES.json'

_DEFAULT_BASE_URL = 'http://127.0.0.1:30000/v1/'
_DEFAULT_API_KEY = 'EMPTY'
_DEFAULT_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_EXTRA_BODY = {
    'chat_template_kwargs': {'enable_thinking': True},
    'separate_reasoning': True,
}


def _load_properties(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return list(payload['properties'])


def _load_ap_specs(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload['current_state_aps']


def _run_commands(commands: List[str], repo_path: str) -> str:
    """Run each shell command in ``repo_path`` and concatenate stdout+stderr."""
    parts = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            out = (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            out = '<timeout>'
        except Exception as exc:
            out = f'<error: {exc}>'
        parts.append(f'$ {cmd}\n{out}')
    return '\n\n'.join(parts)


def _ask_llm(ap_name: str, ap_description: str, evidence: str, client, model: str) -> bool:
    """Call the LLM to determine whether the AP holds given the shell evidence."""
    import re
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a git repository state analyser. '
                'Given shell command outputs from a git repository, answer whether a '
                'specific condition (atomic proposition) is TRUE or FALSE. '
                'Reply with exactly one word: TRUE or FALSE.'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'Atomic proposition: {ap_name}\n'
                f'Definition: {ap_description}\n\n'
                f'Shell evidence:\n{evidence}\n\n'
                'Is the atomic proposition TRUE or FALSE given this evidence?'
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        extra_body=_DEFAULT_EXTRA_BODY,
    )
    raw = response.choices[0].message.content or ''
    answer = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()
    return answer.startswith('T')


def _eval_ast(node: Dict, merged_aps: Dict[str, bool]) -> bool:
    """Recursively evaluate a single-step LTL AST node against a merged AP dict."""
    t = node['type']
    if t == 'ap':
        name = node['name']
        if name not in merged_aps:
            raise KeyError(f'AP not found in evaluation context: {name!r}')
        return merged_aps[name]
    if t == 'not':
        return not _eval_ast(node['operand'], merged_aps)
    if t == 'and':
        return all(_eval_ast(arg, merged_aps) for arg in node['args'])
    if t == 'or':
        return any(_eval_ast(arg, merged_aps) for arg in node['args'])
    if t == 'implies':
        return (not _eval_ast(node['left'], merged_aps)) or _eval_ast(node['right'], merged_aps)
    if t == 'globally':
        # In a single-step context G(φ) reduces to φ.
        return _eval_ast(node['operand'], merged_aps)
    if t == 'next':
        # Cannot evaluate next in a single-step context; vacuously true.
        return True
    raise ValueError(f'Unsupported AST node type: {t!r}')


def _eval_ast_trace(
    node: Dict,
    ap_trace: List[Dict[str, bool]],
    index: int,
) -> bool:
    """Evaluate an LTL AST node at position ``index`` over a finite trace."""
    t = node['type']
    if index >= len(ap_trace):
        # Past the end of the trace; all path conditions vacuously true except G.
        return True
    merged = ap_trace[index]
    if t == 'ap':
        name = node['name']
        if name not in merged:
            raise KeyError(f'AP not found at step {index}: {name!r}')
        return merged[name]
    if t == 'not':
        return not _eval_ast_trace(node['operand'], ap_trace, index)
    if t == 'and':
        return all(_eval_ast_trace(arg, ap_trace, index) for arg in node['args'])
    if t == 'or':
        return any(_eval_ast_trace(arg, ap_trace, index) for arg in node['args'])
    if t == 'implies':
        return (
            not _eval_ast_trace(node['left'], ap_trace, index)
            or _eval_ast_trace(node['right'], ap_trace, index)
        )
    if t == 'globally':
        return all(
            _eval_ast_trace(node['operand'], ap_trace, i)
            for i in range(index, len(ap_trace))
        )
    if t == 'next':
        next_index = index + 1
        if next_index >= len(ap_trace):
            return True  # vacuously true past end of finite trace
        return _eval_ast_trace(node['operand'], ap_trace, next_index)
    raise ValueError(f'Unsupported AST node type: {t!r}')


class PropertyVerifier:
    """Verify git-agent FSM properties against live repository state.

    Parameters
    ----------
    repo_path:
        Absolute path to the git repository to inspect.
    model:
        OpenAI-compatible model ID used for AP evaluation.
    client:
        An ``openai.OpenAI`` client instance. If omitted, one is created from
        the OpenAI-compatible base URL and API key.
    """

    def __init__(
        self,
        repo_path: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str = _DEFAULT_API_KEY,
        model: Optional[str] = None,
        client=None,
    ):
        self._repo_path = repo_path
        self._client = client or self._make_client(base_url, api_key)
        self._model = model or _DEFAULT_MODEL
        self._properties = _load_properties(PROPERTY_FILE)
        self._aps = _load_ap_specs(AP_FILE)
        self._ap_by_name: Dict[str, Dict] = {
            a['name']: a for a in self._aps
        }

    @staticmethod
    def _make_client(base_url: str, api_key: str):
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key=api_key)

    @property
    def properties(self) -> List[Dict]:
        return list(self._properties)

    @property
    def aps(self) -> List[Dict]:
        return list(self._aps)

    @staticmethod
    def _default_commands_for_ap(ap_name: str) -> List[str]:
        del ap_name
        return [
            'git status --short --branch',
            'git branch -vv',
            'git log --oneline --decorate -20',
            'git remote -v',
        ]

    def observe_ap(self, name: str) -> bool:
        """Return one AP truth value from the current repository state."""
        spec = self._ap_by_name.get(name)
        description = spec.get('description', '') if spec else name
        commands = (
            spec.get('git_commands', [])
            if spec is not None
            else self._default_commands_for_ap(name)
        )
        evidence = _run_commands(commands, self._repo_path) if commands else '(no commands defined)'
        return _ask_llm(name, description, evidence, self._client, self._model)

    def evaluate_aps(self, ap_names: Optional[Iterable[str]] = None) -> Dict[str, bool]:
        """Run git commands and call LLM to evaluate all APs.

        Returns a mapping ``{ap_name: bool}`` for every AP.
        """
        names = (
            [spec['name'] for spec in self._aps]
            if ap_names is None
            else list(ap_names)
        )
        return {name: self.observe_ap(name) for name in names}

    def verify_transition(
        self,
        *,
        ap_values: Optional[Dict[str, bool]] = None,
    ) -> Dict:
        """Evaluate all properties against a single FSM step.

        Parameters
        ----------
        ap_values:
            Pre-evaluated AP booleans.  If ``None``, ``evaluate_aps``
            is called automatically to read the current repository state.

        Returns
        -------
        dict with keys: all_satisfied, violations, property_results, derived_aps
        """
        if ap_values is None:
            ap_values = self.evaluate_aps()
        merged: Dict[str, bool] = dict(ap_values)

        property_results = []
        for prop in self._properties:
            satisfied = _eval_ast(prop['ast'], merged)
            property_results.append({
                'id': prop['id'],
                'natural_language': prop['natural_language'],
                'ltl': prop['ltl'],
                'satisfied': satisfied,
            })

        violations = [r for r in property_results if not r['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': violations,
            'property_results': property_results,
            'derived_aps': dict(sorted(merged.items())),
        }

    def verify_trace(
        self,
        steps: List[Dict[str, bool]],
    ) -> Dict:
        """Evaluate properties over a finite trace of FSM steps.

        Parameters
        ----------
        steps:
            Sequence of ``ap_values`` dicts, one per step, each mapping AP
            name to boolean.  Pre-evaluate via ``evaluate_aps`` at observation
            time for each step.

        Returns
        -------
        dict with keys: all_satisfied, violations, property_results, ap_trace
        """
        ap_trace = list(steps)

        property_results = []
        for prop in self._properties:
            satisfied = _eval_ast_trace(prop['ast'], ap_trace, index=0)
            property_results.append({
                'id': prop['id'],
                'natural_language': prop['natural_language'],
                'ltl': prop['ltl'],
                'satisfied': satisfied,
            })

        violations = [r for r in property_results if not r['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': violations,
            'property_results': property_results,
            'ap_trace': ap_trace,
        }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate git-agent-fsm properties against a live repository.',
    )
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        print(
            'error: current working directory does not exist. '
            'Please cd to a valid directory or pass repo_path explicitly.',
            file=__import__('sys').stderr,
        )
        return 2
    parser.add_argument(
        'repo_path',
        nargs='?',
        default=cwd,
        help='path to the git repository to inspect (default: cwd)',
    )
    parser.add_argument('--base-url', default=_DEFAULT_BASE_URL)
    parser.add_argument('--api-key', default=_DEFAULT_API_KEY)
    parser.add_argument('--model', default=None, help='model id (default: SHRDLU-compatible Qwen model)')
    args = parser.parse_args(argv)

    verifier = PropertyVerifier(
        args.repo_path,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    print(f'Evaluating {len(verifier.aps)} APs in {args.repo_path} ...')
    ap_values = verifier.evaluate_aps()

    print('\nAP_VALUES')
    for name, value in sorted(ap_values.items()):
        flag = 'TRUE ' if value else 'FALSE'
        print(f'  {flag}  {name}')

    result = verifier.verify_transition(ap_values=ap_values)

    print('\nPROPERTY_RESULTS')
    for item in result['property_results']:
        flag = 'PASS' if item['satisfied'] else 'FAIL'
        print(f'  {flag}  {item["id"]}  {item["ltl"]}')

    if result['violations']:
        print('\nVIOLATIONS')
        for v in result['violations']:
            print(f'  {v["id"]}: {v["natural_language"]}')

    return 0 if result['all_satisfied'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
