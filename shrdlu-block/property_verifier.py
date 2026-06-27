"""AP and property verification utilities for SHRDLU world snapshots."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = ['ACTIVE_PROPERTY_IDS', 'PROPERTY_FILE', 'TransitionPropertyVerifier']


PROPERTY_FILE = Path(__file__).resolve().parent / 'resources' / 'SHRDLU_AP_CANDIDATES.json'
ACTIVE_PROPERTY_IDS = (
    'prop.object_4_on_6_stays_on_6',
    'prop.object_4_not_on_3_and_6_simultaneously',
    'prop.no_object_resting_on_4',
    'prop.no_object_resting_on_8',
    'prop.no_object_resting_on_10',
    'prop.object_on_10_implies_next_closed',
)

PROPERTY_SPECS = [
    {
        'id': 'prop.object_4_on_6_stays_on_6',
        'ltl': 'G(object_4_resting_on_6 -> G(object_4_resting_on_6))',
        'description': 'Once object 4 is on object 6, it stays on object 6 forever.',
    },
    {
        'id': 'prop.object_4_not_on_3_and_6_simultaneously',
        'ltl': 'G(!(object_4_resting_on_3 && object_4_resting_on_6))',
        'description': 'Object 4 is never simultaneously resting on both object 3 and object 6.',
    },
    {
        'id': 'prop.no_object_resting_on_4',
        'ltl': 'G(!some_object_resting_on_4)',
        'description': 'No object is ever resting on object 4.',
    },
    {
        'id': 'prop.no_object_resting_on_8',
        'ltl': 'G(!some_object_resting_on_8)',
        'description': 'No object is ever resting on object 8.',
    },
    {
        'id': 'prop.no_object_resting_on_10',
        'ltl': 'G(!some_object_resting_on_10)',
        'description': 'No object is ever resting on object 10.',
    },
    {
        'id': 'prop.object_on_10_implies_next_closed',
        'ltl': 'G(some_object_resting_on_10 -> X(grasper_closed))',
        'description': 'Whenever something is resting on object 10, the next state has the grasper closed.',
    },
]


class TransitionPropertyVerifier:
    """Evaluate the current AP set and concrete LTL properties."""

    def __init__(self, properties: Iterable[Dict[str, object]]):
        del properties
        self._aps = self._load_ap_specs(PROPERTY_FILE)
        self._properties = [spec for spec in PROPERTY_SPECS if spec['id'] in ACTIVE_PROPERTY_IDS]

    @classmethod
    def from_file(cls, path: Path = PROPERTY_FILE) -> 'TransitionPropertyVerifier':
        return cls(cls._load_ap_specs(path))

    @property
    def properties(self) -> List[Dict[str, object]]:
        return list(self._properties)

    @property
    def aps(self) -> List[Dict[str, object]]:
        return list(self._aps)

    def verify_transition(
        self,
        pre_state: Dict[str, object],
        action: Dict[str, object],
        post_state: Dict[str, object],
        *,
        pre_scene=None,
        post_scene=None,
    ) -> Dict[str, object]:
        del pre_state, action, pre_scene, post_scene
        ap_cache: Dict[str, bool] = {}
        property_results = []
        for spec in self._aps:
            name = str(spec.get('name', ''))
            satisfied = self._eval_ap(name, post_state)
            ap_cache[name] = satisfied
            property_results.append({
                'id': name,
                'natural_language': spec.get('description'),
                'satisfied': satisfied,
            })
        violations = [result for result in property_results if not result['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': violations,
            'property_results': property_results,
            'derived_aps': dict(sorted(ap_cache.items())),
        }

    def verify_trace(self, states: List[Dict[str, object]]) -> Dict[str, object]:
        ap_trace = [self._evaluate_state_aps(state) for state in states]
        property_results = []
        for spec in self._properties:
            satisfied = self._eval_property(spec['id'], ap_trace)
            property_results.append({
                'id': spec['id'],
                'natural_language': spec['description'],
                'ltl': spec['ltl'],
                'satisfied': satisfied,
            })
        violations = [result for result in property_results if not result['satisfied']]
        return {
            'all_satisfied': not violations,
            'violations': violations,
            'property_results': property_results,
            'ap_trace': ap_trace,
        }

    def _eval_ap(self, name: str, state: Dict[str, object]) -> bool:
        if name == 'some_object_resting_on_4':
            return any(obj.get('resting_on') == 4 for obj in state.get('objects', []))
        if name == 'some_object_resting_on_8':
            return any(obj.get('resting_on') == 8 for obj in state.get('objects', []))
        if name == 'some_object_resting_on_10':
            return any(obj.get('resting_on') == 10 for obj in state.get('objects', []))
        if name == 'object_5_resting_on_2':
            return self._object_resting_on(state, 5, 2)
        if name == 'object_4_resting_on_3':
            return self._object_resting_on(state, 4, 3)
        if name == 'object_8_resting_on_7':
            return self._object_resting_on(state, 8, 7)
        if name == 'object_10_resting_on_9':
            return self._object_resting_on(state, 10, 9)
        if name == 'grasper_closed':
            return state.get('grasper_closed') is True
        if name == 'grasper_lowered':
            return state.get('grasper_lowered') is True
        if name == 'object_4_resting_on_6':
            return self._object_resting_on(state, 4, 6)
        raise ValueError('Unsupported atomic proposition: %s' % name)

    def observe_ap(self, name: str, state: Dict[str, object]) -> bool:
        """Return one AP truth value from a SHRDLU world-state snapshot."""
        return bool(self._eval_ap(name, state))

    def _evaluate_state_aps(
        self,
        state: Dict[str, object],
        ap_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, bool]:
        values = {}
        names = (
            [str(spec.get('name', '')) for spec in self._aps]
            if ap_names is None
            else list(ap_names)
        )
        for name in names:
            values[name] = self.observe_ap(name, state)
        return values

    def _eval_property(self, prop_id: str, ap_trace: List[Dict[str, bool]]) -> bool:
        if prop_id == 'prop.object_4_on_6_stays_on_6':
            return self._prop_object_4_on_6_stays_on_6(ap_trace)
        if prop_id == 'prop.object_4_not_on_3_and_6_simultaneously':
            return self._globally(ap_trace, lambda s, i: not (s['object_4_resting_on_3'] and s['object_4_resting_on_6']))
        if prop_id == 'prop.no_object_resting_on_4':
            return self._globally(ap_trace, lambda s, i: not s['some_object_resting_on_4'])
        if prop_id == 'prop.no_object_resting_on_8':
            return self._globally(ap_trace, lambda s, i: not s['some_object_resting_on_8'])
        if prop_id == 'prop.no_object_resting_on_10':
            return self._globally(ap_trace, lambda s, i: not s['some_object_resting_on_10'])
        if prop_id == 'prop.lowered_eventually_raised':
            return self._globally(
                ap_trace,
                lambda s, i: (not s['grasper_lowered']) or self._eventually(ap_trace, i, lambda future: not future['grasper_lowered']),
            )
        if prop_id == 'prop.closed_eventually_open':
            return self._globally(
                ap_trace,
                lambda s, i: (not s['grasper_closed']) or self._eventually(ap_trace, i, lambda future: not future['grasper_closed']),
            )
        if prop_id == 'prop.object_on_10_implies_next_closed':
            return self._globally(
                ap_trace,
                lambda s, i: (not s['some_object_resting_on_10']) or self._next(ap_trace, i, lambda nxt: nxt['grasper_closed']),
            )
        raise ValueError('Unsupported property id: %s' % prop_id)

    @staticmethod
    def _globally(ap_trace: List[Dict[str, bool]], predicate) -> bool:
        return all(predicate(state, index) for index, state in enumerate(ap_trace))

    @staticmethod
    def _eventually(ap_trace: List[Dict[str, bool]], start_index: int, predicate) -> bool:
        return any(predicate(ap_trace[index]) for index in range(start_index, len(ap_trace)))

    @staticmethod
    def _next(ap_trace: List[Dict[str, bool]], index: int, predicate) -> bool:
        next_index = index + 1
        if next_index >= len(ap_trace):
            return False
        return predicate(ap_trace[next_index])

    def _prop_object_4_on_6_stays_on_6(self, ap_trace: List[Dict[str, bool]]) -> bool:
        for index, state in enumerate(ap_trace):
            if state['object_4_resting_on_6'] and not all(
                future['object_4_resting_on_6'] for future in ap_trace[index:]
            ):
                return False
        return True

    @staticmethod
    def _load_ap_specs(path: Path) -> List[Dict[str, object]]:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return list(payload.get('current_state_aps', []))

    @staticmethod
    def _get_object(snapshot: Dict[str, object], obj_id: int) -> Optional[Dict[str, object]]:
        for obj in snapshot.get('objects', []):
            if obj.get('obj_id') == obj_id:
                return obj
        return None

    @classmethod
    def _object_resting_on(cls, snapshot: Dict[str, object], obj_id: int, support_id: int) -> bool:
        obj = cls._get_object(snapshot, obj_id)
        return obj is not None and obj.get('resting_on') == support_id


def _demo_plan() -> List[Dict[str, object]]:
    return [
        {'name': 'move_grasper', 'args': {'x': 0.15, 'y': -0.1}},
        {'name': 'lower_grasper', 'args': {}},
        {'name': 'close_grasper', 'args': {}},
        {'name': 'raise_grasper', 'args': {}},
        {'name': 'move_grasper', 'args': {'x': -0.3, 'y': 0.05}},
        {'name': 'lower_grasper', 'args': {}},
        {'name': 'open_grasper', 'args': {}},
    ]


def _print_ap_results(title: str, result: Dict[str, object]) -> None:
    print(title)
    for item in result['property_results']:
        print(f"{item['id']}\t{item['satisfied']}")


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import argparse

    from shrdlu_agents.simulator_api import DEFAULT_SIMULATOR_URL, HttpSimulatorClient

    parser = argparse.ArgumentParser(
        description='Verify SHRDLU block-world properties using simulator HTTP snapshots.',
    )
    parser.add_argument(
        '--simulator-url',
        default=os.environ.get('SHRDLU_SIMULATOR_URL', DEFAULT_SIMULATOR_URL),
        help='base URL for the already-running simulator service',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='HTTP timeout in seconds',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='run the old demo action trace against the live simulator service',
    )
    parser.add_argument(
        '--no-reset-before-demo',
        action='store_true',
        help='do not reset the simulator before running --demo',
    )
    args = parser.parse_args(argv)

    client = HttpSimulatorClient(args.simulator_url, timeout=args.timeout)
    state = client.snapshot()
    verifier = TransitionPropertyVerifier.from_file()
    result = verifier.verify_transition(state, {'name': 'init', 'args': {}}, state)
    trace_result = verifier.verify_trace([state])

    print('INITIAL_AP_RESULTS')
    for item in result['property_results']:
        print(f"{item['id']}\t{item['satisfied']}")

    print('\nINITIAL_TRACE_PROPERTY_RESULTS')
    for item in trace_result['property_results']:
        print(f"{item['id']}\t{item['satisfied']}\t{item['ltl']}")

    print('\nOBJECTS')
    for obj in state['objects']:
        print(obj['obj_id'], obj['kind'], obj['color'], 'resting_on=', obj['resting_on'])

    print('\nGRASPER')
    print('grasper_closed=', state['grasper_closed'])
    print('grasper_lowered=', state['grasper_lowered'])

    if not args.demo:
        return 0

    print('\nDEMO_TRACE_MOVE_RED_PYRAMID_ONTO_MEDIUM_GREEN_BLOCK')
    before = client.snapshot()
    if not args.no_reset_before_demo:
        before = client.reset()
    before_ap = verifier.verify_transition(before, {'name': 'init', 'args': {}}, before)
    demo_states = [before]
    demo_results = []
    current = before
    for action in _demo_plan():
        outcome = client.execute_action(action)
        current = client.snapshot()
        demo_states.append(current)
        demo_results.append((action, outcome))
    after = demo_states[-1]
    after_ap = verifier.verify_transition(before, {'name': 'demo_plan', 'args': {}}, after)
    demo_trace_result = verifier.verify_trace(demo_states)

    print()
    _print_ap_results('DEMO_AP_BEFORE', before_ap)
    print()
    _print_ap_results('DEMO_AP_AFTER', after_ap)
    print()
    print('DEMO_ACTION_RESULTS')
    for action, outcome in demo_results:
        print(action, '=>', outcome)
    print()
    print('DEMO_PROPERTY_RESULTS_ON_TRACE')
    for item in demo_trace_result['property_results']:
        print(f"{item['id']}\t{item['satisfied']}\t{item['ltl']}")
    print()
    print('DEMO_TRACE_LENGTH', len(demo_states))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
