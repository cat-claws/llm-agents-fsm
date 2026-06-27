"""OpenAI-compatible merged FSM/planning agents for the SHRDLU blocks environment."""

from __future__ import annotations

import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.planning_modes import (
    NONBLOCKING_VIOLATION_POLICIES,
    PLANNING_BATCH,
    PLANNING_STEP,
    VIOLATION_IGNORE,
    VIOLATION_RETRY,
    normalize_planning_granularity,
    normalize_violation_policy,
    property_guidance_text as _property_guidance_text,
    property_policy_text as _shared_property_policy_text,
)
from utils.property_catalog import (
    aps_from_properties,
    load_property_catalog,
    observe_ap_values,
)
from utils.session import (
    accepted_nodes_by_depth,
    annotate_node_executed,
    append_node,
    build_tree_summary,
    extract_property_ids_from_violations,
    make_action,
    make_planning_node,
    make_planning_tree,
    make_session,
    make_state_path_entry,
    make_verification,
    set_node_outcome,
)
from utils.tla_verifier import (
    verify_ap_trace as _verify_tla_ap_trace,
)

from shrdlu_agents.shrdlu_agent_basic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RESULT_DIR,
    OpenAICompatibleShrdluAgent,
    PLAN_SCHEMA,
)
from shrdlu_agents.simulator_api import ALLOWED_SIMULATOR_ACTION_NAMES, SimulatorAPI
from shrdlu_blocks.simulator.state_pred import predict_world_state_after_actions

_TLA_PROPERTIES_FILE = Path(__file__).resolve().parent / 'resources' / 'SHRDLU_PROPERTIES_AST.json'
_TLA_PROPERTIES = load_property_catalog(_TLA_PROPERTIES_FILE)
_STATE_AP_NAMES, _TRANSITION_AP_NAMES = aps_from_properties(_TLA_PROPERTIES)
_AP_NAMES: List[str] = _STATE_AP_NAMES + _TRANSITION_AP_NAMES

__all__ = [
    'FsmOpenAICompatibleShrdluAgent',
]


def _verify_ap_trace(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    *,
    module_name: str = 'ShrdluTrace',
    timeout: int = 60,
    is_complete_trace: bool = True,
) -> Dict:
    return _verify_tla_ap_trace(
        ap_trace,
        ap_names,
        _TLA_PROPERTIES,
        module_name=module_name,
        timeout=timeout,
        is_complete_trace=is_complete_trace,
    )


def _normalize_planning_granularity(
    value: str | None,
    default: str = PLANNING_BATCH,
    *,
    invalid: Literal["default", "raise"] = "raise",
) -> str:
    return normalize_planning_granularity(
        value,
        default=default,
        invalid=invalid,
    )


def _normalize_violation_policy(
    value: str | None,
    default: str = VIOLATION_RETRY,
    *,
    invalid: Literal["default", "raise"] = "raise",
) -> str:
    return normalize_violation_policy(value, default=default, invalid=invalid)


BATCH_PLAN_SYSTEM_PROMPT = """You are planning a SHRDLU blocks-world task before execution.

Rules:
- Think through the full remaining task first, then return the complete remaining action suffix.
- Treat each suffix attempt as self-contained; do not assume you will get to repair it later.
- Treat the plan as a dry-run sequence of primitive simulator calls that will be verified before execution.
- Use only the allowed primitive action names and the matching JSON args listed in the allowed actions schema.
- Never invent argument names not listed in the schema. Never use null or descriptive strings where a concrete numeric argument is required.
- Ground every action argument from the initial world state, accepted action trace, or structured planning state summary.
- The plan must be complete: include every action from the current state all the way to goal completion.
- Resolve object references by every user-mentioned attribute simultaneously. "green small block" must match one object with color=green, kind=block, size=small — not a green object resting on a small block.
- For pick/place goals, identify one concrete source from source_candidates and one concrete destination with can_support=true from destination_candidates before writing the suffix.
- Ground move_grasper(x, y) by copying coordinates from the chosen object in the initial world state summary.
- Plan; do not refuse.
- Do not explain alternatives or reasoning. Keep the response short and factual.

Return strict JSON only.
"""

STEP_PREDICTIVE_PLAN_SYSTEM_PROMPT = """You are planning a SHRDLU blocks-world task before execution.

Rules:
- Plan exactly one primitive action for the current predicted state.
- Treat the action as a dry-run primitive simulator call that will be checked before execution.
- Use only one allowed primitive action name and exactly the matching JSON args listed in the allowed actions schema.
- Never invent argument names not listed in the schema. Never use null or descriptive strings where a concrete numeric argument is required.
- Ground every action argument from the initial world state, accepted action trace, or structured planning state summary.
- If the goal is already satisfied in the current predicted state, return an empty plan.
- Do not explain alternatives or reasoning. Keep the response short and factual.

Return strict JSON only.
"""

PLAN_USER_PROMPT_TEMPLATE = """\
Goal:
{request}

{grounding_verdict}

Current predicted AP truth values (ap_name: true/false):
{current_ap_bools_json}

Structured planning state summary:
{planning_state_summary}

Accepted action trace so far:
{accepted_trace_json}

{property_section}

Allowed primitive actions:
{action_help}

Failed plan attempts and backtrack feedback:
{failed_attempts_json}

Banned first actions at this node (do NOT start your plan with any of these — they were already tried here):
{banned_first_actions_json}

{plan_instruction}
JSON schema: {{"response": "...", "plan": [{{"name": "...", "args": {{...}}}}], "finish_response": "..."}}
Return strict JSON only."""

PLAN_REPAIR_PROMPT_TEMPLATE = """\
Your previous reply was invalid: {error}
Rewrite it as strict JSON only using this schema:
{{"response": "...", "plan": [{{"name": "...", "args": {{...}}}}], "finish_response": "..."}}
Return the complete remaining action sequence from the current state to goal completion."""

AP_STATE_PREDICTION_SYSTEM_PROMPT = """You are predicting atomic proposition (AP) truth values in a SHRDLU blocks-world simulator after one action.

## Simulator action effects (exact rules)

move_grasper(x, y):
  Precondition: grasper_lowered == false.
  Effect: grasper moves to (x, y). If an object is grasped, it moves with the grasper.
  resting_on is unchanged (it only changes via lower_grasper / raise_grasper / open_grasper).

lower_grasper:
  Precondition: grasper_lowered == false.
  Effect: grasper_lowered = true.
  If NOT holding: grasper descends to the highest object directly below (x, y), or the table.
    No object's resting_on changes.
  If holding (grasped_object != null): held object descends to the highest object directly below (x, y).
    held_object.resting_on = that support object (or null if table). No other resting_on changes.
  If precondition fails: no state change.

raise_grasper:
  Precondition: grasper_lowered == true.
  Effect: grasper_lowered = false.
  If NOT holding: no object resting_on changes.
  If holding: held_object.resting_on = null (object is now airborne).
  If precondition fails: no state change.

close_grasper:
  Precondition: grasper_closed == false.
  Effect: grasper_closed = true.
  If lowered AND grasper.resting_on is a graspable object: grasped_object = that object id.
  Otherwise: grasped_object stays null. No resting_on changes.
  If precondition fails: no state change.

open_grasper:
  Precondition: grasper_closed == true.
  If holding: further precondition: grasper_lowered == true AND held_object.resting_on != null AND support is valid.
    On success: grasped_object = null, grasper_closed = false. held_object.resting_on stays as set by lower_grasper.
    If further precondition fails: raises error, no state change.
  If NOT holding: grasper_closed = false. No other change.

## Constraints

- Work from the initial world state + action history delta, not from the AP booleans alone.
- resting_on changes only via: lower_grasper (when holding), raise_grasper (clears to null), open_grasper (held object released, resting_on stays).
- grasper_lowered changes only via lower_grasper / raise_grasper.
- grasper_closed changes only via close_grasper / open_grasper.
- If an action fails its precondition: no state changes — copy all current AP values unchanged.
- Every AP must appear in ap_results exactly once as a boolean (true or false). No strings, no nulls.

## Output

Fill the "reasoning" field in this order before writing ap_results:
  1. object_positions   — for every object in the world, state its current (x, y) position. Start from the initial world state and apply any move_grasper actions that moved a held object to derive the current position of each object.
  2. grasped_object    — what object (if any) is currently held, from the action history delta.
  3. precondition_check — does this action pass its precondition? If not, state no-change.
  4. world_delta        — which fields change (grasper_lowered, grasper_closed, grasped_object, which resting_on)?
  5. ap_derivation      — evaluate each AP formula against the resulting world state.

Required JSON shape:
{"reasoning": {"object_positions": "...", "grasped_object": "...", "precondition_check": "...", "world_delta": "...", "ap_derivation": "..."}, "response": "...", "ap_results": {"<ap_name>": true, ...}}
Return strict JSON only."""

AP_STATE_PREDICTION_PROMPT_TEMPLATE = """\
Initial world state (authoritative — object positions, resting_on, grasped_object at t=0):
{init_world_state_json}

Accepted actions so far (applied in order to the initial world state):
{accepted_trace_json}

Accumulated world-state delta from accepted actions:
{world_state_delta}

Current AP truth values (derived from the predicted state after accepted actions):
{current_ap_bools_json}

Next action to predict:
{action_json}

Atomic propositions (name: evaluation rule):
{ap_catalog_text}

Return strict JSON only."""

AP_STATE_SCHEMA = {
    'type': 'object',
    'properties': {
        'reasoning': {
            'type': 'object',
            'properties': {
                'object_positions': {'type': 'string'},
                'grasped_object': {'type': 'string'},
                'precondition_check': {'type': 'string'},
                'world_delta': {'type': 'string'},
                'ap_derivation': {'type': 'string'},
            },
            'required': ['object_positions', 'grasped_object', 'precondition_check', 'world_delta', 'ap_derivation'],
        },
        'response': {
            'type': 'string',
        },
        'ap_results': {
            'type': 'object',
            'additionalProperties': {'type': 'boolean'},
        },
    },
    'required': ['reasoning', 'response', 'ap_results'],
}


class _FsmShrdluAgentMixin:
    """Plan before execution with configurable granularity and property policy."""

    def _init_fsm_planner(self, max_branch_retries: int = 3):
        self._max_branch_retries = int(max_branch_retries)
        self._property_text = self._build_property_text()

    @staticmethod
    def _snapshot_json(snapshot) -> str:
        return json.dumps(snapshot, indent=2, sort_keys=True)

    @staticmethod
    def _json_or_none(value) -> str:
        if value in (None, [], {}):
            return 'None'
        return json.dumps(value, indent=2, sort_keys=True)

    @staticmethod
    def _recent_actions(accepted_trace: List[Dict[str, object]], limit: int = 6) -> List[str]:
        names = []
        for action in accepted_trace[-limit:]:
            if isinstance(action, dict):
                names.append(str(action.get('name', '')))
        return names

    @staticmethod
    def _action_signature(action: Dict[str, object]) -> str:
        return json.dumps({
            'name': action.get('name'),
            'args': action.get('args', {}),
        }, sort_keys=True)

    @classmethod
    def _recent_action_signatures(
        cls,
        accepted_trace: List[Dict[str, object]],
        limit: int = 6,
    ) -> List[str]:
        signatures = []
        for action in accepted_trace[-limit:]:
            if isinstance(action, dict):
                signatures.append(cls._action_signature(action))
        return signatures

    @classmethod
    def _identical_repeat_warning(cls, accepted_trace: List[Dict[str, object]]) -> str:
        signatures = cls._recent_action_signatures(accepted_trace, limit=3)
        if len(signatures) >= 2 and signatures[-1] == signatures[-2]:
            return 'identical repeated action detected: %s' % signatures[-1]
        return 'none'

    @classmethod
    def _alternating_warning(cls, accepted_trace: List[Dict[str, object]]) -> str:
        names = cls._recent_actions(accepted_trace, limit=6)
        if len(names) < 4:
            return 'none'
        if (
            len(set(names[-4:])) == 2
            and names[-4] == names[-2]
            and names[-3] == names[-1]
            and names[-4] != names[-3]
        ):
            return 'recent alternating loop detected: %s' % ' -> '.join(names[-4:])
        return 'none'

    @classmethod
    def _planning_state_summary(
        cls,
        current_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
        request: str = '',
    ) -> str:
        grounding = cls._grounding_summary(current_state, request)
        highlighted_objects = grounding['goal_relevant']['highlighted_objects']
        object_catalog = grounding['object_catalog']
        source_candidates = grounding['source_candidates']
        destination_candidates = grounding['destination_candidates']
        request_focus = grounding['request_focus']
        exact_source_matches = grounding['exact_source_matches']
        exact_destination_matches = grounding['exact_destination_matches']
        object_count = grounding['object_count']
        goal_relevant = grounding['goal_relevant']
        summary = {
            'request_focus': request_focus,
            'goal_relevant': {
                'default_grasper': goal_relevant.get('default_grasper'),
                'grasper_lowered': goal_relevant.get('grasper_lowered'),
                'grasper_closed': goal_relevant.get('grasper_closed'),
                'grasped_object': goal_relevant.get('grasped_object'),
                'highlighted_objects': highlighted_objects,
                'highlighted_count': len(highlighted_objects),
            },
            'recent_actions': cls._recent_actions(accepted_trace),
            'recent_action_signatures': cls._recent_action_signatures(accepted_trace),
            'alternating_warning': cls._alternating_warning(accepted_trace),
            'identical_repeat_warning': cls._identical_repeat_warning(accepted_trace),
            'object_count': object_count,
            'object_catalog': object_catalog,
            'argument_grounding_rules': {
                'highlight_object': 'Use obj_id from object_catalog.',
                'unhighlight_object': 'Use obj_id from object_catalog.',
                'move_grasper': 'Use numeric x and y copied from object_catalog positions or explicit numeric state value.',
                'all_goals': 'For all/each/every goals, choose one concrete object id per step.',
                'pick_place_goals': (
                    'Choose one concrete source object from source_candidates and one concrete '
                    'destination object from destination_candidates.'
                ),
            },
            'candidate_source_objects': exact_source_matches,
            'candidate_destination_objects': exact_destination_matches,
            'binding_warning': (
                'Never satisfy a phrase like "green small block" by combining attributes from multiple objects.'
            ),
            'source_candidates': source_candidates,
            'destination_candidates': destination_candidates,
        }
        return json.dumps(summary, indent=2, sort_keys=True)

    @classmethod
    def _grounding_summary(cls, current_state: Dict[str, object], request: str) -> Dict[str, object]:
        highlighted_objects = []
        object_catalog = []
        for obj in current_state.get('objects', []):
            if not isinstance(obj, dict):
                continue
            tags = obj.get('tags', {}) if isinstance(obj.get('tags', {}), dict) else {}
            if bool(tags.get('highlight', False)):
                highlighted_objects.append(obj.get('obj_id'))
            pos = obj.get('position', {}) if isinstance(obj.get('position', {}), dict) else {}
            object_catalog.append({
                'obj_id': obj.get('obj_id'),
                'kind': obj.get('kind'),
                'color': obj.get('color'),
                'graspable': bool(obj.get('graspable', False)),
                'can_support': bool(obj.get('can_support', False)),
                'size': tags.get('size'),
                'height': tags.get('height'),
                'width': tags.get('width'),
                'highlighted': bool(tags.get('highlight', False)),
                'resting_on': obj.get('resting_on'),
                'position': {
                    'x': pos.get('x'),
                    'y': pos.get('y'),
                    'z': pos.get('z'),
                },
            })
        source_candidates = [
            {
                'obj_id': item['obj_id'],
                'kind': item['kind'],
                'color': item['color'],
                'size': item.get('size'),
                'height': item.get('height'),
                'width': item.get('width'),
                'position': item['position'],
            }
            for item in object_catalog
            if item.get('obj_id') is not None and item.get('graspable')
        ]
        destination_candidates = [
            {
                'obj_id': item['obj_id'],
                'kind': item['kind'],
                'color': item['color'],
                'size': item.get('size'),
                'height': item.get('height'),
                'width': item.get('width'),
                'can_support': item.get('can_support'),
                'position': item['position'],
            }
            for item in object_catalog
            if item.get('obj_id') is not None and item.get('can_support')
        ]
        request_focus = cls._request_focus(request)
        return {
            'request_focus': request_focus,
            'goal_relevant': {
                'default_grasper': current_state.get('default_grasper'),
                'grasper_lowered': current_state.get('grasper_lowered'),
                'grasper_closed': current_state.get('grasper_closed'),
                'grasped_object': current_state.get('grasped_object'),
                'highlighted_objects': highlighted_objects,
                'highlighted_count': len(highlighted_objects),
            },
            'object_count': len(current_state.get('objects', [])),
            'object_catalog': object_catalog,
            'exact_source_matches': cls._exact_matches(
                request_focus.get('source', {}),
                source_candidates,
            ),
            'exact_destination_matches': cls._exact_matches(
                request_focus.get('destination', {}),
                destination_candidates,
            ),
            'source_candidates': source_candidates,
            'destination_candidates': destination_candidates,
        }

    @classmethod
    def _grounding_verdict_text(cls, current_state: Dict[str, object], request: str) -> str:
        grounding = cls._grounding_summary(current_state, request)
        request_focus = grounding['request_focus']
        source_phrase = request_focus.get('source_phrase') or '(none)'
        destination_phrase = request_focus.get('destination_phrase') or '(none)'

        def labels(items: List[Dict[str, object]]) -> str:
            if not items:
                return 'none'
            parts = []
            for item in items:
                parts.append(
                    'obj_id={obj_id} color={color} kind={kind} size={size} height={height} width={width}'.format(
                        obj_id=item.get('obj_id'),
                        color=item.get('color'),
                        kind=item.get('kind'),
                        size=item.get('size'),
                        height=item.get('height'),
                        width=item.get('width'),
                    )
                )
            return '; '.join(parts)

        return '\n'.join([
            'Grounding context:',
            'Source phrase: %s' % source_phrase,
            'Destination phrase: %s' % destination_phrase,
            'Relevant source objects: %s' % labels(grounding['exact_source_matches']),
            'Relevant destination objects: %s' % labels(grounding['exact_destination_matches']),
            'Destination binding rule: choose one destination object id from the described candidates; do not merge attributes across nearby, stacked, or co-located objects.',
            'Invalid binding example: a red small block plus a green pyramid at the same x,y does not equal a green small block.',
            'Plan; do not refuse.',
        ])

    @classmethod
    def _request_focus(cls, request: str) -> Dict[str, object]:
        text = (request or '').strip().lower()
        normalized = re.sub(r'[^a-z0-9\s]', ' ', text)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        move_markers = [
            ' onto ', ' on top of ', ' on ', ' into ', ' in ', ' to ',
        ]
        source_text = normalized
        destination_text = ''
        padded = ' %s ' % normalized if normalized else ' '
        for marker in move_markers:
            idx = padded.find(marker)
            if idx != -1:
                source_text = padded[1:idx].strip()
                destination_text = padded[idx + len(marker):-1].strip()
                break
        source_text = cls._strip_leading_goal_verb(source_text)
        destination_text = cls._strip_leading_determiner(destination_text)

        return {
            'raw_request': request,
            'normalized_request': normalized,
            'source_phrase': source_text,
            'destination_phrase': destination_text,
            'source': {
                'colors': cls._extract_attribute_tokens(
                    source_text,
                    {'red', 'green', 'blue', 'white', 'black', 'yellow'},
                ),
                'kinds': cls._extract_attribute_tokens(
                    source_text,
                    {'block', 'blocks', 'pyramid', 'pyramids', 'box', 'boxes', 'table'},
                ),
                'sizes': cls._extract_attribute_tokens(
                    source_text,
                    {'small', 'medium', 'big', 'tall', 'short', 'wide', 'narrow'},
                ),
            },
            'destination': {
                'colors': cls._extract_attribute_tokens(
                    destination_text,
                    {'red', 'green', 'blue', 'white', 'black', 'yellow'},
                ),
                'kinds': cls._extract_attribute_tokens(
                    destination_text,
                    {'block', 'blocks', 'pyramid', 'pyramids', 'box', 'boxes', 'table'},
                ),
                'sizes': cls._extract_attribute_tokens(
                    destination_text,
                    {'small', 'medium', 'big', 'tall', 'short', 'wide', 'narrow'},
                ),
            },
        }

    @staticmethod
    def _strip_leading_goal_verb(text: str) -> str:
        words = text.split()
        while words and words[0] in {
            'put', 'move', 'place', 'stack', 'set', 'bring', 'take', 'drop',
            'highlight', 'unhighlight',
        }:
            words = words[1:]
        while words and words[0] in {'the', 'a', 'an'}:
            words = words[1:]
        return ' '.join(words)

    @staticmethod
    def _strip_leading_determiner(text: str) -> str:
        words = text.split()
        while words and words[0] in {'the', 'a', 'an'}:
            words = words[1:]
        return ' '.join(words)

    @staticmethod
    def _extract_attribute_tokens(text: str, allowed_tokens) -> List[str]:
        tokens = []
        for token in text.split():
            singular = token[:-1] if token.endswith('s') else token
            if token in allowed_tokens:
                tokens.append(token)
            elif singular in allowed_tokens:
                tokens.append(singular)
        seen = set()
        result = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    @staticmethod
    def _exact_matches(target: Dict[str, object], candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        colors = set(target.get('colors', []))
        kinds = {k[:-1] if k.endswith('s') else k for k in target.get('kinds', [])}
        size_like = set(target.get('sizes', []))
        matches = []
        for item in candidates:
            if colors and item.get('color') not in colors:
                continue
            if kinds and item.get('kind') not in kinds:
                continue
            if 'small' in size_like or 'medium' in size_like or 'big' in size_like:
                if item.get('size') not in size_like:
                    continue
            if 'tall' in size_like or 'short' in size_like:
                if item.get('height') not in size_like:
                    continue
            if 'wide' in size_like or 'narrow' in size_like:
                if item.get('width') not in size_like:
                    continue
            matches.append(item)
        return matches

    def _build_property_text(self) -> str:
        return _property_guidance_text(_TLA_PROPERTIES)

    def _property_monitoring_metadata(self) -> Dict[str, object]:
        return {
            'enabled': True,
            'property_file': str(_TLA_PROPERTIES_FILE),
            'property_count': len(_TLA_PROPERTIES),
            'ap_source': 'properties_ast',
            'ap_count': len(_AP_NAMES),
        }

    def _run_agent_loop(self, request: str) -> str:
        initial_world_state = self._env.snapshot()
        initial_state = self._build_initial_ap_state(initial_world_state)
        planning_mode = "%s_%s" % (self._planning_granularity, self._violation_policy)
        action_help = self._env.action_help()
        trace = make_session(
            agent='shrdlu-agent-fsm',
            model=self._model,
            domain='shrdlu',
            request=request,
            properties=_TLA_PROPERTIES,
            planning_config={
                'host': self._host,
                'planning_mode': planning_mode,
                'planning_granularity': self._planning_granularity,
                'violation_policy': self._violation_policy,
                'max_steps': self._max_steps,
                'max_branch_retries': self._max_branch_retries,
                'property_monitoring': self._property_monitoring_metadata(),
            },
        )
        trace['planning_tree'] = make_planning_tree(
            mode=planning_mode,
            max_steps=self._max_steps,
            max_branch_retries=self._max_branch_retries,
            planning_granularity=self._planning_granularity,
            violation_policy=self._violation_policy,
            properties=_TLA_PROPERTIES,
            initial_state=initial_state,
            initial_world_state=initial_world_state,
            action_help=action_help,
        )
        trace['status'] = 'planning'
        result_path = self._start_result_session(trace)

        result = self._search_plan(
            request=request,
            current_state=initial_state,
            init_world_state=initial_world_state,
            preceding_ap_trace=[initial_state],
            accepted_trace=[],
            depth=0,
            planning_tree=trace['planning_tree'],
            action_help=action_help,
            parent_node_id=None,
            inherited_failures=[],
            hint_plan=None,
            trace=trace,
            result_path=result_path,
        )

        trace['planning_tree']['feasible'] = bool(result.get('success'))
        trace['planning_tree']['accepted_plan'] = result.get('plan', []) if result.get('success') else []
        trace['planning_tree']['finish_response'] = result.get('finish_response')
        trace['planning_tree']['planning_response'] = result.get('planning_response')
        if result.get('failure'):
            trace['planning_tree']['failure'] = result['failure']

        if not result.get('success'):
            base_message = self._normalize_response_text(
                result.get('finish_response', 'No feasible property-satisfying plan found.'),
                is_finish=True,
            )
            violated = self._collect_violated_properties(result.get('failure'))
            if violated:
                violated_text = 'Properties violated: ' + ', '.join(sorted(violated))
                final_message = base_message + '\n' + violated_text
            else:
                final_message = base_message
            trace['status'] = 'infeasible'
            trace['final_message'] = final_message
            trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
            result_path = self._write_result(trace, result_path)
            return self._append_result_notice(final_message, result_path)

        plan = result['plan']
        response_text = self._normalize_response_text(
            result.get('planning_response', 'Verified plan ready.'),
            is_finish=not plan,
        )
        finish_response = self._normalize_response_text(
            result.get('finish_response', 'Done.'),
            is_finish=True,
        )

        if not plan:
            trace['status'] = 'finished'
            trace['final_message'] = finish_response
            trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
            result_path = self._write_result(trace, result_path)
            return finish_response if response_text == finish_response else self._format_reply(
                response_text,
                finish_response,
            )

        executed_ap_trace = [initial_state]
        trace['status'] = 'executing'
        accepted_by_depth = accepted_nodes_by_depth(
            trace['planning_tree'],
            include_finish=True,
        )
        self._checkpoint_result(trace, result_path)
        for step_index, action in enumerate(plan):
            try:
                result_text = self._env.execute_action(action)
            except Exception as exc:
                result_text = "ERROR: %s" % exc
            post_state = self._env.snapshot()
            ap_state = self._build_initial_ap_state(post_state)
            executed_ap_trace.append(ap_state)
            tla_result = _verify_ap_trace(executed_ap_trace, _AP_NAMES)
            node = accepted_by_depth.get(step_index)
            if node is not None:
                annotate_node_executed(
                    node,
                    execution_step=step_index,
                    execution_result=result_text,
                    ap_state=ap_state,
                    ap_changes=self._diff_ap_states(executed_ap_trace[-2], ap_state),
                    tla_verification=tla_result,
                    observation_after=self._env.snapshot_text(),
                )
            self._checkpoint_result(trace, result_path)
            if isinstance(result_text, str) and result_text.startswith('ERROR:'):
                final_message = self._format_reply(
                    response_text + "\n\nPlan execution failed.",
                    "Executed %s.\nResult: %s" % (self._format_action(action), result_text),
                )
                trace['status'] = 'error'
                trace['final_message'] = final_message
                trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
                result_path = self._write_result(trace, result_path)
                return self._append_result_notice(final_message, result_path)

        # Post-execution grasper cleanup: if the grasper is not already raised and
        # open, bring it to a clean state. The LLM is not asked to plan this — we
        # simply inspect the live world state and run the minimum sequence.
        self._execute_grasper_cleanup(trace)

        final_message = finish_response
        if response_text != finish_response:
            final_message = self._format_reply(response_text, finish_response)
        trace['status'] = 'finished'
        trace['final_message'] = final_message
        trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
        result_path = self._write_result(trace, result_path)
        return final_message

    def _search_plan(
        self,
        *,
        request: str,
        current_state: Dict[str, object],
        init_world_state: Dict[str, object],
        preceding_ap_trace: List[Dict[str, bool]],
        accepted_trace: List[Dict[str, object]],
        depth: int,
        planning_tree: Dict[str, object],
        action_help: str,
        parent_node_id: Optional[int],
        inherited_failures: List[Dict[str, object]],
        hint_plan: Optional[List[Dict[str, object]]],
        trace: Dict[str, object],
        result_path: Optional[str],
    ) -> Dict[str, object]:
        """Search for a feasible plan from the current state.

        Each call corresponds to exactly one node in the planning tree, which
        represents the choice of *one action* at the current state.  The node
        may try up to ``max_branch_retries`` different actions (children).  For
        each candidate action the node:

          1. Reuses ``hint_plan`` from the parent's verified suffix tail when
             batch planning is enabled; otherwise asks the LLM for a plan from
             this state.
          2. Verifies only the *first* action of that suffix via AP prediction
             + TLC.
          3. If the first action passes, recurses into a child node passing the
             remaining suffix as ``hint_plan``.
          4. If the child subtree dies the node tries a new action (next child
             slot) — true backtracking.
          5. If all ``max_branch_retries`` actions fail the node is dead and
             propagates failure to its parent.

        This ensures one node per action in the tree, so backtracking walks
        back exactly one action at a time.
        """
        if len(planning_tree['nodes']) >= self._max_steps:
            return {
                'success': False,
                'failure': {
                    'type': 'max_tries',
                    'depth': depth,
                    'nodes_created': len(planning_tree['nodes']),
                    'message': 'Planning exceeded the max node budget of %d.' % self._max_steps,
                },
                'finish_response': 'No feasible property-satisfying plan found.',
            }

        node_id = len(planning_tree['nodes'])
        node = make_planning_node(
            node_id=node_id,
            parent_node_id=parent_node_id,
            depth=depth,
            state_before=copy.deepcopy(current_state),
            state_path=self._zip_accepted_steps(accepted_trace, preceding_ap_trace),
        )
        append_node(planning_tree, node, link_parent=True)
        self._checkpoint_result(trace, result_path)

        failed_attempts = list(inherited_failures)
        banned_first_actions: List[Dict[str, object]] = []
        current_hint = list(hint_plan) if hint_plan and self._planning_granularity == PLANNING_BATCH else []

        for child_index in range(self._max_branch_retries):
            if current_hint:
                plan_prompt = None
                content = ''
                plan_bundle = {
                    'response': 'Reusing previously planned suffix tail.',
                    'plan': copy.deepcopy(current_hint),
                    'finish_response': 'Done.',
                }
                attempts = [{
                    'attempt_index': 0,
                    'reuse_hint_plan': True,
                    'hint_plan_length': len(current_hint),
                }]
                current_hint = []
            else:
                plan_prompt = self._build_plan_prompt(
                    request=request,
                    action_help=action_help,
                    current_state=current_state,
                    init_world_state=init_world_state,
                    accepted_trace=accepted_trace,
                    failed_attempts=failed_attempts,
                    banned_first_actions=banned_first_actions,
                )
                history = [
                    {'role': 'system', 'content': self._planning_system_prompt()},
                    {'role': 'user', 'content': plan_prompt},
                ]
                try:
                    content, plan_bundle, attempts = self._request_plan(history)
                except Exception as exc:
                    failure = {
                        'type': 'planning_error',
                        'depth': depth,
                        'child_index': child_index,
                        'message': str(exc),
                    }
                    node['attempts'].append({
                        'child_index': child_index,
                        'planner_prompt': plan_prompt,
                        'error': str(exc),
                    })
                    self._checkpoint_result(trace, result_path)
                    failed_attempts.append(failure)
                    current_hint = []
                    continue

            response_text = self._normalize_response_text(
                plan_bundle.get('response', ''),
                is_finish=not plan_bundle['plan'],
            )
            finish_response = self._normalize_response_text(
                plan_bundle.get('finish_response', 'Done.'),
                is_finish=True,
            )
            attempt_trace = {
                'child_index': child_index,
                'planner_prompt': plan_prompt,
                'planner_attempts': attempts,
                'planner_response': content,
                'planner_decision': plan_bundle,
            }
            if plan_prompt is None:
                attempt_trace['plan_source'] = 'hint_plan'

            if self._planning_granularity == PLANNING_STEP and len(plan_bundle['plan']) > 1:
                attempt_trace['truncated_to_single_step'] = True
                plan_bundle = dict(plan_bundle)
                plan_bundle['plan'] = plan_bundle['plan'][:1]

            if not plan_bundle['plan']:
                attempt_trace['accepted'] = True
                attempt_trace['finish'] = True
                node['attempts'].append(attempt_trace)
                set_node_outcome(node, result='finish', finish_response=finish_response)
                self._checkpoint_result(trace, result_path)
                return {
                    'success': True,
                    'plan': [],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }

            action = plan_bundle['plan'][0]
            tail = plan_bundle['plan'][1:] if self._planning_granularity == PLANNING_BATCH else []

            step_verification = self._verify_single_step(
                action=action,
                current_state=current_state,
                init_world_state=init_world_state,
                preceding_ap_trace=preceding_ap_trace,
                accepted_trace=accepted_trace,
                is_last_step=(not tail and self._planning_granularity == PLANNING_BATCH),
            )
            attempt_trace['action'] = action
            attempt_trace['step_verification'] = step_verification

            if not step_verification['passed']:
                failure = step_verification['failure']
                can_continue = (
                    self._violation_policy in NONBLOCKING_VIOLATION_POLICIES
                    and isinstance(failure, dict)
                    and failure.get('type') == 'tla_property_violation'
                    and step_verification.get('predicted_ap_state') is not None
                )
                if can_continue:
                    attempt_trace['ignored_property_violation'] = failure
                    step_verification['ignored_by_policy'] = True
                else:
                    attempt_trace['accepted'] = False
                    attempt_trace['failure_feedback'] = failure
                    node['attempts'].append(attempt_trace)
                    self._checkpoint_result(trace, result_path)
                    failed_attempts.append(failure)
                    banned_first_actions.append(action)
                    current_hint = []
                    continue

            predicted_ap_state = step_verification['predicted_ap_state']
            new_preceding_ap_trace = preceding_ap_trace + [predicted_ap_state]

            if not tail and self._planning_granularity == PLANNING_BATCH:
                attempt_trace['accepted'] = True
                node['attempts'].append(attempt_trace)
                tlc = step_verification.get('prediction_detail', {}).get('tla_verification', {})
                verif = make_verification(
                    passed=step_verification['passed'],
                    properties_checked=tlc.get('properties_checked', []),
                    violations=tlc.get('tlc_result', {}).get('violations', []),
                )
                node_result = (
                    'accepted_with_ignored_violations'
                    if step_verification.get('ignored_by_policy')
                    else 'accepted'
                )
                set_node_outcome(
                    node,
                    result=node_result,
                    action_label=action.get('name', 'unknown'),
                    tool='simulator_action',
                    args=action.get('args', {}),
                    state_after=predicted_ap_state,
                    verification=verif,
                    finish_response=finish_response,
                )
                self._checkpoint_result(trace, result_path)
                return {
                    'success': True,
                    'plan': [action],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }

            child_result = self._search_plan(
                request=request,
                current_state=predicted_ap_state,
                init_world_state=init_world_state,
                preceding_ap_trace=new_preceding_ap_trace,
                accepted_trace=accepted_trace + [action],
                depth=depth + 1,
                planning_tree=planning_tree,
                action_help=action_help,
                parent_node_id=node_id,
                inherited_failures=[],
                hint_plan=tail if self._planning_granularity == PLANNING_BATCH else None,
                trace=trace,
                result_path=result_path,
            )
            attempt_trace['child_node_id'] = child_result.get('node_id')
            node['attempts'].append(attempt_trace)
            self._checkpoint_result(trace, result_path)

            if child_result.get('success'):
                attempt_trace['accepted'] = True
                tlc = step_verification.get('prediction_detail', {}).get('tla_verification', {})
                verif = make_verification(
                    passed=step_verification['passed'],
                    properties_checked=tlc.get('properties_checked', []),
                    violations=tlc.get('tlc_result', {}).get('violations', []),
                )
                node_result = (
                    'accepted_with_ignored_violations'
                    if step_verification.get('ignored_by_policy')
                    else 'accepted'
                )
                set_node_outcome(
                    node,
                    result=node_result,
                    action_label=action.get('name', 'unknown'),
                    tool='simulator_action',
                    args=action.get('args', {}),
                    state_after=predicted_ap_state,
                    verification=verif,
                )
                return {
                    'success': True,
                    'plan': [action] + child_result.get('plan', []),
                    'planning_response': response_text,
                    'finish_response': child_result.get('finish_response', finish_response),
                    'node_id': node_id,
                }

            # Child subtree dead — backtrack and try another plan from this
            # node.  Do not ban the first action here: the first action passed
            # local verification, and the repair may need to keep it while
            # changing later actions (for example, moving to a covered object
            # before clearing the blocker at the same x/y coordinate).
            attempt_trace['accepted'] = False
            attempt_trace['child_failure'] = child_result.get('failure')
            failed_attempts.append(child_result.get('failure', {
                'type': 'child_failure',
                'depth': depth + 1,
                'message': 'Child subtree exhausted.',
            }))
            current_hint = []
            continue

        exhaustion_failure = {
            'type': 'branch_exhausted',
            'depth': depth,
            'node_id': node_id,
            'failed_attempts': failed_attempts,
            'message': 'All %d action attempts at this node were exhausted.' % self._max_branch_retries,
        }
        set_node_outcome(node, result='backtracked', failure=exhaustion_failure)
        self._checkpoint_result(trace, result_path)
        return {
            'success': False,
            'failure': exhaustion_failure,
            'finish_response': 'No feasible property-satisfying plan found.',
            'node_id': node_id,
        }

    def _verify_single_step(
        self,
        *,
        action: Dict[str, object],
        current_state: Dict[str, object],
        init_world_state: Dict[str, object],
        preceding_ap_trace: List[Dict[str, bool]],
        accepted_trace: List[Dict[str, object]],
        is_last_step: bool,
    ) -> Dict[str, object]:
        """Verify one action via AP state prediction + TLC.

        Returns a dict with:
          passed              — bool
          predicted_ap_state  — the AP state after the action (if passed)
          failure             — failure dict (if not passed)
          prediction_detail   — raw prediction info for trace logging
        """
        try:
            predicted_world_state, prediction_notes = predict_world_state_after_actions(
                init_world_state,
                accepted_trace + [action],
            )
        except Exception as exc:
            return {
                'passed': False,
                'predicted_ap_state': None,
                'failure': {
                    'type': 'prediction_error',
                    'action': action,
                    'message': str(exc),
                },
                'prediction_detail': {'error': str(exc)},
            }

        predicted_ap_state = self._build_initial_ap_state(predicted_world_state)
        full_ap_trace = preceding_ap_trace + [predicted_ap_state]
        tlc_result = _verify_ap_trace(full_ap_trace, _AP_NAMES, is_complete_trace=is_last_step)
        passed = tlc_result['tlc_result'].get('success') or tlc_result['tlc_result'].get('skipped')

        detail = {
            'prediction_source': 'deterministic_symbolic_replay',
            'prediction_summary': '\n'.join(prediction_notes),
            'predicted_world_state': predicted_world_state,
            'predicted_ap_state': predicted_ap_state,
            'predicted_ap_changes': self._diff_ap_states(current_state, predicted_ap_state),
            'tla_verification': tlc_result,
        }

        precondition_failures = [
            note for note in prediction_notes
            if 'precondition failed' in note.lower()
        ]
        if precondition_failures:
            return {
                'passed': False,
                'predicted_ap_state': predicted_ap_state,
                'failure': {
                    'type': 'action_precondition_failed',
                    'action': action,
                    'message': precondition_failures[-1],
                },
                'prediction_detail': detail,
            }

        if not passed:
            return {
                'passed': False,
                'predicted_ap_state': predicted_ap_state,
                'failure': {
                    'type': 'tla_property_violation',
                    'action': action,
                    'violations': tlc_result['tlc_result'].get('violations', []),
                    'message': 'TLC found property violations after action.',
                },
                'prediction_detail': detail,
            }

        return {
            'passed': True,
            'predicted_ap_state': predicted_ap_state,
            'failure': None,
            'prediction_detail': detail,
        }

    def _planning_system_prompt(self) -> str:
        if self._planning_granularity == PLANNING_STEP:
            return STEP_PREDICTIVE_PLAN_SYSTEM_PROMPT
        return BATCH_PLAN_SYSTEM_PROMPT

    def _plan_instruction(self) -> str:
        if self._planning_granularity == PLANNING_STEP:
            return (
                'Return exactly one next primitive action from the current state. '
                'If the goal is already satisfied, return an empty plan. '
                'Do not return more than one action.'
            )
        return 'Return the complete remaining action sequence from the current state to goal completion.'

    def _property_prompt_section(self) -> str:
        if self._violation_policy == VIOLATION_IGNORE:
            return ''
        return 'Property policy:\n%s\n\nProperties:\n%s\n' % (
            _shared_property_policy_text(self._violation_policy),
            self._property_text,
        )

    def _build_plan_prompt(
        self,
        *,
        request: str,
        action_help: str,
        current_state: Dict[str, object],
        init_world_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
        failed_attempts: List[Dict[str, object]],
        banned_first_actions: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        current_ap_bools = dict(current_state) if isinstance(current_state, dict) else {}
        return PLAN_USER_PROMPT_TEMPLATE.format(
            request=request,
            grounding_verdict=self._grounding_verdict_text(init_world_state, request),
            current_ap_bools_json=self._snapshot_json(current_ap_bools),
            planning_state_summary=self._planning_state_summary(
                init_world_state,
                accepted_trace,
                request=request,
            ),
            accepted_trace_json=self._json_or_none(accepted_trace),
            property_section=self._property_prompt_section(),
            action_help=action_help,
            plan_instruction=self._plan_instruction(),
            failed_attempts_json=self._json_or_none(failed_attempts[-5:]) if failed_attempts else 'None',
            banned_first_actions_json=self._json_or_none(banned_first_actions) if banned_first_actions else 'None',
        )

    def _build_ap_state_prediction_prompt(
        self,
        *,
        current_ap_state: Dict[str, bool],
        action: Dict[str, object],
        init_world_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
    ) -> str:
        ap_catalog_text = '\n'.join(
            self._ap_formula(name) for name in _AP_NAMES
        )
        return AP_STATE_PREDICTION_PROMPT_TEMPLATE.format(
            init_world_state_json=self._snapshot_json(init_world_state),
            accepted_trace_json=self._json_or_none(accepted_trace),
            world_state_delta=self._world_state_delta_summary(init_world_state, accepted_trace),
            current_ap_bools_json=self._snapshot_json(current_ap_state),
            action_json=self._json_or_none(action),
            ap_catalog_text=ap_catalog_text,
        )

    def _execute_grasper_cleanup(self, trace: Dict[str, object]) -> None:
        """After plan execution, bring the grasper to a clean state (raised, open).

        Reads the live world state and runs only what is safe:
          lowered + closed + holding + supported  → open_grasper, raise_grasper
          lowered + closed + holding + unsupported → nothing
          lowered + closed + empty                → raise_grasper
          lowered + open                          → raise_grasper
          raised  + closed + empty                → open_grasper
          raised  + closed + holding              → nothing
          raised  + open                          → nothing

        We avoid cleanup when the grasper is still holding an unsupported
        object. In that state, auto-opening would fail and auto-raising would
        silently keep the object airborne, which is not a safe generic
        "cleanup" action.
        """
        state = self._env.snapshot()
        lowered = bool(state.get('grasper_lowered', False))
        closed = bool(state.get('grasper_closed', False))
        grasped_object = state.get('grasped_object')
        holding = grasped_object is not None
        supported_for_release = self._cleanup_release_supported(state, grasped_object)

        cleanup: List[Dict[str, object]] = []
        cleanup_note = None
        if lowered and closed and holding and supported_for_release:
            cleanup = [
                {'name': 'open_grasper', 'args': {}},
                {'name': 'raise_grasper', 'args': {}},
            ]
        elif holding:
            cleanup_note = (
                'Skipped grasper cleanup because the grasper is still holding '
                'an object that is not safely releasable.'
            )
        elif lowered:
            cleanup = [{'name': 'raise_grasper', 'args': {}}]
        elif closed:
            cleanup = [{'name': 'open_grasper', 'args': {}}]

        if cleanup_note:
            trace.setdefault('cleanup_notes', []).append(cleanup_note)

        for action in cleanup:
            try:
                result_text = self._env.execute_action(action)
            except Exception as exc:
                result_text = 'ERROR: %s' % exc
            trace.setdefault('cleanup_steps', []).append({
                'action': action,
                'result': result_text,
            })

    @staticmethod
    def _cleanup_release_supported(
        state: Dict[str, object],
        grasped_object: object,
    ) -> bool:
        if grasped_object is None:
            return False
        objects = state.get('objects', [])
        held = next(
            (obj for obj in objects if obj.get('obj_id') == grasped_object),
            None,
        )
        if not held:
            return False
        support_id = held.get('resting_on')
        if support_id is None:
            return False
        support = next(
            (obj for obj in objects if obj.get('obj_id') == support_id),
            None,
        )
        return bool(support and support.get('can_support'))

    @staticmethod
    def _build_tree_summary(planning_tree: Dict[str, object]) -> List[Dict[str, object]]:
        return build_tree_summary(planning_tree)

    @staticmethod
    def _zip_accepted_steps(
        accepted_trace: List[Dict[str, object]],
        preceding_ap_trace: List[Dict[str, bool]],
    ) -> List[Dict[str, object]]:
        """Pair each accepted action with the AP state it produced.

        preceding_ap_trace[0] is the state before any accepted action.
        preceding_ap_trace[i+1] is the state after accepted_trace[i].
        Returns a list of {action, state_after} dicts, one per accepted action.
        """
        steps = []
        for i, action in enumerate(accepted_trace):
            ap_after = preceding_ap_trace[i + 1] if i + 1 < len(preceding_ap_trace) else None
            steps.append(make_state_path_entry(
                make_action(
                    action.get('name', 'unknown'),
                    'simulator_action',
                    action.get('args', {}),
                ),
                ap_after,
            ))
        return steps

    @staticmethod
    def _ap_formula(name: str) -> str:
        """Return a Python-style formula string for each AP name."""
        if name.startswith('object_') and '_resting_on_' in name:
            tail = name[len('object_'):]
            obj_id, support = tail.split('_resting_on_')
            return '%s: (object with obj_id==%s).resting_on == %s' % (name, obj_id, support)
        if name.startswith('some_object_resting_on_'):
            support = name[len('some_object_resting_on_'):]
            return '%s: any object has resting_on == %s' % (name, support)
        return '%s: world-state field %s is true' % (name, name)

    @staticmethod
    def _world_state_delta_summary(
        init_world_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
    ) -> str:
        """Narrate what the accepted actions have done to the world state.

        Tracks grasper_lowered, grasper_closed, grasped_object, and per-object
        resting_on by replaying the known simulator effect rules symbolically.
        This bridges the gap between init_world_state and the current predicted
        moment without running the real simulator.
        """
        if not accepted_trace:
            return 'No accepted actions yet — world state is identical to initial.'

        grasper_lowered: bool = bool(init_world_state.get('grasper_lowered', False))
        grasper_closed: bool = bool(init_world_state.get('grasper_closed', False))
        grasped_object = init_world_state.get('grasped_object')

        resting_on: Dict[int, object] = {}
        for obj in init_world_state.get('objects', []):
            if isinstance(obj, dict) and 'obj_id' in obj:
                resting_on[int(obj['obj_id'])] = obj.get('resting_on')

        positions: Dict[int, Dict[str, float]] = {}
        graspable: Dict[int, bool] = {}
        for obj in init_world_state.get('objects', []):
            if isinstance(obj, dict) and 'obj_id' in obj:
                oid = int(obj['obj_id'])
                pos = obj.get('position', {})
                if isinstance(pos, dict):
                    positions[oid] = {'x': pos.get('x', 0.0), 'y': pos.get('y', 0.0)}
                graspable[oid] = bool(obj.get('graspable', False))

        grasper_x: float = 0.0
        grasper_y: float = 0.0

        lines = []
        for action in accepted_trace:
            name = action.get('name', '')
            args = action.get('args', {}) or {}

            if name == 'move_grasper':
                grasper_x = float(args.get('x', grasper_x))
                grasper_y = float(args.get('y', grasper_y))
                if grasped_object is not None:
                    lines.append(
                        'move_grasper(x=%.4f, y=%.4f): grasper (holding obj %s) moved to (%.4f, %.4f).'
                        % (grasper_x, grasper_y, grasped_object, grasper_x, grasper_y)
                    )
                else:
                    lines.append(
                        'move_grasper(x=%.4f, y=%.4f): grasper moved to (%.4f, %.4f), holding nothing.'
                        % (grasper_x, grasper_y, grasper_x, grasper_y)
                    )

            elif name == 'lower_grasper':
                if grasper_lowered:
                    lines.append('lower_grasper: PRECONDITION FAILED (already lowered) — no state change.')
                else:
                    grasper_lowered = True
                    tol = 1e-4
                    obj_below = next(
                        (oid for oid, pos in positions.items()
                         if abs(pos['x'] - grasper_x) <= tol and abs(pos['y'] - grasper_y) <= tol
                         and oid != grasped_object),
                        None,
                    )
                    if grasped_object is not None:
                        old = resting_on.get(int(grasped_object))
                        resting_on[int(grasped_object)] = obj_below
                        lines.append(
                            'lower_grasper: grasper_lowered=true. Held obj %s lowered; '
                            'resting_on: %s → %s.'
                            % (grasped_object, old, obj_below)
                        )
                    else:
                        lines.append(
                            'lower_grasper: grasper_lowered=true. '
                            'Grasper empty; object below at (%.4f, %.4f): %s.'
                            % (grasper_x, grasper_y, obj_below)
                        )

            elif name == 'raise_grasper':
                if not grasper_lowered:
                    lines.append('raise_grasper: PRECONDITION FAILED (already raised) — no state change.')
                else:
                    grasper_lowered = False
                    if grasped_object is not None:
                        old = resting_on.get(int(grasped_object))
                        resting_on[int(grasped_object)] = None
                        lines.append(
                            'raise_grasper: grasper_lowered=false. Held obj %s lifted; '
                            'resting_on: %s → None (airborne).'
                            % (grasped_object, old)
                        )
                    else:
                        lines.append('raise_grasper: grasper_lowered=false. Grasper empty, raised.')

            elif name == 'close_grasper':
                if grasper_closed:
                    lines.append('close_grasper: PRECONDITION FAILED (already closed) — no state change.')
                else:
                    grasper_closed = True
                    if grasper_lowered:
                        tol = 1e-4
                        obj_below = next(
                            (oid for oid, pos in positions.items()
                             if abs(pos['x'] - grasper_x) <= tol and abs(pos['y'] - grasper_y) <= tol),
                            None,
                        )
                        if obj_below is not None and graspable.get(obj_below, False):
                            grasped_object = obj_below
                            lines.append(
                                'close_grasper: grasper_closed=true. '
                                'Grasper was lowered onto graspable obj %s → grasped_object=%s.'
                                % (obj_below, grasped_object)
                            )
                        else:
                            lines.append(
                                'close_grasper: grasper_closed=true. '
                                'Grasper lowered but no graspable object at (%.4f, %.4f) → grasped_object stays null.'
                                % (grasper_x, grasper_y)
                            )
                    else:
                        lines.append(
                            'close_grasper: grasper_closed=true. '
                            'Grasper not lowered → grasped_object stays null.'
                        )

            elif name == 'open_grasper':
                if not grasper_closed:
                    lines.append('open_grasper: PRECONDITION FAILED (already open) — no state change.')
                else:
                    if grasped_object is not None:
                        if not grasper_lowered:
                            lines.append(
                                'open_grasper: PRECONDITION FAILED — holding obj %s but grasper not lowered. '
                                'No state change.'
                                % grasped_object
                            )
                        else:
                            support = resting_on.get(int(grasped_object))
                            if support is None:
                                lines.append(
                                    'open_grasper: PRECONDITION FAILED — holding obj %s but resting_on is null '
                                    '(no support). No state change.'
                                    % grasped_object
                                )
                            else:
                                lines.append(
                                    'open_grasper: grasper_closed=false. '
                                    'Released obj %s; it stays resting_on=%s. grasped_object=null.'
                                    % (grasped_object, support)
                                )
                                grasped_object = None
                                grasper_closed = False
                    else:
                        grasper_closed = False
                        lines.append('open_grasper: grasper_closed=false. Was not holding anything.')

        resting_on_changes = []
        for obj in init_world_state.get('objects', []):
            if not isinstance(obj, dict):
                continue
            oid = int(obj.get('obj_id', -1))
            if oid < 0:
                continue
            init_ro = obj.get('resting_on')
            curr_ro = resting_on.get(oid, init_ro)
            if curr_ro != init_ro:
                resting_on_changes.append('  obj %d: resting_on %s → %s' % (oid, init_ro, curr_ro))

        summary_lines = [
            'After %d accepted action(s):' % len(accepted_trace),
        ] + lines + [
            'Current inferred state:',
            '  grasper_lowered: %s' % grasper_lowered,
            '  grasper_closed: %s' % grasper_closed,
            '  grasped_object: %s' % grasped_object,
        ]
        if resting_on_changes:
            summary_lines.append('  resting_on changes from initial:')
            summary_lines.extend(resting_on_changes)
        else:
            summary_lines.append('  resting_on: no changes from initial state.')
        return '\n'.join(summary_lines)

    def _request_ap_state_prediction(
        self,
        history: List[Dict[str, str]],
        current_ap_state: Dict[str, bool],
    ):
        content = self._chat(list(history), schema=AP_STATE_SCHEMA).strip()
        bundle = self._parse_ap_state_prediction(content, current_ap_state)
        attempt_log = [{'raw_content': content, 'parsed_prediction': bundle}]
        return content, bundle, attempt_log

    def _parse_ap_state_prediction(
        self,
        content: str,
        current_ap_state: Dict[str, bool],
    ) -> Dict[str, object]:
        """Parse the predicted AP state with a guaranteed result.

        Extraction order: JSON parse → regex scan → fallback to current state.
        Any AP not recovered from the model output is copied from current_ap_state.
        """
        extracted: Dict[str, bool] = {}
        response_text = ''

        try:
            json_content = OpenAICompatibleShrdluAgent._extract_json_object(content)
            decision = json.loads(json_content)
            if isinstance(decision, dict):
                ap_results = decision.get('ap_results')
                if isinstance(ap_results, dict):
                    for name in _AP_NAMES:
                        val = ap_results.get(name)
                        if isinstance(val, bool):
                            extracted[name] = val
                response_text = str(decision.get('response', ''))
        except (json.JSONDecodeError, ValueError):
            pass

        if len(extracted) < len(_AP_NAMES):
            for name in _AP_NAMES:
                if name in extracted:
                    continue
                pattern = r'"' + re.escape(name) + r'"\s*:\s*(true|false)'
                m = re.search(pattern, content)
                if m:
                    extracted[name] = m.group(1) == 'true'

        for name in _AP_NAMES:
            if name not in extracted:
                extracted[name] = bool(current_ap_state.get(name, False))

        return {
            'response': response_text,
            'ap_results': {name: extracted[name] for name in _AP_NAMES},
        }

    @staticmethod
    def _collect_violated_properties(failure: Optional[Dict[str, object]]) -> List[str]:
        """Recursively collect all unique property IDs that were violated in a failure tree.

        TLC violation entries are raw stdout lines such as:
          'Error: Property Property_prop_foo_bar is violated.'
        The shared parser handles both dotted ids and TLC-safe underscored ids.
        """
        if not failure:
            return []
        seen: set[str] = set()

        def _walk(f):
            if not isinstance(f, dict):
                return
            if f.get('type') == 'tla_property_violation':
                seen.update(extract_property_ids_from_violations(f.get('violations', [])))
            for v in f.get('failed_attempts', []):
                _walk(v)
            if f.get('child_failure'):
                _walk(f['child_failure'])

        _walk(failure)
        return sorted(seen)

    @staticmethod
    def _diff_ap_states(
        previous: Dict[str, bool],
        current: Dict[str, bool],
    ) -> List[Dict[str, object]]:
        changes = []
        for name in _AP_NAMES:
            prev_val = previous.get(name)
            curr_val = current.get(name)
            if prev_val is not None and prev_val != curr_val:
                changes.append({'name': name, 'before': prev_val, 'after': curr_val})
        return changes

    def _build_initial_ap_state(self, world_state: Dict[str, object]) -> Dict[str, bool]:
        return observe_ap_values(
            _AP_NAMES,
            lambda name: self._property_verifier.observe_ap(name, world_state),
        )

    def _build_initial_property_state(self, world_state: Dict[str, object], scene=None) -> Dict[str, object]:
        """Backward-compatible property-state bundle used by older tests/tools."""
        del scene
        ap_state = self._build_initial_ap_state(world_state)
        return {
            **ap_state,
            'derived_aps': dict(ap_state),
            'property_results': [
                {
                    'id': name,
                    'satisfied': satisfied,
                }
                for name, satisfied in sorted(ap_state.items())
            ],
        }

    @staticmethod
    def _parse_plan(content: str) -> Dict[str, object]:
        _ALLOWED_PRIMITIVE_NAMES = ALLOWED_SIMULATOR_ACTION_NAMES
        content = OpenAICompatibleShrdluAgent._extract_json_object(content)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model did not return valid JSON: %s" % content) from exc
        if not isinstance(decision, dict):
            raise ValueError("Model reply must be a JSON object.")
        if 'plan' not in decision:
            raise ValueError("Model reply must include a plan array.")
        raw_plan = decision.get('plan', [])
        if not isinstance(raw_plan, list):
            raise ValueError("Model reply must include a plan array.")
        plan = []
        for item in raw_plan:
            if not isinstance(item, dict):
                raise ValueError("Each planned action must be a JSON object.")
            normalized = OpenAICompatibleShrdluAgent._normalize_action({'action': item})
            name = str(normalized.get('name', '')).strip()
            if name not in _ALLOWED_PRIMITIVE_NAMES:
                raise ValueError(
                    "Action name %r is not a primitive action. Allowed names: %s"
                    % (name, ', '.join(sorted(_ALLOWED_PRIMITIVE_NAMES)))
                )
            plan.append({
                'name': name,
                'args': normalized.get('args', {}) if isinstance(normalized.get('args', {}), dict) else {},
            })
        return {
            'response': str(decision.get('response', '')),
            'finish_response': str(decision.get('finish_response', '')),
            'plan': plan,
        }

    def _request_plan(self, history: List[Dict[str, str]]):
        attempts = list(history)
        errors = []
        attempt_log = []
        for attempt_index in range(2):
            content = self._chat(attempts, schema=PLAN_SCHEMA).strip()
            try:
                plan_bundle = self._parse_plan(content)
            except ValueError as exc:
                errors.append(str(exc))
                attempt_log.append({
                    'attempt_index': attempt_index,
                    'raw_content': content,
                    'error': str(exc),
                })
                if attempt_index == 1:
                    break
                attempts.extend([
                    {'role': 'assistant', 'content': content},
                    {
                        'role': 'user',
                        'content': PLAN_REPAIR_PROMPT_TEMPLATE.format(error=exc),
                    },
                ])
                continue
            attempt_log.append({
                'attempt_index': attempt_index,
                'raw_content': content,
                'parsed_plan': plan_bundle,
            })
            return content, plan_bundle, attempt_log
        raise ValueError('Invalid plan reply after retry: %s' % errors[-1])


class FsmOpenAICompatibleShrdluAgent(
        _FsmShrdluAgentMixin, OpenAICompatibleShrdluAgent):
    """Merged FSM/planning agent over a local OpenAI API."""

    def __init__(self, env: SimulatorAPI, model: str = DEFAULT_OPENAI_MODEL,
                 base_url: str = DEFAULT_OPENAI_BASE_URL,
                 api_key: str = DEFAULT_OPENAI_API_KEY,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 512,
                 client=None,
                 max_branch_retries: int = 3,
                 planning_granularity: str = PLANNING_BATCH,
                 violation_policy: str = VIOLATION_RETRY,
                 result_dir: Optional[str] = DEFAULT_RESULT_DIR):
        super().__init__(
            env,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            result_dir=result_dir,
            trace_dir=trace_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
        )
        self._planning_granularity = _normalize_planning_granularity(planning_granularity)
        self._violation_policy = _normalize_violation_policy(violation_policy)
        retries = int(max_branch_retries)
        if self._violation_policy in NONBLOCKING_VIOLATION_POLICIES:
            retries = max(1, retries)
        self._init_fsm_planner(max_branch_retries=retries)
