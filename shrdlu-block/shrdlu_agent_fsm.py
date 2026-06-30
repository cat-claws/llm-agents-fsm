"""OpenAI-compatible merged FSM/planning agents for the SHRDLU blocks environment."""

from __future__ import annotations

import copy
import json
import re
import sys
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
    property_guidance_text,
    property_policy_text as _shared_property_policy_text,
)
from utils.planning_terminal import RuntimePlanningConfig, runtime_config_from_values
from utils.property_catalog import (
    aps_from_properties,
    load_property_catalog,
    observe_ap_values,
)
from utils.agent_planning import (
    AgentConfig,
    AgentFlowSpec,
    run_verification,
    run_agent_flow,
)
from utils.session import (
    make_verification,
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

_RESOURCES_DIR = Path(__file__).resolve().parent / 'resources'
_TLA_PROPERTIES_FILE = _RESOURCES_DIR / 'SHRDLU_PROPERTIES_AST.json'
_AP_CANDIDATES_FILE = _RESOURCES_DIR / 'SHRDLU_AP_CANDIDATES.json'
_TLA_PROPERTIES = load_property_catalog(_TLA_PROPERTIES_FILE)
_AP_CATALOG: Dict[str, object] = json.loads(_AP_CANDIDATES_FILE.read_text(encoding='utf-8'))
_AP_CATALOG_METADATA: Dict[str, object] = dict(_AP_CATALOG.get('metadata', {}))
_AP_SPEC_BY_NAME: Dict[str, Dict[str, object]] = {
    str(spec.get('name')): spec
    for spec in _AP_CATALOG.get('current_state_aps', [])
    if isinstance(spec, dict) and spec.get('name')
}
_STATE_AP_NAMES, _TRANSITION_AP_NAMES = aps_from_properties(_TLA_PROPERTIES)
_AP_NAMES: List[str] = _STATE_AP_NAMES + _TRANSITION_AP_NAMES

__all__ = [
    'FsmOpenAICompatibleShrdluAgent',
]


def _noop_llm_call(*_args, **_kwargs):
    return '', []


def _noop_tool_arguments(*_args, **_kwargs):
    return None


def _empty_action_prediction_notes(_tool: str, _args: dict) -> str:
    return ''


def _shared_action_to_simulator(step: Dict[str, object]) -> Dict[str, object]:
    return {
        'name': str(step.get('action_label') or step.get('label') or 'unknown'),
        'args': step.get('args', {}) if isinstance(step.get('args'), dict) else {},
    }


def _simulator_action_to_shared(action: Dict[str, object]) -> Dict[str, object]:
    return {
        'action_label': str(action.get('name', 'unknown')),
        'tool': 'simulator_action',
        'args': action.get('args', {}) if isinstance(action.get('args'), dict) else {},
    }


def _shared_trace_to_simulator(trace: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [_shared_action_to_simulator(step) for step in trace]


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

class _FsmShrdluAgentMixin:
    """Plan before execution with configurable granularity and property policy."""

    def _init_fsm_planner(self, max_branch_retries: int = 3):
        self._max_branch_retries = int(max_branch_retries)
        self._property_text = property_guidance_text(_TLA_PROPERTIES)

    def get_runtime_planning_config(self, retry_default: int | None = None) -> RuntimePlanningConfig:
        """Return the live planning settings for terminal mode controls."""
        default = int(retry_default if retry_default is not None else self._max_branch_retries)
        return runtime_config_from_values(
            planning_granularity=self._planning_granularity,
            violation_policy=self._violation_policy,
            max_retries=self._max_branch_retries,
            retry_default=default,
            max_steps=self._max_steps,
        )

    def set_runtime_planning_config(
        self,
        config: RuntimePlanningConfig,
        retry_default: int | None = None,
    ) -> RuntimePlanningConfig:
        """Update live FSM planning settings and return the normalized config."""
        del retry_default
        self._planning_granularity = _normalize_planning_granularity(config.planning_granularity)
        self._violation_policy = _normalize_violation_policy(config.violation_policy)
        retries = int(config.max_retries)
        if self._violation_policy in NONBLOCKING_VIOLATION_POLICIES:
            retries = max(1, retries)
        self._init_fsm_planner(max_branch_retries=retries)
        return self.get_runtime_planning_config()

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

    def _property_monitoring_metadata(self) -> Dict[str, object]:
        return {
            'enabled': True,
            'property_file': str(_TLA_PROPERTIES_FILE),
            'property_count': len(_TLA_PROPERTIES),
            'ap_source': 'properties_ast',
            'ap_count': len(_AP_NAMES),
        }

    def _run_agent_loop(self, request: str) -> str:
        response, _turn = run_agent_flow(
            goal=request,
            model=self._model,
            config=self._agent_config(),
            spec=self._agent_flow_spec(),
        )
        return response

    def _agent_config(self) -> AgentConfig:
        return AgentConfig(
            planning_granularity=self._planning_granularity,
            violation_policy=self._violation_policy,
            max_plan_steps=self._max_steps,
            max_retries=self._max_branch_retries,
        )

    def _agent_flow_spec(self) -> AgentFlowSpec:
        self._shared_initial_world_state = self._env.snapshot()
        self._shared_action_help = self._env.action_help()
        self._last_planning_response = ''
        self._last_finish_response = ''
        return AgentFlowSpec(
            agent='shrdlu-agent-fsm',
            domain='shrdlu',
            work_dir=self._host,
            properties=_TLA_PROPERTIES,
            aps=_AP_NAMES,
            result_dir=self._result_dir,
            verification_module_name='ShrdluTrace',
            verification_timeout=60,
            observe_ap_for_model=lambda _model: self._observe_live_ap,
            execute_step=self._execute_shared_step,
            summarize_result=self._summarize_shared_result,
            explain_blocked=self._explain_shared_blocked,
            llm_call=_noop_llm_call,
            client=None,
            tool_arguments=_noop_tool_arguments,
            max_planning_tokens=self._max_tokens,
            propose_step_prompt=STEP_PREDICTIVE_PLAN_SYSTEM_PROMPT,
            propose_batch_prompt=BATCH_PLAN_SYSTEM_PROMPT,
            predict_ap_prompt='',
            action_proposal_tool=[],
            action_proposal_tool_name='propose_shrdlu_action',
            plan_proposal_tool=[],
            plan_proposal_tool_name='propose_shrdlu_plan',
            ap_prediction_tool=[],
            ap_prediction_tool_name='predict_shrdlu_ap',
            ap_spec_by_name=_AP_SPEC_BY_NAME,
            ap_catalog_metadata=_AP_CATALOG_METADATA,
            ap_evidence_field='evaluation',
            action_prediction_notes=_empty_action_prediction_notes,
            already_satisfied_response='Done.',
            initial_world_state=lambda: copy.deepcopy(self._shared_initial_world_state),
            action_help=lambda: self._shared_action_help,
            planning_config_extra=self._shared_planning_config_extra,
            request_plan_override=self._request_shared_plan,
            verify_action_override=self._verify_shared_action,
            after_execute=self._execute_grasper_cleanup,
            result_path_callback=self._remember_result_path,
        )

    def _shared_planning_config_extra(self) -> Dict[str, object]:
        return {
            'host': self._host,
            'max_steps': self._max_steps,
            'max_branch_retries': self._max_branch_retries,
            'property_monitoring': self._property_monitoring_metadata(),
        }

    def _remember_result_path(self, result_path: Optional[str]) -> None:
        if result_path is not None:
            self._last_result_path = result_path

    def _observe_live_ap(self, name: str) -> bool:
        return self._property_verifier.observe_ap(name, self._env.snapshot())

    def _execute_shared_step(self, step: Dict[str, object]) -> str:
        action = _shared_action_to_simulator(step)
        try:
            return self._env.execute_action(action)
        except Exception as exc:
            return 'ERROR: %s' % exc

    def _request_shared_plan(
        self,
        *,
        goal: str,
        current_state: Dict[str, bool],
        accepted_trace: List[Dict[str, object]],
        failed_attempts: List[Dict[str, object]],
        banned_first_actions: List[str],
        depth: int,
        model: str,
        config: AgentConfig,
    ) -> Dict[str, object]:
        del depth, model, config
        accepted_actions = _shared_trace_to_simulator(accepted_trace)
        banned_actions = [
            {'name': label, 'args': {}}
            for label in banned_first_actions
        ]
        plan_prompt = self._build_plan_prompt(
            request=goal,
            action_help=self._shared_action_help,
            current_state=current_state,
            init_world_state=self._shared_initial_world_state,
            accepted_trace=accepted_actions,
            failed_attempts=failed_attempts,
            banned_first_actions=banned_actions,
        )
        history = [
            {'role': 'system', 'content': self._planning_system_prompt()},
            {'role': 'user', 'content': plan_prompt},
        ]
        content, plan_bundle, attempts = self._request_plan(history)
        self._last_planning_response = self._normalize_response_text(
            plan_bundle.get('response', ''),
            is_finish=not plan_bundle.get('plan'),
        )
        self._last_finish_response = self._normalize_response_text(
            plan_bundle.get('finish_response', 'Done.'),
            is_finish=True,
        )
        return {
            'response': plan_bundle.get('response', ''),
            'finish_response': self._last_finish_response,
            'plan': [
                _simulator_action_to_shared(action)
                for action in plan_bundle.get('plan', [])
            ],
            'planner_prompt': plan_prompt,
            'planner_response': content,
            'planner_attempts': attempts,
        }

    def _verify_shared_action(
        self,
        *,
        proposal: Dict[str, object],
        s0: Dict[str, bool],
        s_current: Dict[str, bool],
        trace: List[Dict[str, object]],
        model: str,
        is_complete_trace: bool = False,
    ) -> Dict[str, object]:
        del model
        action = _shared_action_to_simulator(proposal)
        accepted_actions = _shared_trace_to_simulator(trace)
        preceding_ap_trace = [s0] + [
            step.get('state_after', {})
            for step in trace
            if isinstance(step.get('state_after'), dict)
        ]
        step_verification = self._verify_single_step(
            action=action,
            current_state=s_current,
            init_world_state=self._shared_initial_world_state,
            preceding_ap_trace=preceding_ap_trace,
            accepted_trace=accepted_actions,
            is_last_step=is_complete_trace,
        )
        predicted_ap_state = step_verification.get('predicted_ap_state') or dict(s_current)
        candidate = {
            **_simulator_action_to_shared(action),
            'state_before': dict(s_current),
            'state_after': predicted_ap_state,
        }
        tlc = step_verification.get('prediction_detail', {}).get('tla_verification', {})
        raw_tlc = tlc.get('tlc_result', {}) if isinstance(tlc, dict) else {}
        failure = step_verification.get('failure')
        violations = []
        if isinstance(failure, dict):
            violations = failure.get('violations', []) or []
        if not violations and isinstance(raw_tlc, dict):
            violations = raw_tlc.get('violations', []) or []
        verif = make_verification(
            passed=bool(step_verification.get('passed')),
            properties_checked=tlc.get('properties_checked', []) if isinstance(tlc, dict) else [],
            violations=violations,
            skipped=bool(raw_tlc.get('skipped')) if isinstance(raw_tlc, dict) else False,
        )
        return {
            'candidate': candidate,
            'state_after': predicted_ap_state,
            'passed': bool(step_verification.get('passed')),
            'violations_str': '; '.join(str(v) for v in violations),
            'verification': verif,
            'failure': failure,
        }

    def _summarize_shared_result(
        self,
        goal: str,
        trace: List[Dict[str, object]],
        exec_results: List[str],
        model: str,
    ) -> str:
        del goal, trace, exec_results, model
        response_text = self._last_planning_response or 'Verified plan ready.'
        finish_response = self._last_finish_response or 'Done.'
        if response_text == finish_response:
            return finish_response
        return self._format_reply(response_text, finish_response)

    def _explain_shared_blocked(
        self,
        goal: str,
        initial_state: Dict[str, bool],
        tried_actions: List[str],
        model: str,
    ) -> str:
        del goal, initial_state, model
        suffix = ''
        if tried_actions:
            suffix = '\nTried actions: %s' % ', '.join(tried_actions)
        return 'No feasible property-satisfying plan found.' + suffix

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
                    'blocking': True,
                },
                'prediction_detail': {'error': str(exc)},
            }

        predicted_ap_state = self._build_initial_ap_state(predicted_world_state)
        states_after = preceding_ap_trace[1:] + [predicted_ap_state]
        verification_trace = [
            {
                'action_label': str(step.get('name', 'unknown')),
                'state_after': state_after,
            }
            for step, state_after in zip(accepted_trace + [action], states_after)
        ]
        _passed, _tla_spec, _summary, tlc_result = run_verification(
            preceding_ap_trace[0],
            verification_trace,
            aps=_AP_NAMES,
            properties=_TLA_PROPERTIES,
            module_name='ShrdluTrace',
            timeout=60,
            is_complete_trace=is_last_step,
        )
        passed = tlc_result['passed']

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
                    'blocking': True,
                },
                'prediction_detail': detail,
            }

        if not passed:
            raw_tlc_result = tlc_result.get('tlc_result', {})
            skipped = bool(raw_tlc_result.get('skipped'))
            if skipped:
                message = raw_tlc_result.get('reason') or 'TLC verification was skipped.'
                violations = []
                failure_type = 'verification_skipped'
            else:
                message = 'TLC found property violations after action.'
                violations = raw_tlc_result.get('violations', [])
                failure_type = 'tla_property_violation'
            return {
                'passed': False,
                'predicted_ap_state': predicted_ap_state,
                'failure': {
                    'type': failure_type,
                    'action': action,
                    'violations': violations,
                    'message': message,
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
    def _ap_formula(name: str) -> str:
        """Return the catalog-defined evaluation rule for one AP."""
        spec = _AP_SPEC_BY_NAME.get(name)
        if not isinstance(spec, dict):
            raise ValueError('AP is missing from SHRDLU_AP_CANDIDATES.json: %s' % name)
        evaluation = spec.get('evaluation')
        if not isinstance(evaluation, dict):
            raise ValueError('AP has no evaluation metadata: %s' % name)
        templates = _AP_CATALOG_METADATA['ap_formula_templates']
        eval_type = str(evaluation.get('type', ''))
        template = templates.get(eval_type) if isinstance(templates, dict) else None
        if not isinstance(template, str):
            raise ValueError('Unsupported AP evaluation type for %s: %s' % (name, eval_type))
        return template.format(name=name, **evaluation)

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
