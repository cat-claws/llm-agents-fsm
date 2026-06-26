"""OpenAI-compatible predictive preplanned agents for the SHRDLU blocks environment."""

import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from utils.session import SCHEMA_VERSION  # noqa: E402

from shrdlu_agents.shrdlu_agent_basic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACE_DIR,
    OpenAICompatibleShrdluAgent,
)
from shrdlu_agents.simulator_api import SimulatorAPI
from shrdlu_blocks.simulator.state_pred import predict_world_state_after_actions

__all__ = [
    'PredictivePreplannedOpenAICompatibleShrdluAgent',
]


PREDICTIVE_STEP_SYSTEM_PROMPT = """You are planning a SHRDLU blocks-world task one action at a time before execution.

Rules:
- Plan only the single next action for the current predicted state.
- Treat your answer as a dry-run primitive simulator call, not a multi-step plan.
- Use only one allowed primitive action name and exactly the matching JSON args.
- Never invent argument names that are not listed in the allowed action schema.
- Never use null, placeholders, or descriptive strings where a concrete numeric or integer argument is required.
- Ground every action argument from the structured planning state summary or current predicted state JSON.
- Read candidate_source_objects and candidate_destination_objects before scanning the larger object lists.
- Base the next action on the goal, the current predicted state, the accepted action trace so far, and any failed attempts shown to you.
- Prefer the shortest valid progress toward the goal.
- If one valid primitive action directly advances or completes the goal, choose that action instead of a detour.
- Match the action family to the goal: for highlight goals prefer highlight_object or unhighlight_object; for movement/stacking goals prefer grasper and motion actions.
- Do not choose grasper manipulation actions for a pure highlighting goal unless the goal explicitly requires moving or grasping objects.
- For pick/place or stacking goals, identify one concrete source object from source_candidates and one concrete destination object from destination_candidates before choosing the next primitive action.
- Match every user-mentioned destination attribute that exists in the scene representation, including kind, color, size, height, width, and supportability when relevant.
- Bind all user-mentioned attributes to one single object. For example, "green small block" means one object with color=green, kind=block, and size=small; it does not mean a green object on top of a small block.
- Never combine attributes across stacked or co-located objects.
- For pick/place or stacking goals, ground move_grasper(x, y) by copying the exact x and y of the chosen source or destination object.
- For pick/place or stacking goals, never choose open_grasper unless the current predicted state shows the held object is lowered onto a destination object that can support it.
- Do not undo a previously accepted action unless the current predicted state makes that reversal necessary for the goal.
- Avoid repeated alternating patterns such as lower_grasper -> raise_grasper -> lower_grasper or close_grasper -> open_grasper -> close_grasper unless they are strictly required.
- Do not repeat the exact same primitive action with the exact same args if the current predicted state already reflects that action's effect.
- If the goal is already satisfied in the current predicted state, return the finish action immediately.
- Do not output more than one action.
- Use the grounding context as planning input, not as a user-facing refusal template.
- Do not explain alternatives or reasoning.
- Keep the response short and factual.

Return strict JSON only.
"""

PREDICTIVE_STATE_SYSTEM_PROMPT = """State prediction is performed by an exact shadow simulator in Python."""

PREDICTED_STATE_SCHEMA = {
    'type': 'object',
    'properties': {
        'response': {
            'type': 'string',
        },
        'predicted_state': {
            'type': 'object',
        },
    },
    'required': ['response', 'predicted_state'],
}


class _PredictivePreplannedShrdluAgentMixin:
    """Plan via action prediction and backtracking before executing once."""

    def _init_predictive_preplanned(self, max_branch_retries: int = 3):
        self._max_branch_retries = int(max_branch_retries)
        self._property_text = self._build_property_text()

    def _run_agent_loop(self, request: str) -> str:
        trace = {
            'schema_version': SCHEMA_VERSION,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'agent': 'shrdlu-agent-plan',
            'domain': 'shrdlu',
            'model': self._model,
            'host': self._host,
            'max_steps': self._max_steps,
            'request': request,
            'planning_mode': 'predictive_preplanned',
            'max_branch_retries': self._max_branch_retries,
            'property_monitoring': self._property_monitoring_metadata(),
            'planning_tree': {
                'mode': 'predictive_preplanned',
                'max_steps': self._max_steps,
                'max_branch_retries': self._max_branch_retries,
                'properties': [
                    {
                        'id': item.get('id'),
                        'natural_language': item.get('natural_language'),
                    }
                    for item in self._property_verifier.properties
                ],
                'nodes': [],
                'feasible': False,
                'accepted_plan': [],
            },
            'steps': [],
        }
        initial_state = self._env.snapshot()
        action_help = self._env.action_help()
        trace['planning_tree']['initial_state'] = initial_state
        trace['planning_tree']['action_help'] = action_help
        trace['status'] = 'planning'
        trace_path = self._start_trace_session(trace)

        result = self._search_plan(
            request=request,
            current_state=initial_state,
            current_scene=copy.deepcopy(initial_state),
            accepted_trace=[],
            depth=0,
            previous_property_status=None,
            planning_tree=trace['planning_tree'],
            action_help=action_help,
            parent_node_id=None,
            inherited_failures=[],
            trace=trace,
            trace_path=trace_path,
        )

        trace['planning_tree']['feasible'] = bool(result.get('success'))
        trace['planning_tree']['accepted_plan'] = result.get('plan', []) if result.get('success') else []
        trace['planning_tree']['finish_response'] = result.get('finish_response')
        trace['planning_tree']['planning_response'] = result.get('planning_response')
        if result.get('failure'):
            trace['planning_tree']['failure'] = result['failure']

        if not result.get('success'):
            final_message = self._normalize_response_text(
                result.get('finish_response', 'No feasible property-satisfying plan found.'),
                is_finish=True,
            )
            trace['status'] = 'infeasible'
            trace['final_message'] = final_message
            trace_path = self._write_trace(trace, trace_path)
            return self._append_trace_notice(final_message, trace_path)

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
            self._write_trace(trace, trace_path)
            return finish_response if response_text == finish_response else self._format_reply(
                response_text,
                finish_response,
            )

        previous_property_status = None
        trace['status'] = 'executing'
        self._checkpoint_trace(trace, trace_path)
        for step_index, action in enumerate(plan):
            step_trace = {
                'step_index': step_index,
                'planned_action': action,
            }
            pre_state = self._env.snapshot()
            pre_scene = copy.deepcopy(getattr(self._env, 'scene', None))
            try:
                result_text = self._env.execute_action(action)
            except Exception as exc:
                result_text = "ERROR: %s" % exc
            post_state = self._env.snapshot()
            property_trace, previous_property_status = self._monitor_transition_properties(
                pre_state,
                action,
                post_state,
                pre_scene=pre_scene,
                post_scene=getattr(self._env, 'scene', None),
                previous_property_status=previous_property_status,
            )
            step_trace.update({
                'action_result': result_text,
                'property_verification': property_trace,
                'observation_after': self._env.snapshot_text(),
            })
            trace['steps'].append(step_trace)
            self._checkpoint_trace(trace, trace_path)
            if isinstance(result_text, str) and result_text.startswith('ERROR:'):
                final_message = self._format_reply(
                    response_text + "\n\nPlan execution failed.",
                    "Executed %s.\nResult: %s" % (self._format_action(action), result_text),
                )
                trace['status'] = 'error'
                trace['final_message'] = final_message
                trace_path = self._write_trace(trace, trace_path)
                return self._append_trace_notice(final_message, trace_path)

        final_message = finish_response
        if response_text != finish_response:
            final_message = self._format_reply(response_text, finish_response)
        trace['status'] = 'finished'
        trace['final_message'] = final_message
        self._write_trace(trace, trace_path)
        return final_message

    def _search_plan(
        self,
        *,
        request: str,
        current_state: Dict[str, object],
        current_scene,
        accepted_trace: List[Dict[str, object]],
        depth: int,
        previous_property_status: Optional[Dict[str, bool]],
        planning_tree: Dict[str, object],
        action_help: str,
        parent_node_id: Optional[int],
        inherited_failures: List[Dict[str, object]],
        trace: Dict[str, object],
        trace_path: Optional[str],
    ) -> Dict[str, object]:
        if depth >= self._max_steps:
            return {
                'success': False,
                'failure': {
                    'type': 'max_depth',
                    'depth': depth,
                    'message': 'Planning exceeded the max step budget before reaching finish.',
                },
                'finish_response': 'No feasible property-satisfying plan found.',
            }

        node_id = len(planning_tree['nodes'])
        node = {
            'node_id': node_id,
            'parent_node_id': parent_node_id,
            'depth': depth,
            'accepted_trace': copy.deepcopy(accepted_trace),
            'current_state': copy.deepcopy(current_state),
            'attempts': [],
            'children': [],
            'result': 'searching',
        }
        planning_tree['nodes'].append(node)
        self._checkpoint_trace(trace, trace_path)
        failed_attempts = list(inherited_failures)

        for retry_index in range(self._max_branch_retries):
            step_prompt = self._build_next_action_prompt(
                request=request,
                action_help=action_help,
                current_state=current_state,
                accepted_trace=accepted_trace,
                failed_attempts=failed_attempts,
            )
            history = [
                {'role': 'system', 'content': PREDICTIVE_STEP_SYSTEM_PROMPT},
                {'role': 'user', 'content': step_prompt},
            ]
            try:
                content, decision, attempts = self._request_decision(history)
            except Exception as exc:
                failure = {
                    'type': 'planning_error',
                    'depth': depth,
                    'retry_index': retry_index,
                    'message': str(exc),
                }
                node['attempts'].append({
                    'retry_index': retry_index,
                    'planner_prompt': step_prompt,
                    'error': str(exc),
                })
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                continue

            action = decision.get('action', {})
            response_text = self._normalize_response_text(
                decision.get('response', ''),
                is_finish=action.get('name') == 'finish',
            )
            attempt_trace = {
                'retry_index': retry_index,
                'planner_prompt': step_prompt,
                'planner_attempts': attempts,
                'planner_response': content,
                'planner_decision': decision,
            }

            if action.get('name') == 'finish':
                attempt_trace['accepted'] = True
                attempt_trace['finish'] = True
                node['attempts'].append(attempt_trace)
                node['result'] = 'finish'
                self._checkpoint_trace(trace, trace_path)
                node['finish_response'] = response_text
                return {
                    'success': True,
                    'plan': [],
                    'planning_response': response_text,
                    'finish_response': response_text,
                    'node_id': node_id,
                }

            try:
                simulated = self._simulate_action_transition(current_scene, action)
            except Exception as exc:
                failure = {
                    'type': 'prediction_error',
                    'depth': depth,
                    'retry_index': retry_index,
                    'action': action,
                    'message': str(exc),
                }
                attempt_trace['shadow_simulation_error'] = str(exc)
                attempt_trace['accepted'] = False
                node['attempts'].append(attempt_trace)
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                continue

            property_trace, predicted_property_status = self._monitor_transition_properties(
                current_state,
                action,
                simulated['post_state'],
                pre_scene=simulated['pre_scene'],
                post_scene=simulated['post_scene'],
                previous_property_status=previous_property_status,
            )
            attempt_trace.update({
                'shadow_result': simulated['action_result'],
                'predicted_property_changes': property_trace.get('changed_properties', []),
                'property_verification': property_trace,
            })

            if property_trace.get('all_satisfied', False):
                child_result = self._search_plan(
                    request=request,
                    current_state=simulated['post_state'],
                    current_scene=simulated['post_scene'],
                    accepted_trace=accepted_trace + [action],
                    depth=depth + 1,
                    previous_property_status=predicted_property_status,
                    planning_tree=planning_tree,
                    action_help=action_help,
                    parent_node_id=node_id,
                    inherited_failures=[],
                    trace=trace,
                    trace_path=trace_path,
                )
                if child_result.get('success'):
                    attempt_trace['accepted'] = True
                    attempt_trace['child_node_id'] = child_result.get('node_id')
                    node['attempts'].append(attempt_trace)
                    self._checkpoint_trace(trace, trace_path)
                    if child_result.get('node_id') is not None:
                        node['children'].append(child_result['node_id'])
                    node['result'] = 'accepted'
                    node['accepted_action'] = action
                    node['accepted_predicted_state'] = simulated['post_state']
                    node['verification'] = {
                        'passed': True,
                        'properties_checked': [
                            r.get('id') for r in property_trace.get('property_results', [])
                        ],
                        'violations': [],
                        'skipped': False,
                    }
                    return {
                        'success': True,
                        'plan': [action] + child_result.get('plan', []),
                        'planning_response': response_text if depth == 0 else child_result.get('planning_response'),
                        'finish_response': child_result.get('finish_response', 'Done.'),
                        'node_id': node_id,
                    }
                failure = child_result.get('failure', {
                    'type': 'child_failure',
                    'depth': depth + 1,
                    'message': 'Child branch failed after this action.',
                })
                attempt_trace['accepted'] = False
                attempt_trace['child_failure'] = failure
                node['attempts'].append(attempt_trace)
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                continue

            failure = {
                'type': 'property_violation',
                'depth': depth,
                'retry_index': retry_index,
                'action': action,
                'violations': property_trace.get('violations', []),
                'changed_properties': property_trace.get('changed_properties', []),
            }
            attempt_trace['accepted'] = False
            attempt_trace['failure_feedback'] = failure
            attempt_trace['verification'] = {
                'passed': False,
                'properties_checked': [
                    r.get('id') for r in property_trace.get('property_results', [])
                ],
                'violations': [v.get('id') for v in property_trace.get('violations', [])],
                'skipped': False,
            }
            node['attempts'].append(attempt_trace)
            self._checkpoint_trace(trace, trace_path)
            failed_attempts.append(failure)

        node['result'] = 'backtracked'
        self._checkpoint_trace(trace, trace_path)
        failure = {
            'type': 'branch_exhausted',
            'depth': depth,
            'node_id': node_id,
            'failed_attempts': failed_attempts,
            'message': 'All retries at this predicted state were exhausted.',
        }
        node['failure'] = failure
        self._checkpoint_trace(trace, trace_path)
        return {
            'success': False,
            'failure': failure,
            'finish_response': 'No feasible property-satisfying plan found.',
            'node_id': node_id,
        }

    def _build_next_action_prompt(
        self,
        *,
        request: str,
        action_help: str,
        current_state: Dict[str, object],
        accepted_trace: List[Dict[str, object]],
        failed_attempts: List[Dict[str, object]],
    ) -> str:
        lines = [
            'Goal:\n%s' % request,
            self._grounding_verdict_text(current_state, request),
            'Current predicted state JSON:\n%s' % self._snapshot_json(current_state),
            'Structured planning state summary:\n%s' % self._planning_state_summary(
                current_state,
                accepted_trace,
                request=request,
            ),
            'Accepted action trace so far:\n%s' % self._json_or_none(accepted_trace),
            'Properties to satisfy:\n%s' % self._property_text,
        ]
        if failed_attempts:
            lines.append(
                'Failed attempts and backtrack feedback:\n%s'
                % self._json_or_none(failed_attempts[-5:])
            )
        else:
            lines.append('Failed attempts and backtrack feedback:\nNone')
        lines.extend([
            'Plan only the single next step for this state.',
            'Treat the action as a dry-run primitive simulator call.',
            'Choose exactly one primitive action object and nothing else.',
            'Use only the exact argument names and types from the allowed primitive actions list.',
            'Ground every argument from the structured planning state summary or current predicted state JSON.',
            'Read the object descriptions and candidate object ids before using the raw world state.',
            'Treat the grounding verdict as a grounding aid, not as an exact-match filter on the user request.',
            'For highlight_object and unhighlight_object, obj_id must be a concrete integer object id taken from object_catalog, never null.',
            'For move_grasper, use numeric x and y coordinates only, copied from a real object position or another explicit numeric state value; do not use invented keys like target or position.',
            'For goals using words like all, each, or every, still plan one concrete object id at a time.',
            'Prefer the shortest progress-making action.',
            'Match the primitive to the goal type: highlight goals should usually use highlight_object or unhighlight_object, not grasper actions.',
            'For pick/place or stacking goals, first ground a concrete source object from source_candidates and a concrete destination object from destination_candidates.',
            'Never combine attributes across stacked or co-located objects. A destination must be one single object id that individually matches the requested description.',
            'Use user words like color, kind, size, height, and width as soft grounding cues against the described objects, not as a rigid exact-match checklist.',
            'Bind all requested attributes to one single object when you choose an object id. Example: "green small block" should not become a green object supported by a small block.',
            'For pick/place or stacking goals, prefer the shortest valid manipulation skeleton: move_grasper(source x,y) -> lower_grasper -> close_grasper -> raise_grasper -> move_grasper(destination x,y) -> lower_grasper -> open_grasper.',
            'Use open_grasper only when the current predicted state already makes the held object supported by the destination after lowering.',
            'Do not propose an action that merely undoes the latest accepted action unless that reversal is necessary.',
            'If you see an alternating loop pattern in the accepted trace or failed attempts, break the loop instead of continuing it.',
            'If the most recent accepted action has the same name and same args as the action you are about to choose, do not repeat it unless a new state change makes it necessary.',
            'If repeated retries at this state keep failing for the same primitive, switch to a different primitive family.',
            'If a previous retry failed because of invalid args, change the args and keep them schema-valid and concretely grounded.',
            'If the goal is already satisfied, return the finish action.',
            'JSON schema: {"response": "...", "action": {"name": "...", "args": {...}}}',
            'Use {"response": "Done.", "action": {"name": "finish", "args": {}}} when no next step is needed.',
            'Example dry-run primitive call:',
            '{"response": "I will raise the grasper.", "action": {"name": "raise_grasper", "args": {}}}',
            'Return strict JSON only.',
        ])
        return '\n\n'.join(lines)

    def _build_state_prediction_prompt(
        self,
        current_state: Dict[str, object],
        action: Dict[str, object],
    ) -> str:
        return '\n\n'.join([
            'Current state JSON:\n%s' % self._snapshot_json(current_state),
            'Planned next action JSON:\n%s' % self._json_or_none(action),
            'Predict the next symbolic simulator state after this one action.',
            "Important: predict the simulator result, not the planner's hoped-for result.",
            'If this action violates a precondition in the current state, predict no state change instead of a successful effect.',
            'Keep object ids stable and preserve fields you are not changing.',
            'Use sparse state writes only; omit unchanged objects.',
            'JSON schema: {"response": "...", "predicted_state": {...}}',
            'Return strict JSON only.',
        ])

    def _simulate_action_transition(self, current_scene, action: Dict[str, object]) -> Dict[str, object]:
        pre_state = copy.deepcopy(current_scene)
        post_state, notes = predict_world_state_after_actions(pre_state, [action])
        return {
            'action_result': notes[-1] if notes else 'OK',
            'pre_state': pre_state,
            'post_state': post_state,
            'pre_scene': None,
            'post_scene': None,
        }

    def _request_state_prediction(self, history: List[Dict[str, str]], current_state: Dict[str, object]):
        attempts = list(history)
        errors = []
        attempt_log = []
        for attempt_index in range(2):
            content = self._chat(attempts, schema=PREDICTED_STATE_SCHEMA).strip()
            try:
                bundle = self._parse_state_prediction(content, current_state)
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
                        'content': (
                            'Your previous reply was invalid: %s\n'
                            'Rewrite it as strict JSON only using this schema:\n'
                            '{"response": "...", "predicted_state": {...}}'
                        ) % exc,
                    },
                ])
                continue
            attempt_log.append({
                'attempt_index': attempt_index,
                'raw_content': content,
                'parsed_prediction': bundle,
            })
            return content, bundle, attempt_log
        raise ValueError('Invalid predicted-state reply after retry: %s' % errors[-1])

    def _parse_state_prediction(self, content: str, current_state: Dict[str, object]) -> Dict[str, object]:
        content = OpenAICompatibleShrdluAgent._extract_json_object(content)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError('Model did not return valid JSON: %s' % content) from exc
        if not isinstance(decision, dict):
            raise ValueError('Model reply must be a JSON object.')
        predicted_state = decision.get('predicted_state')
        if not isinstance(predicted_state, dict):
            raise ValueError('Model reply must include a predicted_state object.')
        return {
            'response': str(decision.get('response', '')),
            'predicted_state': self._merge_predicted_state(current_state, predicted_state),
        }

    @staticmethod
    def _snapshot_json(snapshot: Dict[str, object]) -> str:
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
        if len(set(names[-4:])) == 2 and names[-4] == names[-2] and names[-3] == names[-1] and names[-4] != names[-3]:
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
                'move_grasper': 'Use numeric x and y copied from object_catalog positions or explicit numeric state values.',
                'all_goals': 'For all/each/every goals, choose one concrete object id per step.',
                'pick_place_goals': 'Choose one concrete source object from source_candidates and one concrete destination object from destination_candidates. Use the per-object descriptions to decide which object id best matches the request.',
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
        # Preserve order while removing duplicates.
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
        lines = []
        for item in self._property_verifier.properties:
            lines.append('%s: %s' % (item.get('id'), item.get('natural_language')))
        return '\n'.join(lines)

    @staticmethod
    def _merge_predicted_state(current_state: Dict[str, object], predicted_state: Dict[str, object]) -> Dict[str, object]:
        merged = copy.deepcopy(current_state)
        for key in ('default_grasper', 'grasper_closed', 'grasper_lowered', 'grasped_object'):
            if key in predicted_state:
                merged[key] = predicted_state[key]

        current_objects = {
            obj['obj_id']: copy.deepcopy(obj)
            for obj in current_state.get('objects', [])
            if isinstance(obj, dict) and 'obj_id' in obj
        }
        for item in predicted_state.get('objects', []):
            if not isinstance(item, dict) or 'obj_id' not in item:
                continue
            obj_id = item['obj_id']
            base = current_objects.get(obj_id, {
                'obj_id': obj_id,
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'tags': {},
            })
            merged_obj = copy.deepcopy(base)
            for key in ('kind', 'color', 'graspable', 'can_support', 'resting_on', 'grasped_by'):
                if key in item:
                    merged_obj[key] = item[key]
            if 'position' in item and isinstance(item['position'], dict):
                merged_position = dict(merged_obj.get('position', {}))
                merged_position.update(item['position'])
                merged_obj['position'] = merged_position
            if 'tags' in item and isinstance(item['tags'], dict):
                merged_tags = dict(merged_obj.get('tags', {}))
                merged_tags.update(item['tags'])
                merged_obj['tags'] = merged_tags
            current_objects[obj_id] = merged_obj
        merged['objects'] = [current_objects[obj_id] for obj_id in sorted(current_objects)]
        return merged


class PredictivePreplannedOpenAICompatibleShrdluAgent(
        _PredictivePreplannedShrdluAgentMixin, OpenAICompatibleShrdluAgent):
    """Predictive tree-search preplanned agent over a local OpenAI API."""

    def __init__(self, env: SimulatorAPI, model: str = DEFAULT_OPENAI_MODEL,
                 base_url: str = DEFAULT_OPENAI_BASE_URL,
                 api_key: str = DEFAULT_OPENAI_API_KEY,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = DEFAULT_TRACE_DIR,
                 temperature: float = 0.2,
                 max_tokens: int = 512,
                 client=None,
                 max_branch_retries: int = 3):
        super().__init__(
            env,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            trace_dir=trace_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
        )
        self._init_predictive_preplanned(max_branch_retries=max_branch_retries)
