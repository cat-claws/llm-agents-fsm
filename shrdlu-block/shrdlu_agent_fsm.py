"""OpenAI-compatible suffix-replanning predictive agents for the SHRDLU blocks environment."""

import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
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
    PLAN_SCHEMA,
)
from shrdlu_agents.simulator_api import ALLOWED_SIMULATOR_ACTION_NAMES, SimulatorAPI
from shrdlu_agents.shrdlu_agent_plan import (
    _PredictivePreplannedShrdluAgentMixin,
)
from shrdlu_agents.property_verifier import TransitionPropertyVerifier
from shrdlu_blocks.simulator.state_pred import predict_world_state_after_actions
from shrdlu_agents.tla_verifier import verify_ap_trace

_AP_CANDIDATES: List[Dict[str, str]] = TransitionPropertyVerifier.from_file().aps
_AP_NAMES: List[str] = [ap['name'] for ap in _AP_CANDIDATES]

__all__ = [
    'SuffixPredictivePreplannedOpenAICompatibleShrdluAgent',
]


SUFFIX_PREDICTIVE_PLAN_SYSTEM_PROMPT = """You are planning a SHRDLU blocks-world task before execution.

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

SUFFIX_PLAN_USER_PROMPT_TEMPLATE = """\
Goal:
{request}

{grounding_verdict}

Current predicted AP truth values (ap_name: true/false):
{current_ap_bools_json}

Structured planning state summary:
{planning_state_summary}

Accepted action trace so far:
{accepted_trace_json}

Properties to satisfy:
{property_text}

Allowed primitive actions:
{action_help}

Failed suffix attempts and backtrack feedback:
{failed_attempts_json}

Banned first actions at this node (do NOT start your plan with any of these — they were already tried here):
{banned_first_actions_json}

Return the complete remaining action sequence from the current state to goal completion.
JSON schema: {{"response": "...", "plan": [{{"name": "...", "args": {{...}}}}], "finish_response": "..."}}
Return strict JSON only."""

SUFFIX_PLAN_REPAIR_PROMPT_TEMPLATE = """\
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


class _SuffixPredictivePreplannedShrdluAgentMixin(_PredictivePreplannedShrdluAgentMixin):
    """Plan a full remaining suffix, verify along it, then replan from the first failure point."""

    def _run_agent_loop(self, request: str) -> str:
        initial_world_state = self._env.snapshot()
        initial_state = self._build_initial_ap_state(initial_world_state)
        trace = {
            'schema_version': SCHEMA_VERSION,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'agent': 'shrdlu-agent-fsm',
            'domain': 'shrdlu',
            'model': self._model,
            'host': self._host,
            'max_steps': self._max_steps,
            'request': request,
            'planning_mode': 'suffix_predictive_preplanned',
            'max_branch_retries': self._max_branch_retries,
            'property_monitoring': self._property_monitoring_metadata(),
            'planning_tree': {
                'mode': 'suffix_predictive_preplanned',
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
        action_help = self._env.action_help()
        trace['planning_tree']['initial_ap_state'] = initial_state
        trace['planning_tree']['initial_world_state'] = initial_world_state
        trace['planning_tree']['action_help'] = action_help
        trace['status'] = 'planning'
        trace_path = self._start_trace_session(trace)

        result = self._search_plan_suffix(
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
            trace_path=trace_path,
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
            trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
            trace_path = self._write_trace(trace, trace_path)
            return finish_response if response_text == finish_response else self._format_reply(
                response_text,
                finish_response,
            )

        executed_ap_trace = [initial_state]
        trace['status'] = 'executing'
        self._checkpoint_trace(trace, trace_path)
        for step_index, action in enumerate(plan):
            step_trace = {
                'step_index': step_index,
                'planned_action': action,
            }
            try:
                result_text = self._env.execute_action(action)
            except Exception as exc:
                result_text = "ERROR: %s" % exc
            post_state = self._env.snapshot()
            ap_state = self._build_initial_ap_state(post_state)
            executed_ap_trace.append(ap_state)
            tla_result = verify_ap_trace(executed_ap_trace, _AP_NAMES)
            step_trace.update({
                'action_result': result_text,
                'ap_state': ap_state,
                'ap_changes': self._diff_ap_states(executed_ap_trace[-2], ap_state),
                'tla_verification': tla_result,
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
                trace['planning_tree']['tree_summary'] = self._build_tree_summary(trace['planning_tree'])
                trace_path = self._write_trace(trace, trace_path)
                return self._append_trace_notice(final_message, trace_path)

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
        trace_path = self._write_trace(trace, trace_path)
        return final_message

    def _search_plan_suffix(
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
        trace_path: Optional[str],
    ) -> Dict[str, object]:
        """Search for a feasible plan suffix from the current state.

        Each call corresponds to exactly one node in the planning tree, which
        represents the choice of *one action* at the current state.  The node
        may try up to ``max_branch_retries`` different actions (children).  For
        each candidate action the node:

          1. Reuses ``hint_plan`` from the parent's verified suffix tail when
             available; otherwise asks the LLM for a full suffix from this state.
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
        node = {
            'node_id': node_id,
            'parent_node_id': parent_node_id,
            'depth': depth,
            'accepted_steps': self._zip_accepted_steps(accepted_trace, preceding_ap_trace),
            'current_ap_state': copy.deepcopy(current_state),
            'attempts': [],
            'children': [],
            'result': 'searching',
        }
        planning_tree['nodes'].append(node)
        self._checkpoint_trace(trace, trace_path)

        failed_attempts = list(inherited_failures)
        # Track first actions already tried at this node to avoid same-sibling repeats.
        banned_first_actions: List[Dict[str, object]] = []
        # The hint from the parent; reused for the first child attempt only.
        current_hint = list(hint_plan) if hint_plan else []

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
                plan_prompt = self._build_suffix_plan_prompt(
                    request=request,
                    action_help=action_help,
                    current_state=current_state,
                    init_world_state=init_world_state,
                    accepted_trace=accepted_trace,
                    failed_attempts=failed_attempts,
                    banned_first_actions=banned_first_actions,
                )
                history = [
                    {'role': 'system', 'content': SUFFIX_PREDICTIVE_PLAN_SYSTEM_PROMPT},
                    {'role': 'user', 'content': plan_prompt},
                ]
                try:
                    content, plan_bundle, attempts = self._request_suffix_plan(history)
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
                    self._checkpoint_trace(trace, trace_path)
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

            # Empty plan — LLM says goal already satisfied at this state.
            if not plan_bundle['plan']:
                attempt_trace['accepted'] = True
                attempt_trace['finish'] = True
                node['attempts'].append(attempt_trace)
                node['result'] = 'finish'
                node['finish_response'] = finish_response
                self._checkpoint_trace(trace, trace_path)
                return {
                    'success': True,
                    'plan': [],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }

            # Take only the first action from the suffix for this node.
            action = plan_bundle['plan'][0]
            tail = plan_bundle['plan'][1:]

            # Verify this single action via AP prediction + TLC.
            step_verification = self._verify_single_step(
                action=action,
                current_state=current_state,
                init_world_state=init_world_state,
                preceding_ap_trace=preceding_ap_trace,
                accepted_trace=accepted_trace,
                is_last_step=(not tail),
            )
            attempt_trace['action'] = action
            attempt_trace['step_verification'] = step_verification

            if not step_verification['passed']:
                failure = step_verification['failure']
                attempt_trace['accepted'] = False
                attempt_trace['failure_feedback'] = failure
                node['attempts'].append(attempt_trace)
                self._checkpoint_trace(trace, trace_path)
                failed_attempts.append(failure)
                banned_first_actions.append(action)
                current_hint = []
                continue

            # This action passes — recurse into the child node for the next step.
            predicted_ap_state = step_verification['predicted_ap_state']
            new_preceding_ap_trace = preceding_ap_trace + [predicted_ap_state]

            if not tail:
                attempt_trace['accepted'] = True
                node['attempts'].append(attempt_trace)
                node['result'] = 'accepted'
                node['accepted_action'] = action
                node['accepted_predicted_ap_state'] = predicted_ap_state
                node['finish_response'] = finish_response
                self._checkpoint_trace(trace, trace_path)
                return {
                    'success': True,
                    'plan': [action],
                    'planning_response': response_text,
                    'finish_response': finish_response,
                    'node_id': node_id,
                }

            child_result = self._search_plan_suffix(
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
                hint_plan=tail,
                trace=trace,
                trace_path=trace_path,
            )
            attempt_trace['child_node_id'] = child_result.get('node_id')
            if child_result.get('node_id') is not None:
                node['children'].append(child_result['node_id'])
            node['attempts'].append(attempt_trace)
            self._checkpoint_trace(trace, trace_path)

            if child_result.get('success'):
                attempt_trace['accepted'] = True
                node['result'] = 'accepted'
                node['accepted_action'] = action
                node['accepted_predicted_ap_state'] = predicted_ap_state
                return {
                    'success': True,
                    'plan': [action] + child_result.get('plan', []),
                    'planning_response': response_text,
                    'finish_response': child_result.get('finish_response', finish_response),
                    'node_id': node_id,
                }

            # Child subtree dead — backtrack and try another suffix from this
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

        node['result'] = 'backtracked'
        exhaustion_failure = {
            'type': 'branch_exhausted',
            'depth': depth,
            'node_id': node_id,
            'failed_attempts': failed_attempts,
            'message': 'All %d action attempts at this node were exhausted.' % self._max_branch_retries,
        }
        node['failure'] = exhaustion_failure
        self._checkpoint_trace(trace, trace_path)
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
        tlc_result = verify_ap_trace(full_ap_trace, _AP_NAMES, is_complete_trace=is_last_step)
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

    def _build_suffix_plan_prompt(
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
        return SUFFIX_PLAN_USER_PROMPT_TEMPLATE.format(
            request=request,
            grounding_verdict=self._grounding_verdict_text(init_world_state, request),
            current_ap_bools_json=self._snapshot_json(current_ap_bools),
            planning_state_summary=self._planning_state_summary(
                init_world_state,
                accepted_trace,
                request=request,
            ),
            accepted_trace_json=self._json_or_none(accepted_trace),
            property_text=self._property_text,
            action_help=action_help,
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
            self._ap_formula(ap['name']) for ap in _AP_CANDIDATES
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
        """Build a compact per-node summary for quick tree inspection.

        Each entry contains only the fields needed to understand tree shape,
        branching, depth, and which properties were violated where.
        The full detail stays in planning_tree['nodes'].
        """
        # TLC replaces dots with underscores: prop.foo_bar → Property_prop_foo_bar.
        # Match the underscored form and restore the first underscore to a dot.
        prop_short = re.compile(r'Property_(prop_[^\s]+?)(?:\s|$|\.)')

        def extract_props(violations: list) -> List[str]:
            props = []
            for v in violations:
                for m in prop_short.finditer(str(v)):
                    # Convert first underscore back to dot: prop_foo → prop.foo
                    raw = m.group(1)
                    restored = raw.replace('_', '.', 1)
                    props.append(restored)
            return sorted(set(props))

        summary = []
        for node in planning_tree.get('nodes', []):
            attempt_summaries = []
            for attempt in node.get('attempts', []):
                fb = attempt.get('failure_feedback') or {}
                ftype = fb.get('type', '')
                violations = fb.get('violations', [])
                props = extract_props(violations)

                # Find the suffix step index where TLA first failed.
                viol_suffix_idx = None
                for step in attempt.get('predicted_rollout', []):
                    tlc = step.get('tla_verification', {}).get('tlc_result', {})
                    if not (tlc.get('success') or tlc.get('skipped')):
                        viol_suffix_idx = step.get('suffix_index')
                        break

                plan_len = len(attempt.get('planner_decision', {}).get('plan', []))
                attempt_summaries.append({
                    'accepted': attempt.get('accepted'),
                    'failure_type': ftype or None,
                    'violated_props': props or None,
                    'violation_at_suffix_step': viol_suffix_idx,
                    'plan_length': plan_len if plan_len else None,
                    'child_node_id': attempt.get('child_node_id'),
                })

            entry = {
                'node_id': node['node_id'],
                'parent_node_id': node.get('parent_node_id'),
                'depth': node.get('depth'),
                'result': node.get('result'),
                'children': node.get('children', []),
                'attempts': attempt_summaries,
            }
            summary.append(entry)
        return summary

    @staticmethod
    def _zip_accepted_steps(
        accepted_trace: List[Dict[str, object]],
        preceding_ap_trace: List[Dict[str, bool]],
    ) -> List[Dict[str, object]]:
        """Pair each accepted action with the AP state it produced.

        preceding_ap_trace[0] is the state before any accepted action.
        preceding_ap_trace[i+1] is the state after accepted_trace[i].
        Returns a list of {action, ap_state_after} dicts, one per accepted action.
        """
        steps = []
        for i, action in enumerate(accepted_trace):
            ap_after = preceding_ap_trace[i + 1] if i + 1 < len(preceding_ap_trace) else None
            steps.append({
                'action': copy.deepcopy(action),
                'ap_state_after': copy.deepcopy(ap_after),
            })
        return steps

    @staticmethod
    def _ap_formula(name: str) -> str:
        """Return a Python-style formula string for each AP name."""
        # object_N_resting_on_M  →  parts: ['object', 'N', 'resting', 'on', 'M']
        if name.startswith('object_') and '_resting_on_' in name:
            tail = name[len('object_'):]          # 'N_resting_on_M'
            obj_id, support = tail.split('_resting_on_')
            return '%s: (object with obj_id==%s).resting_on == %s' % (name, obj_id, support)
        # some_object_resting_on_N
        if name.startswith('some_object_resting_on_'):
            support = name[len('some_object_resting_on_'):]
            return '%s: any object has resting_on == %s' % (name, support)
        # grasper_closed / grasper_lowered
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

        # Map obj_id -> resting_on from initial snapshot.
        resting_on: Dict[int, object] = {}
        for obj in init_world_state.get('objects', []):
            if isinstance(obj, dict) and 'obj_id' in obj:
                resting_on[int(obj['obj_id'])] = obj.get('resting_on')

        # Map obj_id -> position (x, y) from initial snapshot (used to infer
        # what object is below the grasper after move_grasper).
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
        grasper_pos = init_world_state.get('objects')  # not used directly; we track via move_grasper

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
                    # Find the object at the grasper's (x, y) to infer resting_on.
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

        # Emit current inferred state.
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

        # 1. Try JSON parse.
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

        # 2. Regex scan for any APs still missing.
        if len(extracted) < len(_AP_NAMES):
            for name in _AP_NAMES:
                if name in extracted:
                    continue
                pattern = r'"' + re.escape(name) + r'"\s*:\s*(true|false)'
                m = re.search(pattern, content)
                if m:
                    extracted[name] = m.group(1) == 'true'

        # 3. Fill remaining gaps from current_ap_state (assume no change).
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
        This extracts the prop.* id by stripping the 'Property_' prefix.
        """
        if not failure:
            return []
        seen: set = set()

        def _extract(violation_str: str) -> Optional[str]:
            m = re.search(r'Property_(prop\.[^\s.]+)', str(violation_str))
            return m.group(1) if m else None

        def _walk(f):
            if not isinstance(f, dict):
                return
            if f.get('type') == 'tla_property_violation':
                for v in f.get('violations', []):
                    prop_id = _extract(v)
                    if prop_id:
                        seen.add(prop_id)
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
        return self._property_verifier._evaluate_state_aps(world_state)

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

    def _request_suffix_plan(self, history: List[Dict[str, str]]):
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
                        'content': SUFFIX_PLAN_REPAIR_PROMPT_TEMPLATE.format(error=exc),
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


class SuffixPredictivePreplannedOpenAICompatibleShrdluAgent(
        _SuffixPredictivePreplannedShrdluAgentMixin, OpenAICompatibleShrdluAgent):
    """Suffix-replanning predictive preplanned agent over a local OpenAI API."""

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
