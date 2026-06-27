"""OpenAI-compatible natural-language agents for the SHRDLU blocks world."""

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import openai

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.session import (
    SCHEMA_VERSION,
    append_result_notice as _shared_append_result_notice,
    checkpoint_result as _shared_checkpoint_result,
    start_result_session as _shared_start_result_session,
    write_result as _shared_write_result,
)

from shrdlu_agents.property_verifier import PROPERTY_FILE, TransitionPropertyVerifier
from shrdlu_agents.simulator_api import SimulatorAPI

__all__ = [
    'OpenAICompatibleShrdluAgent',
]

DEFAULT_MAX_STEPS = 50
DEFAULT_RESULT_DIR = str(Path(__file__).resolve().parents[2] / 'playground-llm-agents-fsm' / 'results')
DEFAULT_TRACE_DIR = DEFAULT_RESULT_DIR
DEFAULT_OPENAI_BASE_URL = 'http://127.0.0.1:30000/v1/'
DEFAULT_OPENAI_API_KEY = 'EMPTY'
DEFAULT_OPENAI_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'


SYSTEM_PROMPT = """You control a blocks-world simulator through a small validated action API.

Rules:
- Return exactly one action at a time.
- Use only the allowed action names and argument types you are given.
- Base decisions only on the current world state and latest action result.
- If the task is complete, return the finish action instead of another simulator action.
- If the user asks a conversational question, asks for a status summary, or explicitly says not to act, use the finish action.
- Do not repeat an action that already succeeded unless the latest simulator result clearly shows it failed or the world state changed.
- After a successful move, highlight, open, close, lower, or raise action that satisfies the request, return finish on the next step.
- Keep the response short and factual.

Return strict JSON only.

Examples:
{"response": "I will move the grasper over the blue block.", "action": {"name": "move_grasper", "args": {"x": -0.1, "y": 0.4}}}
{"response": "Done.", "action": {"name": "finish", "args": {}}}
"""

DECISION_SCHEMA = {
    'type': 'object',
    'properties': {
        'response': {
            'type': 'string',
        },
        'action': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                },
                'args': {
                    'type': 'object',
                },
            },
            'required': ['name', 'args'],
        },
    },
    'required': ['response', 'action'],
}

ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {
            'type': 'string',
        },
        'args': {
            'type': 'object',
        },
    },
    'required': ['name', 'args'],
}

PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'response': {
            'type': 'string',
        },
        'plan': {
            'type': 'array',
            'items': ACTION_SCHEMA,
        },
        'finish_response': {
            'type': 'string',
        },
    },
    'required': ['response', 'plan', 'finish_response'],
}


class _ShrdluAgentBase:
    """Shared tool-using agent loop for the SHRDLU blocks environment."""

    def __init__(self, env: SimulatorAPI, model: str, host: str,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 result_dir: Optional[str] = DEFAULT_RESULT_DIR,
                 trace_dir: Optional[str] = None):
        if trace_dir is not None:
            result_dir = trace_dir
        self._env = env
        self._model = model
        self._host = host.rstrip('/')
        self._max_steps = max_steps
        self._result_dir = Path(result_dir) if result_dir else None
        self._property_verifier = TransitionPropertyVerifier.from_file()
        self._last_result_path: Optional[str] = None

    @property
    def env(self) -> SimulatorAPI:
        return self._env

    @property
    def last_result_path(self) -> Optional[str]:
        return self._last_result_path

    @property
    def last_trace_path(self) -> Optional[str]:
        """Backward-compatible alias for older terminal helpers."""
        return self._last_result_path

    def handle_user_input(self, text: str) -> str:
        """Handle a natural-language request against the live environment."""
        request = (text or '').strip()
        if not request:
            return 'Please enter a command or instruction.'
        if request.lower() in {'reset', '/reset'}:
            self._env.reset()
            return 'Environment reset.\n\n' + self._env.snapshot_text()
        return self._run_agent_loop(request)

    def _run_agent_loop(self, request: str) -> str:
        trace = {
            'schema_version': SCHEMA_VERSION,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'agent': 'shrdlu-agent-basic',
            'domain': 'shrdlu',
            'model': self._model,
            'host': self._host,
            'max_steps': self._max_steps,
            'request': request,
            'property_monitoring': self._property_monitoring_metadata(),
            'steps': [],
        }
        history: List[Dict[str, str]] = [{
            'role': 'system',
            'content': SYSTEM_PROMPT,
        }]
        action_help = self._env.action_help()
        observation = self._env.snapshot_text()
        last_result = 'No simulator command has been executed yet.'
        previous_property_status = None

        for step_index in range(self._max_steps):
            prompt = self._build_user_prompt(request, action_help, observation, last_result)
            history.append({
                'role': 'user',
                'content': prompt,
            })
            try:
                content, decision, attempts = self._request_decision(history)
            except Exception as exc:
                trace['steps'].append({
                    'step_index': step_index,
                    'prompt': prompt,
                    'error': str(exc),
                })
                trace['status'] = 'error'
                trace['final_message'] = "Agent error: %s" % exc
                result_path = self._write_result(trace)
                return self._append_result_notice(
                    "Agent error: %s" % exc,
                    result_path,
                )
            history.append({'role': 'assistant', 'content': content})
            action = decision.get('action', {})
            response_text = self._normalize_response_text(
                decision.get('response', ''),
                action.get('name') == 'finish',
            )
            step_trace = {
                'step_index': step_index,
                'prompt': prompt,
                'attempts': attempts,
                'decision': decision,
            }
            if action.get('name') == 'finish':
                trace['steps'].append(step_trace)
                trace['status'] = 'finished'
                trace['final_message'] = response_text
                self._write_result(trace)
                return self._format_reply(response_text, None)

            pre_state = self._env.snapshot()
            pre_scene = copy.deepcopy(getattr(self._env, 'scene', None))
            try:
                result = self._env.execute_action(action)
            except Exception as exc:
                result = "ERROR: %s" % exc
            post_state = self._env.snapshot()
            property_trace, previous_property_status = self._monitor_transition_properties(
                pre_state,
                action,
                post_state,
                pre_scene=pre_scene,
                post_scene=getattr(self._env, 'scene', None),
                previous_property_status=previous_property_status,
            )
            executed_action = self._format_action(action)
            observation = self._env.snapshot_text()
            last_result = "Executed %s.\nResult: %s" % (executed_action, result)
            step_trace.update({
                'executed_action': action,
                'action_result': result,
                'property_verification': property_trace,
                'observation_after': observation,
            })
            trace['steps'].append(step_trace)
            if step_index == self._max_steps - 1:
                final_message = self._format_reply(
                    response_text + "\n\nReached max agent steps.",
                    last_result,
                )
                trace['status'] = 'max_steps'
                trace['final_message'] = final_message
                result_path = self._write_result(trace)
                return self._append_result_notice(final_message, result_path)
        trace['status'] = 'stopped'
        trace['final_message'] = 'Agent stopped without producing a result.'
        result_path = self._write_result(trace)
        return self._append_result_notice('Agent stopped without producing a result.', result_path)

    @staticmethod
    def _build_user_prompt(request: str, action_help: str, observation: str,
                           last_result: str) -> str:
        return "\n\n".join([
            "User request:\n%s" % request,
            action_help,
            observation,
            "Latest simulator result:\n%s" % last_result,
            'JSON schema: {"response": "...", "action": {"name": "...", "args": {...}}}',
            'Use {"response": "...", "action": {"name": "finish", "args": {}}} when done.',
            "Return strict JSON only.",
        ])

    @staticmethod
    def _parse_decision(content: str) -> Dict[str, str]:
        content = _ShrdluAgentBase._extract_json_object(content)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model did not return valid JSON: %s" % content) from exc
        if not isinstance(decision, dict):
            raise ValueError("Model reply must be a JSON object.")
        action = _ShrdluAgentBase._normalize_action(decision)
        return {
            'response': str(decision.get('response', '')),
            'action': {
                'name': str(action.get('name', '')).strip(),
                'args': action.get('args', {}) if isinstance(action.get('args', {}), dict) else {},
            },
        }

    def _request_decision(self, history: List[Dict[str, str]]):
        attempts = list(history)
        errors = []
        attempt_log = []
        for attempt_index in range(2):
            content = self._chat(attempts).strip()
            try:
                decision = self._parse_decision(content)
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
                            "Your previous reply was invalid: %s\n"
                            "Rewrite it as strict JSON only using this schema:\n"
                            '{"response": "...", "action": {"name": "...", "args": {...}}}'
                        ) % exc,
                    },
                ])
                continue
            action_name = decision['action']['name']
            if not action_name:
                errors.append('Model reply must include a non-empty action name.')
                attempt_log.append({
                    'attempt_index': attempt_index,
                    'raw_content': content,
                    'error': 'Model reply must include a non-empty action name.',
                    'parsed_decision': decision,
                })
                if attempt_index == 1:
                    break
                attempts.extend([
                    {'role': 'assistant', 'content': content},
                    {
                        'role': 'user',
                        'content': (
                            "Your previous reply used an empty action name.\n"
                            "Return strict JSON only and choose a valid action name or finish."
                        ),
                    },
                ])
                continue
            attempt_log.append({
                'attempt_index': attempt_index,
                'raw_content': content,
                'parsed_decision': decision,
            })
            return content, decision, attempt_log
        raise ValueError("Invalid model reply after retry: %s" % errors[-1])

    @staticmethod
    def _format_reply(response_text: str, command_result: Optional[str]) -> str:
        if not command_result:
            return response_text
        return response_text + "\n\n" + command_result

    @staticmethod
    def _format_action(action: Dict[str, object]) -> str:
        return json.dumps(action, sort_keys=True)

    @staticmethod
    def _extract_json_object(content: str) -> str:
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            return content
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return content
        return content[start:end + 1]

    @staticmethod
    def _normalize_response_text(text: str, is_finish: bool) -> str:
        text = (text or '').strip()
        if text:
            return text
        if is_finish:
            return 'Done.'
        return 'No response provided.'

    @staticmethod
    def _normalize_action(decision: Dict[str, object]) -> Dict[str, object]:
        raw_action = decision.get('action')
        if isinstance(raw_action, dict):
            return raw_action
        if isinstance(raw_action, str):
            return {
                'name': raw_action,
                'args': _ShrdluAgentBase._extract_action_args(decision),
            }
        action_name = decision.get('name') or decision.get('action_name')
        if isinstance(action_name, str):
            return {
                'name': action_name,
                'args': _ShrdluAgentBase._extract_action_args(decision),
            }
        raise ValueError("Model reply must include an action object.")

    @staticmethod
    def _extract_action_args(decision: Dict[str, object]) -> Dict[str, object]:
        for key in ('args', 'arguments', 'parameters'):
            value = decision.get(key)
            if isinstance(value, dict):
                return value
        raw_action = decision.get('action')
        if isinstance(raw_action, dict):
            for key in ('args', 'arguments', 'parameters'):
                value = raw_action.get(key)
                if isinstance(value, dict):
                    return value
        return {}

    def _start_result_session(self, record: Dict[str, object]) -> Optional[str]:
        result_path = _shared_start_result_session(record, self._result_dir)
        if result_path is not None:
            self._last_result_path = result_path
        return result_path

    def _checkpoint_result(self, record: Dict[str, object], result_path: Optional[str]) -> Optional[str]:
        return _shared_checkpoint_result(record, result_path)

    def _write_result(self, record: Dict[str, object], result_path: Optional[str] = None) -> Optional[str]:
        result_path = _shared_write_result(record, self._result_dir, result_path)
        if result_path is not None:
            self._last_result_path = result_path
        return result_path

    @staticmethod
    def _append_result_notice(message: str, result_path: Optional[str]) -> str:
        return _shared_append_result_notice(message, result_path)

    def _start_trace_session(self, trace: Dict[str, object]) -> Optional[str]:
        return self._start_result_session(trace)

    def _checkpoint_trace(self, trace: Dict[str, object], trace_path: Optional[str]) -> Optional[str]:
        return self._checkpoint_result(trace, trace_path)

    def _write_trace(self, trace: Dict[str, object], trace_path: Optional[str] = None) -> Optional[str]:
        return self._write_result(trace, trace_path)

    @staticmethod
    def _append_trace_notice(message: str, trace_path: Optional[str]) -> str:
        return _ShrdluAgentBase._append_result_notice(message, trace_path)

    def _property_monitoring_metadata(self) -> Dict[str, object]:
        return {
            'enabled': True,
            'property_file': str(PROPERTY_FILE),
            'property_count': len(self._property_verifier.properties),
        }

    def _monitor_transition_properties(
        self,
        pre_state: Dict[str, object],
        action: Dict[str, object],
        post_state: Dict[str, object],
        *,
        pre_scene,
        post_scene,
        previous_property_status: Optional[Dict[str, bool]],
    ):
        verification = self._property_verifier.verify_transition(
            pre_state,
            action,
            post_state,
            pre_scene=pre_scene,
            post_scene=post_scene,
        )
        current_property_status = {
            item['id']: bool(item['satisfied'])
            for item in verification['property_results']
            if item.get('id')
        }
        changed_properties = []
        if previous_property_status is not None:
            for property_id, current_value in current_property_status.items():
                previous_value = previous_property_status.get(property_id)
                if previous_value is None or previous_value == current_value:
                    continue
                changed_properties.append({
                    'id': property_id,
                    'before': previous_value,
                    'after': current_value,
                })
        verification['changed_properties'] = changed_properties
        return verification, current_property_status

    def _chat(self, messages: List[Dict[str, str]], schema: Dict[str, object] = DECISION_SCHEMA) -> str:
        del messages, schema
        raise NotImplementedError('Subclasses must provide a chat backend.')


class OpenAICompatibleShrdluAgent(_ShrdluAgentBase):
    """OpenAI-compatible chat-completions agent for the SHRDLU blocks environment."""

    def __init__(self, env: SimulatorAPI, model: str = DEFAULT_OPENAI_MODEL,
                 base_url: str = DEFAULT_OPENAI_BASE_URL,
                 api_key: str = DEFAULT_OPENAI_API_KEY,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 trace_dir: Optional[str] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 512,
                 enable_thinking: bool = True,
                 separate_reasoning: bool = True,
                 client=None,
                 result_dir: Optional[str] = DEFAULT_RESULT_DIR):
        super().__init__(
            env,
            model=model,
            host=base_url,
            max_steps=max_steps,
            result_dir=result_dir,
            trace_dir=trace_dir,
        )
        if client is None:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._client = client
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._enable_thinking = bool(enable_thinking)
        self._separate_reasoning = bool(separate_reasoning)

    def _chat(self, messages: List[Dict[str, str]], schema: Dict[str, object] = DECISION_SCHEMA) -> str:
        del schema
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body={
                    'chat_template_kwargs': {'enable_thinking': self._enable_thinking},
                    'separate_reasoning': self._separate_reasoning,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenAI-compatible chat error at %s: %s" % (self._host, exc)
            ) from exc
        try:
            return response.choices[0].message.content or ''
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenAI-compatible response: %r" % response) from exc
