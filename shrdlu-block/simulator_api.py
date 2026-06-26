"""Client-side protocol for controlling a standalone SHRDLU simulator."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Protocol
from urllib import error, request

ALLOWED_SIMULATOR_ACTION_NAMES = frozenset({
    'move_grasper',
    'lower_grasper',
    'raise_grasper',
    'close_grasper',
    'open_grasper',
    'highlight_object',
    'unhighlight_object',
})
DEFAULT_SIMULATOR_URL = 'http://127.0.0.1:8000'


class SimulatorAPI(Protocol):
    """Minimal API surface used by agents."""

    def reset(self) -> Dict[str, object]:
        ...

    def execute_action(self, action: Dict[str, object]) -> Optional[str]:
        ...

    def snapshot(self) -> Dict[str, object]:
        ...

    def snapshot_text(self) -> str:
        ...

    def action_help(self) -> str:
        ...

    def event_log(self, limit: int = 50) -> List[Dict[str, object]]:
        ...


class HttpSimulatorClient:
    """Small client for the simulator's exposed HTTP API."""

    def __init__(self, base_url: str = DEFAULT_SIMULATOR_URL, timeout: float = 30.0):
        self._base_url = base_url.rstrip('/')
        self._timeout = float(timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    def reset(self) -> Dict[str, object]:
        payload = self._post('/api/reset', {})
        return self._snapshot_from_payload(payload)

    def execute_action(self, action: Dict[str, object]) -> Optional[str]:
        payload = self._post('/api/action', {'action': action})
        if not payload.get('ok', False):
            raise RuntimeError(str(payload.get('output', 'Simulator action failed.')))
        return str(payload.get('output') or 'OK')

    def snapshot(self) -> Dict[str, object]:
        return self._snapshot_from_payload(self._state_payload())

    def snapshot_text(self) -> str:
        payload = self._state_payload()
        text = payload.get('snapshot_text')
        if isinstance(text, str):
            return text
        return self._snapshot_to_text(self._snapshot_from_payload(payload))

    def action_help(self) -> str:
        payload = self._state_payload()
        text = payload.get('action_help')
        if isinstance(text, str):
            return text
        return self._fallback_action_help()

    def event_log(self, limit: int = 50) -> List[Dict[str, object]]:
        payload = self._state_payload()
        events = payload.get('event_log') or []
        return list(events[-int(limit):])

    def _state_payload(self) -> Dict[str, object]:
        return self._get('/api/state')

    def _get(self, path: str) -> Dict[str, object]:
        try:
            with request.urlopen(self._base_url + path, timeout=self._timeout) as response:
                return self._decode_response(response.read())
        except error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError('Simulator HTTP error %s: %s' % (exc.code, details)) from exc
        except error.URLError as exc:
            raise RuntimeError('Could not reach simulator at %s' % self._base_url) from exc

    def _post(self, path: str, body: Dict[str, object]) -> Dict[str, object]:
        data = json.dumps(body).encode('utf-8')
        req = request.Request(
            self._base_url + path,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as response:
                return self._decode_response(response.read())
        except error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError('Simulator HTTP error %s: %s' % (exc.code, details)) from exc
        except error.URLError as exc:
            raise RuntimeError('Could not reach simulator at %s' % self._base_url) from exc

    @staticmethod
    def _decode_response(data: bytes) -> Dict[str, object]:
        payload = json.loads(data.decode('utf-8'))
        if not isinstance(payload, dict):
            raise RuntimeError('Simulator returned a non-object JSON payload.')
        return payload

    @staticmethod
    def _snapshot_from_payload(payload: Dict[str, object]) -> Dict[str, object]:
        snapshot = payload.get('snapshot')
        if not isinstance(snapshot, dict):
            raise RuntimeError('Simulator response did not include a snapshot object.')
        return snapshot

    @staticmethod
    def _fallback_action_help() -> str:
        lines = [
            'Allowed actions:',
            'Return JSON as {"response": "...", "action": {"name": "...", "args": {...}}}',
            'Use {"response": "...", "action": {"name": "finish", "args": {}}} when done.',
        ]
        for name in sorted(ALLOWED_SIMULATOR_ACTION_NAMES):
            lines.append('  %s args={...}' % name)
        return '\n'.join(lines)

    @staticmethod
    def _snapshot_to_text(state: Dict[str, object]) -> str:
        lines = [
            'World state:',
            'default_grasper=%r' % (state.get('default_grasper'),),
            'grasper_closed=%r' % (state.get('grasper_closed'),),
            'grasper_lowered=%r' % (state.get('grasper_lowered'),),
            'grasped_object=%r' % (state.get('grasped_object'),),
            'objects:',
        ]
        for obj in state.get('objects', []):
            position = obj.get('position', {})
            lines.append(
                '  id={obj_id} kind={kind} color={color} graspable={graspable} support={support} '
                'resting_on={resting_on} grasped_by={grasped_by} pos=({x:.3f}, {y:.3f}, {z:.3f})'.format(
                    obj_id=obj.get('obj_id'),
                    kind=obj.get('kind'),
                    color=obj.get('color'),
                    graspable=obj.get('graspable'),
                    support=obj.get('can_support'),
                    resting_on=obj.get('resting_on'),
                    grasped_by=obj.get('grasped_by'),
                    x=float(position.get('x', 0.0)),
                    y=float(position.get('y', 0.0)),
                    z=float(position.get('z', 0.0)),
                )
            )
        return '\n'.join(lines)
