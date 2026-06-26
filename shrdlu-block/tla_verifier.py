"""TLA+ trace verification for SHRDLU AP traces — thin wrapper over utils/tla_verifier.

Public API is unchanged; implementation delegates to the shared module so
both git-agent-fsm and the SHRDLU agents use one canonical TLC runner.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure utils/ (two levels up) is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.tla_verifier import (      # noqa: E402
    build_tla_spec   as _build_tla_spec,
    build_tla_cfg    as _build_tla_cfg,
    run_tlc          as _run_tlc,
    verify_ap_trace  as _verify_ap_trace,
    load_properties  as _load_properties,
    has_liveness     as _has_liveness,
)
from shrdlu_agents.property_verifier import ACTIVE_PROPERTY_IDS

__all__ = ['build_tla_spec', 'build_tla_cfg', 'run_tlc', 'verify_ap_trace']

_PROPERTIES_FILE = Path(__file__).resolve().parent / 'resources' / 'SHRDLU_PROPERTIES_AST.json'


def _load_shrdlu_properties(active_ids=ACTIVE_PROPERTY_IDS) -> List[Dict]:
    props = _load_properties(_PROPERTIES_FILE)
    return [p for p in props if p.get('id') in active_ids]


def build_tla_spec(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
) -> str:
    """Return a TLA+ module string (plain-state style) for the given AP trace."""
    if properties is None:
        properties = _load_shrdlu_properties()
    tla_str, _cfg, _pnames = _build_tla_spec(
        ap_trace, ap_names, properties,
        module_name=module_name,
    )
    return tla_str


def build_tla_cfg(
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
) -> str:
    """Return a TLC .cfg file string referencing all active properties."""
    if properties is None:
        properties = _load_shrdlu_properties()
    _tla, cfg, _pnames = _build_tla_spec(
        # single-state dummy trace just to get the cfg — ap_names derived from props
        [{}], [], properties,
        module_name=module_name,
    )
    return cfg


def run_tlc(
    tla_spec: str,
    cfg: str,
    module_name: str = 'ShrdluTrace',
    timeout: int = 60,
) -> Dict:
    """Run TLC on the given spec and cfg strings. Returns a result dict."""
    return _run_tlc(tla_spec, cfg, module_name=module_name, timeout=timeout)


def verify_ap_trace(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
    timeout: int = 60,
    is_complete_trace: bool = True,
) -> Dict:
    """Build and run TLC on an AP trace. Returns combined result dict."""
    if properties is None:
        properties = _load_shrdlu_properties()
    result = _verify_ap_trace(
        ap_trace, ap_names, properties,
        module_name=module_name,
        timeout=timeout,
        is_complete_trace=is_complete_trace,
    )
    # keep legacy key 'tla_cfg' alongside 'tla_spec'
    if 'tla_cfg' not in result:
        result['tla_cfg'] = ''
    return result
