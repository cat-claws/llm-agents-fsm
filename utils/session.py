"""Shared session serialisation for all llm-agents-fsm agents.

Every agent (git-agent, git-agent-fsm, shrdlu-*) writes
sessions in the same JSON schema so their planning trees can be analysed
uniformly.  The session is the durable run record; `planning_tree` is the main
planning artifact inside it, even when the tree is just a single root-to-leaf
chain.

Schema (version "1.0"):
{
  "schema_version": "1.0",
  "timestamp_utc":  "2026-06-26T...",
  "agent":          "git-agent-fsm | git-agent | shrdlu-agent-fsm | shrdlu-agent-basic | ...",
  "model":          "gpt-4o-mini",
  "domain":         "git | shrdlu",
  "work_dir":       "/path/to/cwd",
  "request":        "user query / goal",
  "status":         "finished | infeasible | error | max_steps",
  "final_message":  "...",
  "properties": [
    {"id": "prop.git.01", "natural_language": "..."}
  ],
  "planning_config": { ... },
  "planning_tree": {
    "mode":          "...",
    "nodes":         [ ... ],
    "feasible":      true,
    "accepted_plan": [{"label": "...", "tool": "...", "args": {}}],
    "execution_step": {
      "execution_step":   0,
      "execution_result": "..."
    }
  },
  "llm_log": []
}
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.planning_tree import (
    make_node as make_planning_node,
    make_verification,
    make_skipped_verification,
    make_tree as make_planning_tree,
    make_action,
    make_state_path_entry,
    set_node_outcome,
    add_child,
    annotate_node_executed,
    append_node,
    find_node,
    mark_feasible,
    accepted_plan_from_nodes,
    accepted_nodes,
    accepted_nodes_by_depth,
    mark_accepted_branch_backtracked,
    build_tree_summary,
    extract_property_ids_from_violations,
    make_attempt,
    NodeCounter,
)
from utils.tree_report import default_tree_report_path, render_saved_tree_html

SCHEMA_VERSION = "1.0"

_SESSION_KEY_ORDER = [
    "schema_version",
    "timestamp_utc",
    "agent",
    "model",
    "domain",
    "work_dir",
    "request",
    "status",
    "final_message",
    "properties",
    "planning_tree",
    "llm_log",
    "planning_config",
    "_live",
]

_TREE_KEY_ORDER = [
    "mode",
    "nodes",
    "feasible",
    "accepted_plan",
    "max_steps",
    "max_retries",
    "max_branch_retries",
    "planning_granularity",
    "violation_policy",
    "properties",
    "initial_state",
    "initial_world_state",
    "action_help",
    "finish_response",
    "planning_response",
    "failure",
    "tree_summary",
]

_NODE_KEY_ORDER = [
    "node_id",
    "parent_node_id",
    "depth",
    "children",
    "state_path",
    "state_before",
    "action",
    "state_after",
    "verification",
    "attempts",
    "result",
    "outcome",
    "execution_step",
]

_ACTION_KEY_ORDER = ["label", "tool", "args"]
_VERIFICATION_KEY_ORDER = [
    "passed",
    "properties_checked",
    "violations",
    "tla_spec",
    "skipped",
]
_STATE_PATH_ENTRY_KEY_ORDER = ["action", "state_after"]
_EXECUTION_STEP_KEY_ORDER = ["execution_step", "execution_result"]


def _ordered_dict(source: Dict[str, Any], key_order: List[str]) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for key in key_order:
        if key in source:
            ordered[key] = source[key]
    for key, value in source.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _replace_in_order(target: Dict[str, Any], ordered: Dict[str, Any]) -> None:
    target.clear()
    target.update(ordered)


def _normalize_action(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict):
        action = {}
    normalized = {
        "label": action.get("label"),
        "tool": action.get("tool") or "none",
        "args": action.get("args") if isinstance(action.get("args"), dict) else {},
    }
    for key, value in action.items():
        if key not in normalized:
            normalized[key] = value
    return _ordered_dict(normalized, _ACTION_KEY_ORDER)


def _normalize_verification(verification: Any) -> Dict[str, Any]:
    if not isinstance(verification, dict):
        return make_skipped_verification("pending")
    return _ordered_dict(verification, _VERIFICATION_KEY_ORDER)


def _normalize_state_path_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    normalized = dict(entry)
    if "action" in normalized:
        normalized["action"] = _normalize_action(normalized["action"])
    return _ordered_dict(normalized, _STATE_PATH_ENTRY_KEY_ORDER)


def _normalize_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    normalized = {
        "node_id": node.get("node_id"),
        "parent_node_id": node.get("parent_node_id"),
        "depth": node.get("depth", 0),
        "children": node.get("children") if isinstance(node.get("children"), list) else [],
        "state_path": [
            _normalize_state_path_entry(entry)
            for entry in node.get("state_path", [])
        ] if isinstance(node.get("state_path"), list) else [],
        "state_before": node.get("state_before") if isinstance(node.get("state_before"), dict) else {},
        "action": _normalize_action(node.get("action")),
        "state_after": node.get("state_after") if isinstance(node.get("state_after"), dict) else {},
        "verification": _normalize_verification(node.get("verification")),
        "attempts": node.get("attempts") if isinstance(node.get("attempts"), list) else [],
        "result": node.get("result", "searching"),
        "outcome": node.get("outcome"),
        "execution_step": node.get("execution_step"),
    }
    if isinstance(normalized["execution_step"], dict):
        normalized["execution_step"] = _ordered_dict(
            normalized["execution_step"],
            _EXECUTION_STEP_KEY_ORDER,
        )
    for key, value in node.items():
        if key not in normalized:
            normalized[key] = value
    _replace_in_order(node, _ordered_dict(normalized, _NODE_KEY_ORDER))
    return node


def normalize_planning_tree(planning_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a planning tree to the shared serialized envelope in-place."""
    if not isinstance(planning_tree, dict):
        return planning_tree
    if isinstance(planning_tree.get("nodes"), list):
        for index, node in enumerate(planning_tree["nodes"]):
            planning_tree["nodes"][index] = _normalize_node(node)
    if "feasible" not in planning_tree:
        planning_tree["feasible"] = False
    if "accepted_plan" not in planning_tree:
        planning_tree["accepted_plan"] = []
    _replace_in_order(planning_tree, _ordered_dict(planning_tree, _TREE_KEY_ORDER))
    return planning_tree


def normalize_result_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a saved session/result record before writing JSON."""
    planning_tree = record.get("planning_tree")
    if isinstance(planning_tree, dict):
        record["planning_tree"] = normalize_planning_tree(planning_tree)
    _replace_in_order(record, _ordered_dict(record, _SESSION_KEY_ORDER))
    return record


def make_session(
    *,
    agent: str,
    model: str,
    domain: str,
    request: str,
    work_dir: Optional[str] = None,
    properties: Optional[List[Dict]] = None,
    planning_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a fresh session dict with required top-level fields.

    Args:
        agent:           Agent identifier string.
        model:           LLM model name.
        domain:          'git' | 'shrdlu' | …
        request:         User query / goal.
        work_dir:        Working directory (git agents).
        properties:      Active LTL properties [{id, natural_language}, …].
        planning_config: Agent-specific planning parameters stored verbatim.
    """
    session: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent":          agent,
        "model":          model,
        "domain":         domain,
        "work_dir":       work_dir,
        "request":        request,
        "status":         "running",
        "final_message":  None,
        "properties":     [
            {"id": p.get("id", "?"), "natural_language": p.get("natural_language", "")}
            for p in (properties or [])
        ],
        "planning_tree": {
            "nodes":         [],
            "feasible":      False,
            "accepted_plan": [],
        },
        "llm_log": [],
    }
    if planning_config is not None:
        session["planning_config"] = planning_config
    return session


def save_session(
    session: Dict[str, Any],
    sessions_dir: Path,
    *,
    filename_prefix: str = "session",
) -> Path:
    """Write a session record to a timestamped JSON file and return the path."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = sessions_dir / f"{filename_prefix}_{ts}.json"
    path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    render_saved_tree_html(path)
    return path


def start_result_session(
    record: Dict[str, Any],
    result_dir: Optional[Path | str],
) -> Optional[str]:
    """Start a live result JSON without rendering transient tree artifacts."""
    if result_dir is None:
        return None
    result_dir_path = Path(result_dir)
    result_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_path = result_dir_path / ("result_%s.json" % timestamp)
    record["_live"] = True
    normalize_result_record(record)
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(result_path)


def checkpoint_result(
    record: Dict[str, Any],
    result_path: Optional[Path | str],
) -> Optional[str]:
    """Rewrite an in-progress result JSON without rendering transient tree artifacts."""
    if not result_path:
        return None if result_path is None else str(result_path)
    record["_live"] = True
    normalize_result_record(record)
    path = Path(result_path)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(path)


def write_result(
    record: Dict[str, Any],
    result_dir: Optional[Path | str],
    result_path: Optional[Path | str] = None,
) -> Optional[str]:
    """Finalize a result JSON using the same method as SHRDLU result writes."""
    if result_dir is None and not result_path:
        return None
    if result_path is None:
        result_path = start_result_session(record, result_dir)
    if not result_path:
        return None
    record.pop("_live", None)
    normalize_result_record(record)
    path = Path(result_path)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    render_saved_tree_html(path)
    return str(path)


def append_result_notice(message: str, result_path: Optional[Path | str]) -> str:
    """Append JSON and HTML result locations to a user-facing message."""
    if not result_path:
        return message
    notice = message + "\n\nResult saved to %s" % result_path
    html_path = default_tree_report_path(result_path)
    if html_path.exists():
        notice += "\nTree HTML saved to %s" % html_path
    return notice
