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

SCHEMA_VERSION = "1.0"


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
    return path
