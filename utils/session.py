"""Shared session/trace serialisation for all llm-agents-fsm agents.

Every agent (git-agent, git-agent-fsm, shrdlu-*) writes
sessions in the same JSON schema so they can be analysed uniformly.

Schema (version "1.0"):
{
  "schema_version": "1.0",
  "timestamp_utc":  "2026-06-26T...",
  "agent":          "git-agent-fsm | git-agent | shrdlu-agent-fsm | shrdlu-reactive | ...",
  "model":          "gpt-4o-mini",
  "domain":         "git | shrdlu",
  "work_dir":       "/path/to/cwd",          # git agents only
  "request":        "user query / goal",
  "status":         "finished | infeasible | error | max_steps",
  "final_message":  "...",
  "properties": [                            # properties active for this session
    {"id": "prop.git.01", "natural_language": "..."}
  ],
  "planning_tree": {                         # see utils/planning_tree.py
    "nodes": [ ... ],
    "feasible":      true,
    "accepted_plan": [{"label": "...", "tool": "...", "args": {}}]
  },
  "execution": {
    "steps": [
      {
        "step_index":          0,
        "action": {"label": "...", "tool": "...", "args": {}},
        "result":              "...",    # stdout/stderr or simulator result
        "property_verification": {}      # per-step verification (shrdlu style)
      }
    ]
  },
  "llm_log": []   # optional: raw LLM call log for debugging
}
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Re-export planning-tree helpers so callers can do:
#   from utils.session import make_session, make_planning_node, make_verification
from utils.planning_tree import (  # noqa: F401
    make_node        as make_planning_node,
    make_verification,
    make_skipped_verification,
    make_tree        as make_planning_tree,
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
) -> Dict[str, Any]:
    """Create a fresh session dict with required top-level fields."""
    return {
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
        "execution": {
            "steps": [],
        },
        "llm_log": [],
    }


def make_execution_step(
    *,
    step_index: int,
    action_label: str,
    tool: str,
    args: Dict,
    result: str,
    property_verification: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create an execution-step dict."""
    step: Dict[str, Any] = {
        "step_index": step_index,
        "action": {
            "label": action_label,
            "tool":  tool,
            "args":  args,
        },
        "result": result,
    }
    if property_verification is not None:
        step["property_verification"] = property_verification
    return step


def save_session(
    session: Dict[str, Any],
    sessions_dir: Path,
    *,
    filename_prefix: str = "session",
) -> Path:
    """Write session to a timestamped JSON file and return the path."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = sessions_dir / f"{filename_prefix}_{ts}.json"
    path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
