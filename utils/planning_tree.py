"""Shared planning-tree data model for all llm-agents-fsm agents.

The planning tree records every node explored during Phase 4 (search/planning).
Each node represents one candidate action at one depth in the search, with:
  - the action proposed (label, tool, args)
  - the state before and after (AP → bool map)
  - the verification result (TLC / property-verifier output)
  - all LLM attempts at this node (for debugging backtracking)
  - the final result (accepted | rejected | finish | searching | backtracked)

This module does NOT do I/O itself.  Callers build nodes via the helpers,
attach them to a tree dict, then pass the tree to utils.session.save_session.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── node IDs ─────────────────────────────────────────────────────────────────

class NodeCounter:
    """Simple integer counter for assigning sequential node IDs within a tree."""
    def __init__(self, start: int = 0) -> None:
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v

    @property
    def current(self) -> int:
        return self._n


# ── tree / node constructors ──────────────────────────────────────────────────

def make_tree(
    *,
    mode: str = "search",
    max_steps: Optional[int] = None,
    max_retries: Optional[int] = None,
    properties: Optional[List[Dict]] = None,
    initial_state: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Create a fresh planning-tree container.

    Args:
        mode:          'search' (git-fsm) | 'predictive_preplanned' (shrdlu) | etc.
        max_steps:     hard cap on plan length
        max_retries:   per-node retry budget
        properties:    list of {id, natural_language} dicts active for this session
        initial_state: AP → bool snapshot at the start of planning (s0)
    """
    tree: Dict[str, Any] = {
        "mode":           mode,
        "nodes":          [],
        "feasible":       False,
        "accepted_plan":  [],
    }
    if max_steps is not None:
        tree["max_steps"] = max_steps
    if max_retries is not None:
        tree["max_retries"] = max_retries
    if properties is not None:
        tree["properties"] = [
            {"id": p.get("id", "?"), "natural_language": p.get("natural_language", "")}
            for p in properties
        ]
    if initial_state is not None:
        tree["initial_state"] = initial_state
    return tree


def make_node(
    *,
    node_id: int,
    parent_node_id: Optional[int],
    depth: int,
    action_label: str,
    tool: str,
    args: Dict[str, Any],
    state_before: Dict[str, bool],
    state_after: Dict[str, bool],
    verification: Dict[str, Any],
    attempts: Optional[List[Any]] = None,
    result: str = "searching",
) -> Dict[str, Any]:
    """Create a single planning-tree node in the canonical schema.

    Args:
        node_id:        Unique integer within this tree (from NodeCounter).
        parent_node_id: Parent node's ID, or None for the root.
        depth:          Zero-based depth in the search tree.
        action_label:   Short snake_case string describing the action
                        (e.g. 'push_origin', 'rebase_upstream').
        tool:           Tool name used to execute: 'git_cmd' | 'shell_cmd' |
                        'simulator_action' | 'none'.
        args:           Tool arguments dict.
        state_before:   AP → bool snapshot before this action.
        state_after:    AP → bool snapshot predicted/observed after this action.
        verification:   Output of make_verification() — TLC / verifier result.
        attempts:       List of per-attempt dicts (LLM outputs, errors, etc.).
        result:         'accepted' | 'rejected' | 'finish' | 'searching' | 'backtracked'.
    """
    return {
        "node_id":        node_id,
        "parent_node_id": parent_node_id,
        "depth":          depth,
        "action": {
            "label": action_label,
            "tool":  tool,
            "args":  args,
        },
        "state_before":  state_before,
        "state_after":   state_after,
        "verification":  verification,
        "attempts":      attempts if attempts is not None else [],
        "result":        result,
    }


def make_verification(
    *,
    passed: bool,
    properties_checked: List[str],
    violations: Optional[List[str]] = None,
    tla_spec: Optional[str] = None,
    skipped: bool = False,
) -> Dict[str, Any]:
    """Create a verification sub-dict for a planning node.

    Args:
        passed:               True if all checked properties are satisfied.
        properties_checked:   List of property IDs (or TLA+ prop names) that
                              were checked.
        violations:           Violation descriptions from TLC or the property
                              verifier.  Empty list means no violations.
        tla_spec:             The TLA+ module string (omit to save space).
        skipped:              True if the verifier was not run (e.g. no safety
                              properties remained after filtering liveness).
    """
    v: Dict[str, Any] = {
        "passed":             passed,
        "properties_checked": properties_checked,
        "violations":         violations if violations is not None else [],
        "skipped":            skipped,
    }
    if tla_spec is not None:
        v["tla_spec"] = tla_spec
    return v


def make_skipped_verification(reason: str = "no safety properties") -> Dict[str, Any]:
    """Convenience: a verification that was skipped with a note."""
    return make_verification(
        passed=True,
        properties_checked=[],
        skipped=True,
        violations=[f"[skipped] {reason}"],
    )


# ── tree helpers ──────────────────────────────────────────────────────────────

def append_node(tree: Dict[str, Any], node: Dict[str, Any]) -> None:
    """Append node to tree['nodes'] in-place."""
    tree["nodes"].append(node)


def mark_feasible(
    tree: Dict[str, Any],
    accepted_plan: List[Dict[str, Any]],
) -> None:
    """Mark the tree as having found a feasible plan.

    Args:
        tree:          The planning tree dict (modified in-place).
        accepted_plan: List of action dicts: [{label, tool, args}, ...].
    """
    tree["feasible"]      = True
    tree["accepted_plan"] = accepted_plan


def accepted_plan_from_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract accepted_plan from the accepted nodes in the tree.

    Traverses from the root accepted node to the finish node in depth order.
    Only works when nodes were appended in DFS order and result=='accepted'
    for the winning branch.
    """
    accepted = [n for n in tree["nodes"] if n["result"] in ("accepted", "finish")]
    accepted.sort(key=lambda n: n["depth"])
    return [n["action"] for n in accepted]


# ── attempt helpers ───────────────────────────────────────────────────────────

def make_attempt(
    *,
    retry_index: int,
    prompt: Optional[str] = None,
    llm_response: Optional[str] = None,
    accepted: bool = False,
    error: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Create a single attempt record for node['attempts'].

    Args:
        retry_index:   Which retry this is (0-based).
        prompt:        The prompt sent to the LLM for this attempt.
        llm_response:  The raw LLM response text.
        accepted:      Whether this attempt led to the node being accepted.
        error:         Error string if the attempt raised an exception.
        **extra:       Domain-specific fields (e.g. shadow_result, planner_decision).
    """
    a: Dict[str, Any] = {
        "retry_index": retry_index,
        "accepted":    accepted,
    }
    if prompt is not None:
        a["prompt"] = prompt
    if llm_response is not None:
        a["llm_response"] = llm_response
    if error is not None:
        a["error"] = error
    a.update(extra)
    return a
