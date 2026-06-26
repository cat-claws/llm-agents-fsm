"""Shared planning-tree data model for all llm-agents-fsm agents.

Every agent (git-agent-fsm, shrdlu-agent-fsm, …) builds the same tree so
sessions can be analysed and compared uniformly.

Node schema (all fields always present):
  node_id         int       unique within this tree
  parent_node_id  int|None  None for the root
  depth           int       0-based depth in the search tree
  children        [int]     node_ids of child nodes (populated as tree grows)
  action          dict      {label, tool, args}  — the proposed/accepted action
                            (label=None, tool='none', args={} until outcome known)
  state_before    dict      AP → bool snapshot before this action
  state_after     dict      AP → bool snapshot predicted/observed after ({}
                            until outcome known)
  state_path      [dict]    [{action, state_after}, …] for every accepted step
                            leading to this node — i.e. the path from the root
  verification    dict      make_verification() result (skipped until known)
  attempts        [dict]    per-attempt records (LLM outputs, errors, etc.)
  result          str       'searching' | 'accepted' | 'rejected' |
                            'finish' | 'backtracked' |
                            'accepted_with_ignored_violations'
  outcome         dict|None extra outcome metadata set when result is finalised:
                            {finish_response?, failure?}
  execution_step  dict|None set after real execution via annotate_node_executed()

Tree schema (planning_tree dict):
  mode            str       e.g. 'search', 'fsm_step_retry', 'fsm_batch_ignore'
  max_steps       int?
  max_retries     int?      per-node action-retry budget (git style)
  max_branch_retries int?   per-node branch-retry budget (shrdlu style)
  planning_granularity str? 'step' | 'batch'
  violation_policy str?     'retry' | 'ignore'
  properties      [dict]?   [{id, natural_language}, …]
  initial_state   dict?     AP → bool at start of planning (s0)
  initial_world_state dict? raw world snapshot at start (shrdlu)
  action_help     str?      action catalogue text (shrdlu)
  nodes           [node]    all nodes in creation order
  feasible        bool
  accepted_plan   [dict]    [{label, tool, args}, …] winning action sequence
  tree_summary    [dict]?   compact per-node summary (shrdlu)
  finish_response str?      LLM finish message once plan is found
  planning_response str?    LLM planning narrative
  failure         dict?     top-level failure info when infeasible

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


# ── tree constructor ──────────────────────────────────────────────────────────

def make_tree(
    *,
    mode: str = "search",
    max_steps: Optional[int] = None,
    max_retries: Optional[int] = None,
    max_branch_retries: Optional[int] = None,
    planning_granularity: Optional[str] = None,
    violation_policy: Optional[str] = None,
    properties: Optional[List[Dict]] = None,
    initial_state: Optional[Dict[str, bool]] = None,
    initial_world_state: Optional[Dict[str, Any]] = None,
    action_help: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a fresh planning-tree container.

    Args:
        mode:                 'search' | 'fsm_step_retry' | 'fsm_batch_ignore' | …
        max_steps:            hard cap on plan length / node budget
        max_retries:          per-node action-retry budget (git style)
        max_branch_retries:   per-node branch-retry budget (shrdlu style)
        planning_granularity: 'step' | 'batch'
        violation_policy:     'retry' | 'ignore'
        properties:           list of {id, natural_language} dicts
        initial_state:        AP → bool snapshot at start of planning (s0)
        initial_world_state:  raw world snapshot at start (shrdlu)
        action_help:          action catalogue text (shrdlu)
    """
    tree: Dict[str, Any] = {
        "mode":          mode,
        "nodes":         [],
        "feasible":      False,
        "accepted_plan": [],
    }
    if max_steps is not None:
        tree["max_steps"] = max_steps
    if max_retries is not None:
        tree["max_retries"] = max_retries
    if max_branch_retries is not None:
        tree["max_branch_retries"] = max_branch_retries
    if planning_granularity is not None:
        tree["planning_granularity"] = planning_granularity
    if violation_policy is not None:
        tree["violation_policy"] = violation_policy
    if properties is not None:
        tree["properties"] = [
            {"id": p.get("id", "?"), "natural_language": p.get("natural_language", "")}
            for p in properties
        ]
    if initial_state is not None:
        tree["initial_state"] = initial_state
    if initial_world_state is not None:
        tree["initial_world_state"] = initial_world_state
    if action_help is not None:
        tree["action_help"] = action_help
    return tree


# ── node constructor ──────────────────────────────────────────────────────────

def make_node(
    *,
    node_id: int,
    parent_node_id: Optional[int],
    depth: int,
    state_before: Dict[str, bool],
    state_path: Optional[List[Dict[str, Any]]] = None,
    action_label: Optional[str] = None,
    tool: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    state_after: Optional[Dict[str, bool]] = None,
    verification: Optional[Dict[str, Any]] = None,
    attempts: Optional[List[Any]] = None,
    result: str = "searching",
) -> Dict[str, Any]:
    """Create a single planning-tree node in the canonical schema.

    action_label / tool / args / state_after / verification may be omitted at
    construction time (shrdlu style: node is created before the attempt loop,
    then finalised via set_node_outcome()).  They default to sentinel values
    that make it obvious the node is not yet resolved.

    Args:
        node_id:        Unique integer within this tree (from NodeCounter).
        parent_node_id: Parent node's ID, or None for the root.
        depth:          Zero-based depth in the search tree.
        state_before:   AP → bool snapshot at this node before any action.
        state_path:     [{action, state_after}, …] — accepted steps from root
                        to this node, so the full path is always recoverable.
        action_label:   Label of the action taken at this node.
        tool:           Tool used: 'git_cmd' | 'shell_cmd' | 'simulator_action' | 'none'.
        args:           Tool arguments dict.
        state_after:    AP → bool snapshot after the action (predicted/observed).
        verification:   make_verification() result for this node.
        attempts:       List of per-attempt dicts.
        result:         'searching' | 'accepted' | 'rejected' | 'finish' |
                        'backtracked' | 'accepted_with_ignored_violations'.
    """
    return {
        "node_id":        node_id,
        "parent_node_id": parent_node_id,
        "depth":          depth,
        "children":       [],
        "state_path":     state_path if state_path is not None else [],
        "state_before":   state_before,
        "action": {
            "label": action_label,
            "tool":  tool or "none",
            "args":  args or {},
        },
        "state_after":    state_after if state_after is not None else {},
        "verification":   verification if verification is not None else make_skipped_verification("pending"),
        "attempts":       attempts if attempts is not None else [],
        "result":         result,
        "outcome":        None,
        "execution_step": None,
    }


def set_node_outcome(
    node: Dict[str, Any],
    *,
    result: str,
    action_label: Optional[str] = None,
    tool: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    state_after: Optional[Dict[str, bool]] = None,
    verification: Optional[Dict[str, Any]] = None,
    finish_response: Optional[str] = None,
    failure: Optional[Dict[str, Any]] = None,
) -> None:
    """Finalise a node once its outcome is known (in-place).

    Used by agents (shrdlu style) that create the node before the attempt loop
    and fill in the accepted action / state / verification afterwards.

    Args:
        node:            The node dict to update.
        result:          Final result string.
        action_label:    Label of the accepted action (if any).
        tool:            Tool used for the accepted action.
        args:            Args for the accepted action.
        state_after:     Predicted/observed AP state after the accepted action.
        verification:    Verification result for the accepted action.
        finish_response: LLM finish message (when result == 'finish').
        failure:         Failure detail dict (when result == 'backtracked').
    """
    node["result"] = result
    if action_label is not None:
        node["action"]["label"] = action_label
    if tool is not None:
        node["action"]["tool"] = tool
    if args is not None:
        node["action"]["args"] = args
    if state_after is not None:
        node["state_after"] = state_after
    if verification is not None:
        node["verification"] = verification
    outcome: Dict[str, Any] = {}
    if finish_response is not None:
        outcome["finish_response"] = finish_response
    if failure is not None:
        outcome["failure"] = failure
    if outcome:
        node["outcome"] = outcome


def add_child(node: Dict[str, Any], child_id: int) -> None:
    """Register a child node ID on a parent node (in-place)."""
    node["children"].append(child_id)


def annotate_node_executed(
    node: Dict[str, Any],
    *,
    execution_step: int,
    execution_result: str,
    **extra: Any,
) -> None:
    """Annotate a planning-tree node with its execution outcome (in-place).

    Args:
        node:             The node dict to annotate.
        execution_step:   Zero-based index of this step in the executed plan.
        execution_result: stdout/stderr or simulator result from actual execution.
        **extra:          Domain extras: ap_state, ap_changes,
                          tla_verification, observation_after (shrdlu).
    """
    node["execution_step"] = {
        "execution_step":  execution_step,
        "execution_result": execution_result,
        **extra,
    }


# ── verification constructors ─────────────────────────────────────────────────

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
        properties_checked:   List of property IDs checked.
        violations:           Violation descriptions from TLC or the verifier.
        tla_spec:             The TLA+ module string (omit to save space).
        skipped:              True if the verifier was not run.
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
        violations=["[skipped] %s" % reason],
    )


# ── tree helpers ──────────────────────────────────────────────────────────────

def append_node(tree: Dict[str, Any], node: Dict[str, Any]) -> None:
    """Append a node to tree['nodes'] in-place."""
    tree["nodes"].append(node)


def mark_feasible(
    tree: Dict[str, Any],
    accepted_plan: List[Dict[str, Any]],
) -> None:
    """Mark the tree as having found a feasible plan (in-place).

    Args:
        tree:          The planning tree dict.
        accepted_plan: [{label, tool, args}, …] winning action sequence.
    """
    tree["feasible"]      = True
    tree["accepted_plan"] = accepted_plan


def accepted_plan_from_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract accepted_plan from accepted nodes in DFS order."""
    accepted = [n for n in tree["nodes"] if n["result"] in ("accepted", "finish")]
    accepted.sort(key=lambda n: n["depth"])
    return [n["action"] for n in accepted]


# ── attempt helper ────────────────────────────────────────────────────────────

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
        prompt:        The prompt sent to the LLM.
        llm_response:  The raw LLM response text.
        accepted:      Whether this attempt led to the node being accepted.
        error:         Error string if the attempt raised an exception.
        **extra:       Domain extras (e.g. planner_decision, step_verification).
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
