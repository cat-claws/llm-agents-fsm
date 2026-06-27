#!/usr/bin/env python3
"""Render saved JSON tree/result files into a zoomable Graphviz HTML report."""

from __future__ import annotations

import argparse
import glob
import html
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ID_KEYS = ("node_id", "id", "uid", "key", "name", "label")
PARENT_KEYS = ("parent_node_id", "parent_id", "parentId", "parent", "parentNodeId")
CHILD_KEYS = ("children", "child_nodes", "branches")
TITLE_KEYS = ("request", "prompt", "goal", "query", "instruction", "task")
MESSAGE_KEYS = ("final_message", "message", "error", "reason", "description")
STATUS_KEYS = ("result", "status", "state", "type")
TREE_KEYS = ("planning_tree", "tree", "plan_tree", "root")

POSITIVE = {"accepted", "finish", "finished", "success", "succeeded", "passed", "pass", "ok", "complete", "completed"}
NEGATIVE = {"failed", "failure", "error", "errored", "rejected", "violation", "violated", "infeasible", "invalid"}
WARNING = {"accepted_with_ignored_violations", "backtracked", "planning", "executing", "searching", "pending", "skipped", "max_steps", "timeout"}

NODE_STYLES = {
    "root": ("#2980b9", "#1f5f8b", "white", "filled"),
    "success": ("#ecfdf3", "#2f855a", "#14532d", "rounded,filled"),
    "failed": ("#fef2f2", "#dc2626", "#7f1d1d", "rounded,filled,dashed"),
    "warning": ("#fff7ed", "#d97706", "#7c2d12", "rounded,filled,dashed"),
    "unknown": ("#f8fafc", "#94a3b8", "#334155", "rounded,filled"),
}

BADGE_COLORS = {
    "success": "#2ecc71",
    "failed": "#e74c3c",
    "warning": "#e67e22",
    "unknown": "#64748b",
}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def truncate(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def compact_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def format_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    return "(" + ", ".join(f"{key}={compact_value(val)}" for key, val in args.items()) + ")"


def format_action(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    name = action.get("name") or action.get("label") or action.get("tool")
    args = format_args(action.get("args") or action.get("arguments"))
    if name:
        return f"{name}{args}"
    return truncate(action, 100)


def action_for_node(node: dict[str, Any]) -> str:
    direct = format_action(node.get("action"))
    if direct:
        return direct
    path = as_list(node.get("state_path"))
    if path and isinstance(path[-1], dict):
        last_action = format_action(path[-1].get("action"))
        if last_action:
            return f"path: {last_action}"
    return ""


def summarize_changes(node: dict[str, Any], limit: int = 3) -> str:
    execution_step = node.get("execution_step")
    changes = execution_step.get("ap_changes") if isinstance(execution_step, dict) else None
    parts: list[str] = []
    for change in as_list(changes):
        if not isinstance(change, dict):
            continue
        name = change.get("name")
        if name:
            parts.append(f"{name}: {compact_value(change.get('before'))}->{compact_value(change.get('after'))}")
    if not parts:
        return ""
    shown = parts[:limit]
    if len(parts) > limit:
        shown.append(f"+{len(parts) - limit} more")
    return "; ".join(shown)


def summarize_failure(value: Any, limit: int = 110) -> str:
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("failure"), dict):
        failure = value["failure"]
    elif any(key in value for key in ("type", "message", "error", "reason", "violations")):
        failure = value
    else:
        return ""
    ftype = failure.get("type") or "failure"
    message = failure.get("message") or failure.get("error") or failure.get("reason")
    violations = as_list(failure.get("violations"))
    if violations:
        message = violations[0]
    if message:
        return f"{ftype}: {truncate(message, limit)}"
    return str(ftype)


def verification_line(node: dict[str, Any]) -> str:
    verification = node.get("verification")
    if not isinstance(verification, dict):
        verification = node.get("step_verification")
    if not isinstance(verification, dict):
        return ""
    checked = len(as_list(verification.get("properties_checked")))
    if verification.get("skipped"):
        reasons = [str(item).lower() for item in as_list(verification.get("violations"))]
        if not reasons or any("pending" in reason or "no safety properties" in reason for reason in reasons):
            return ""
        suffix = f", {checked} props" if checked else ""
        return f"verify: skipped{suffix}"
    passed = verification.get("passed")
    if passed is True:
        return ""
    if passed is False:
        failure = summarize_failure(verification.get("failure"))
        violations = as_list(verification.get("violations"))
        if not failure and violations:
            failure = truncate(violations[0], 80)
        suffix = f": {truncate(failure, 80)}" if failure else ""
        return f"verify: failed{suffix}"
    return ""


def normalize_status(raw_status: Any, raw_node: Any = None) -> str:
    text = str(raw_status or "").strip().lower()
    if text in POSITIVE:
        return "success"
    if text in NEGATIVE:
        return "failed"
    if text in WARNING:
        return "warning"
    if text == "root":
        return "root"
    if isinstance(raw_node, dict):
        if raw_node.get("error"):
            return "failed"
        verification = raw_node.get("verification")
        if not isinstance(verification, dict):
            verification = raw_node.get("step_verification")
        if isinstance(verification, dict):
            if verification.get("passed") is False:
                return "failed"
            if verification.get("passed") is True and not verification.get("skipped"):
                return "success"
    return "unknown"


def raw_status_text(node: dict[str, Any]) -> str:
    return str(first_present(node, STATUS_KEYS) or "unknown")


def generic_label(raw: Any, node_id: Any, depth: int | None = None, edge_label: str = "") -> str:
    if not isinstance(raw, dict):
        prefix = edge_label or str(node_id)
        return f"{prefix}: {truncate(compact_value(raw), 120)}"

    lines: list[str] = []
    head = str(first_present(raw, ("label", "title", "name", "id", "node_id")) or edge_label or node_id)
    if depth is not None:
        head = f"{head}  depth {depth}"
    lines.append(truncate(head, 90))

    status = first_present(raw, STATUS_KEYS)
    if status is not None:
        lines.append(f"status: {truncate(status, 70)}")

    action = action_for_node(raw)
    if action:
        lines.append(truncate(action, 90))

    verify = verification_line(raw)
    if verify:
        lines.append(verify)

    changes = summarize_changes(raw)
    if changes:
        lines.append("changes: " + truncate(changes, 95))

    for key in ("type", "value", "message", "error", "reason"):
        if key in raw and raw[key] not in (None, "") and key not in STATUS_KEYS:
            value = raw[key]
            if not isinstance(value, (dict, list)):
                lines.append(f"{key}: {truncate(value, 90)}")
        if len(lines) >= 6:
            break
    return "\n".join(lines)


def planning_node_label(node: dict[str, Any]) -> str:
    node_id = node.get("node_id", node.get("id"))
    depth = node.get("depth")
    result = raw_status_text(node)
    action = action_for_node(node) or "root/state"
    lines = [f"#{node_id}  depth {depth}", truncate(action, 90)]
    if str(result).lower() not in POSITIVE:
        lines.append(f"result: {result}")
    verify = verification_line(node)
    if verify:
        lines.append(verify)
    changes = summarize_changes(node)
    if changes:
        lines.append("changes: " + truncate(changes, 95))
    outcome = node.get("outcome")
    if isinstance(outcome, dict):
        if outcome.get("finish_response"):
            lines.append("finish: " + truncate(outcome.get("finish_response"), 90))
        failure = summarize_failure(outcome)
        if failure:
            lines.append("failure: " + failure)
    return "\n".join(lines)


def attempt_label(attempt: dict[str, Any]) -> str:
    idx = first_present(attempt, ("child_index", "retry_index", "attempt_index", "index"))
    action = format_action(attempt.get("action")) or "action"
    if attempt.get("accepted") is True:
        status = "accepted"
    elif attempt.get("accepted") is False:
        status = "rejected"
    else:
        status = raw_status_text(attempt)
    lines = [f"try {idx if idx is not None else '?'}: {status}", truncate(action, 80)]
    feedback = summarize_failure(attempt.get("failure_feedback"))
    step_verification = attempt.get("step_verification")
    if not feedback and isinstance(step_verification, dict):
        feedback = summarize_failure(step_verification.get("failure"))
    if feedback:
        lines.append(truncate(feedback, 95))
    child_failure = summarize_failure(attempt.get("child_failure"))
    if child_failure:
        lines.append("child: " + truncate(child_failure, 85))
    if attempt.get("error"):
        lines.append("error: " + truncate(attempt.get("error"), 90))
    return "\n".join(lines)


def is_finish_attempt(attempt: dict[str, Any]) -> bool:
    if attempt.get("accepted") is not True:
        return False
    if attempt.get("finish") is True:
        return True
    action = attempt.get("action")
    if isinstance(action, dict) and action.get("name") == "finish":
        return True
    decision = attempt.get("planner_decision")
    if isinstance(decision, dict) and decision.get("plan") == []:
        return True
    return False


def finish_attempt_label(attempt: dict[str, Any]) -> str:
    idx = first_present(attempt, ("child_index", "retry_index", "attempt_index", "index"))
    lines = [f"try {idx if idx is not None else '?'}: finish"]
    decision = attempt.get("planner_decision")
    if isinstance(decision, dict):
        finish_response = decision.get("finish_response")
        response = decision.get("response")
        if finish_response:
            lines.append(truncate(finish_response, 90))
        elif response:
            lines.append(truncate(response, 90))
    return "\n".join(lines)


def planning_initial_label(planning_tree: dict[str, Any]) -> str:
    lines = ["initial state", "depth 0"]
    initial_state = planning_tree.get("initial_state")
    if isinstance(initial_state, dict):
        true_aps = [str(key) for key, value in initial_state.items() if value is True]
        if true_aps:
            lines.append("true: " + truncate(", ".join(true_aps), 90))
        else:
            lines.append(f"APs: {len(initial_state)} tracked")
    return "\n".join(lines)


def dot_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\l")
    return '"' + escaped + "\\l" + '"'


def dot_id(prefix: str, raw: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(raw))
    if not safe:
        safe = "root"
    return f"{prefix}{safe}"


def node_style(status: str, root: bool = False) -> tuple[str, str, str, str]:
    if root:
        return NODE_STYLES["root"]
    return NODE_STYLES.get(status, NODE_STYLES["unknown"])


def edge_style(status: str) -> tuple[str, str]:
    if status == "success":
        return "#2f855a", "solid"
    if status == "failed":
        return "#dc2626", "dotted"
    if status == "warning":
        return "#d97706", "dashed"
    return "#64748b", "solid"


def unique(raw: Any, used: set[str]) -> str:
    base = re.sub(r"\s+", "_", str(raw or "node")).strip("_") or "node"
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def model(schema: str) -> dict[str, Any]:
    return {"schema": schema, "nodes": [], "edges": [], "notes": [], "attempts": 0, "truncated": False}


def compute_depths(nodes: list[dict[str, Any]]) -> None:
    by_id = {node["id"]: node for node in nodes}

    def depth(node: dict[str, Any], seen: set[str]) -> int:
        if isinstance(node.get("depth"), int):
            return node["depth"]
        parent = node.get("parent")
        if parent is None or parent not in by_id or parent in seen:
            node["depth"] = 0
            return 0
        node["depth"] = depth(by_id[parent], seen | {str(parent)}) + 1
        return node["depth"]

    for node in nodes:
        depth(node, {str(node["id"])})


def extract_planning_tree(data: dict[str, Any]) -> dict[str, Any] | None:
    planning_tree = data.get("planning_tree")
    if not isinstance(planning_tree, dict):
        return None
    raw_nodes = [node for node in as_list(planning_tree.get("nodes")) if isinstance(node, dict)]
    if not raw_nodes:
        return None

    result = model("planning_tree.nodes")
    used: set[str] = set()
    id_map: dict[Any, str] = {}
    by_raw_id: dict[Any, dict[str, Any]] = {}
    visual_depth_by_raw_id: dict[Any, int] = {}
    initial_id = unique("__initial_state__", used)
    result["nodes"].append(
        {
            "id": initial_id,
            "raw_id": "__initial_state__",
            "parent": None,
            "depth": 0,
            "label": planning_initial_label(planning_tree),
            "status": "root",
            "root": True,
            "raw": {"initial_state": planning_tree.get("initial_state")},
        }
    )

    for idx, raw in enumerate(raw_nodes):
        raw_id = raw.get("node_id", raw.get("id", idx))
        node_id = unique(raw_id, used)
        id_map[raw_id] = node_id
        by_raw_id[raw_id] = raw

    for idx, raw in enumerate(raw_nodes):
        raw_id = raw.get("node_id", raw.get("id", idx))
        parent_raw = raw.get("parent_node_id", raw.get("parent_id"))
        node_id = id_map[raw_id]
        parent = id_map.get(parent_raw) or initial_id
        status = normalize_status(raw_status_text(raw), raw)
        raw_depth = raw.get("depth")
        visual_depth = raw_depth + 1 if isinstance(raw_depth, int) else None
        if visual_depth is not None:
            visual_depth_by_raw_id[raw_id] = visual_depth
        label_node = dict(raw)
        if visual_depth is not None:
            label_node["depth"] = visual_depth
        result["nodes"].append(
            {
                "id": node_id,
                "raw_id": raw_id,
                "parent": parent,
                "depth": visual_depth,
                "label": planning_node_label(label_node),
                "status": status,
                "root": False,
                "raw": raw,
            }
        )

    attempts_by_parent: dict[Any, list[dict[str, Any]]] = {}
    for raw_id, raw in by_raw_id.items():
        attempts = [attempt for attempt in as_list(raw.get("attempts")) if isinstance(attempt, dict)]
        attempts_by_parent[raw_id] = attempts
        result["attempts"] += len(attempts)

    for raw in raw_nodes:
        raw_id = raw.get("node_id", raw.get("id"))
        parent_raw = raw.get("parent_node_id", raw.get("parent_id"))
        if raw_id not in id_map:
            continue
        if parent_raw not in id_map:
            result["edges"].append(
                {
                    "src": initial_id,
                    "dst": id_map[raw_id],
                    "label": "start",
                    "status": normalize_status(raw_status_text(raw), raw),
                }
            )
            continue
        matching = next(
            (attempt for attempt in attempts_by_parent.get(parent_raw, []) if attempt.get("child_node_id") == raw_id),
            None,
        )
        edge_status = normalize_status(raw_status_text(raw), raw)
        label_parts: list[str] = []
        if isinstance(matching, dict):
            label_parts.append(f"try {matching.get('child_index', '?')}")
            action = format_action(matching.get("action"))
            if action:
                label_parts.append(truncate(action, 50))
            if matching.get("accepted") is False:
                edge_status = "warning"
                label_parts.append("backtracked")
        else:
            label_parts.append("child")
        result["edges"].append(
            {
                "src": id_map[parent_raw],
                "dst": id_map[raw_id],
                "label": "\n".join(label_parts),
                "status": edge_status,
            }
        )

    for raw_id, attempts in attempts_by_parent.items():
        for attempt in attempts:
            if is_finish_attempt(attempt):
                finish_id = unique(f"finish_{raw_id}_{attempt.get('child_index', len(result['nodes']))}", used)
                parent = id_map.get(raw_id)
                result["nodes"].append(
                    {
                        "id": finish_id,
                        "raw_id": finish_id,
                        "parent": parent,
                        "depth": visual_depth_by_raw_id.get(raw_id, 0) + 1,
                        "label": finish_attempt_label(attempt),
                        "status": "success",
                        "root": False,
                        "raw": attempt,
                    }
                )
                if parent:
                    result["edges"].append(
                        {
                            "src": parent,
                            "dst": finish_id,
                            "label": "finish",
                            "status": "success",
                        }
                    )
                continue
            if attempt.get("child_node_id") is not None or attempt.get("accepted") is True:
                continue
            note_id = unique(f"attempt_{raw_id}_{attempt.get('child_index', len(result['notes']))}", used)
            result["notes"].append(
                {
                    "id": note_id,
                    "parent": id_map.get(raw_id),
                    "label": attempt_label(attempt),
                    "status": "failed",
                }
            )

    compute_depths(result["nodes"])
    return result


def has_children(raw: Any) -> bool:
    return isinstance(raw, dict) and any(isinstance(raw.get(key), (list, dict)) for key in CHILD_KEYS)


def extract_node_list(raw_nodes: list[Any], schema: str) -> dict[str, Any] | None:
    nodes = [node for node in raw_nodes if isinstance(node, dict)]
    if not nodes:
        return None

    if any(has_children(node) for node in nodes) and not any(first_present(node, PARENT_KEYS) for node in nodes):
        result = model(schema + ".children")
        used: set[str] = set()
        for idx, node in enumerate(nodes):
            flatten_children(node, result, used, None, f"root_{idx}", f"root {idx}", 0)
        return result

    result = model(schema)
    used = set()
    id_map: dict[int, str] = {}
    raw_to_id: dict[Any, str] = {}
    has_parent_links = any(first_present(node, PARENT_KEYS) is not None for node in nodes)

    if not has_parent_links:
        root_id = unique("root", used)
        result["nodes"].append(
            {
                "id": root_id,
                "raw_id": "root",
                "parent": None,
                "depth": 0,
                "label": f"root\n{len(nodes)} nodes",
                "status": "root",
                "root": True,
                "raw": {},
            }
        )

    for idx, raw in enumerate(nodes):
        raw_id = first_present(raw, ID_KEYS)
        if raw_id is None:
            raw_id = idx
        node_id = unique(raw_id, used)
        id_map[idx] = node_id
        raw_to_id[raw_id] = node_id

    for idx, raw in enumerate(nodes):
        raw_id = first_present(raw, ID_KEYS)
        if raw_id is None:
            raw_id = idx
        parent_raw = first_present(raw, PARENT_KEYS)
        parent = raw_to_id.get(parent_raw)
        if parent is None and not has_parent_links:
            parent = result["nodes"][0]["id"]
        status_text = first_present(raw, STATUS_KEYS)
        status = normalize_status(status_text, raw)
        root = parent is None and has_parent_links
        result["nodes"].append(
            {
                "id": id_map[idx],
                "raw_id": raw_id,
                "parent": parent,
                "depth": raw.get("depth"),
                "label": generic_label(raw, raw_id, raw.get("depth")),
                "status": "root" if root else status,
                "root": root,
                "raw": raw,
            }
        )
        if parent is not None:
            label = first_present(raw, ("edge_label", "relation", "child_index", "index"))
            result["edges"].append({"src": parent, "dst": id_map[idx], "label": str(label or ""), "status": status})

    compute_depths(result["nodes"])
    return result


def flatten_children(
    raw: Any,
    result: dict[str, Any],
    used: set[str],
    parent: str | None,
    path: str,
    edge_label: str,
    depth: int,
) -> str:
    raw_id = first_present(raw, ID_KEYS) if isinstance(raw, dict) else path
    node_id = unique(raw_id or path, used)
    status = normalize_status(first_present(raw, STATUS_KEYS) if isinstance(raw, dict) else None, raw)
    root = parent is None
    result["nodes"].append(
        {
            "id": node_id,
            "raw_id": raw_id,
            "parent": parent,
            "depth": depth,
            "label": generic_label(raw, raw_id or path, depth, edge_label),
            "status": "root" if root else status,
            "root": root,
            "raw": raw,
        }
    )
    if parent is not None:
        result["edges"].append({"src": parent, "dst": node_id, "label": edge_label, "status": status})

    if isinstance(raw, dict):
        for child_key in CHILD_KEYS:
            children = raw.get(child_key)
            if isinstance(children, list):
                for idx, child in enumerate(children):
                    flatten_children(child, result, used, node_id, f"{path}_{child_key}_{idx}", f"{child_key}[{idx}]", depth + 1)
            elif isinstance(children, dict):
                for key, child in children.items():
                    flatten_children(child, result, used, node_id, f"{path}_{child_key}_{key}", str(key), depth + 1)
    return node_id


def extract_nested_tree(raw: Any, schema: str) -> dict[str, Any] | None:
    if not isinstance(raw, (dict, list)):
        return None
    result = model(schema)
    used: set[str] = set()
    if isinstance(raw, list):
        if not raw:
            return None
        root = {"label": schema, "children": raw}
        flatten_children(root, result, used, None, "root", "root", 0)
    else:
        if not has_children(raw):
            return None
        flatten_children(raw, result, used, None, "root", "root", 0)
    return result


def json_shape_label(key: str, value: Any) -> str:
    if isinstance(value, dict):
        return f"{key}\nobject: {len(value)} keys"
    if isinstance(value, list):
        return f"{key}\nlist: {len(value)} items"
    return f"{key}\n{truncate(compact_value(value), 90)}"


def extract_json_shape(data: Any, max_nodes: int) -> dict[str, Any]:
    result = model("json-structure")
    used: set[str] = set()

    def add(value: Any, parent: str | None, key: str, path: str, depth: int) -> str | None:
        if len(result["nodes"]) >= max_nodes:
            result["truncated"] = True
            return None
        node_id = unique(path, used)
        root = parent is None
        result["nodes"].append(
            {
                "id": node_id,
                "raw_id": key,
                "parent": parent,
                "depth": depth,
                "label": json_shape_label(key, value),
                "status": "root" if root else "unknown",
                "root": root,
                "raw": value,
            }
        )
        if parent is not None:
            result["edges"].append({"src": parent, "dst": node_id, "label": key, "status": "unknown"})

        if isinstance(value, dict):
            for child_key, child in value.items():
                add(child, node_id, str(child_key), f"{path}_{child_key}", depth + 1)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                add(child, node_id, f"[{idx}]", f"{path}_{idx}", depth + 1)
        return node_id

    add(data, None, "root", "root", 0)
    return result


def extract_tree(data: Any, max_json_nodes: int) -> dict[str, Any]:
    if isinstance(data, dict):
        planning = extract_planning_tree(data)
        if planning is not None:
            return planning

        top_nodes = data.get("nodes")
        if isinstance(top_nodes, list):
            extracted = extract_node_list(top_nodes, "nodes")
            if extracted is not None:
                return extracted

        for key in TREE_KEYS:
            value = data.get(key)
            if isinstance(value, dict):
                nested_nodes = value.get("nodes")
                if isinstance(nested_nodes, list):
                    extracted = extract_node_list(nested_nodes, f"{key}.nodes")
                    if extracted is not None:
                        return extracted
                extracted = extract_nested_tree(value, key)
                if extracted is not None:
                    return extracted
            elif isinstance(value, list):
                extracted = extract_node_list(value, key)
                if extracted is not None:
                    return extracted

        extracted = extract_nested_tree(data, "children")
        if extracted is not None:
            return extracted

    elif isinstance(data, list):
        extracted = extract_node_list(data, "list")
        if extracted is not None:
            return extracted

    return extract_json_shape(data, max_json_nodes)


def render_svg(dot_source: str) -> str:
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_source,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    svg = proc.stdout
    svg = re.sub(r"<\?xml[^>]*>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*(?:\[[\s\S]*?\]\s*)?>\s*", "", svg)
    svg = re.sub(r'\s+width="[^"]+"', "", svg, count=1)
    svg = re.sub(r'\s+height="[^"]+"', "", svg, count=1)
    return svg


def graph_dot(tree: dict[str, Any], graph_name: str | None = None) -> str:
    lines = [
        "digraph tree_report {",
        "  graph [rankdir=TB, bgcolor=\"transparent\", pad=\"0.18\", nodesep=\"0.45\", ranksep=\"0.62\", splines=ortho];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.09,0.06\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#64748b\", arrowsize=0.7];",
    ]
    if graph_name:
        lines.append(f"  labelloc=\"t\"; label={dot_quote(graph_name)};")

    for node in tree["nodes"]:
        fill, color, font, style = node_style(node.get("status", "unknown"), bool(node.get("root")))
        shape = "ellipse" if node.get("root") else "box"
        lines.append(
            "  {id} [label={label}, shape={shape}, fillcolor=\"{fill}\", color=\"{color}\", fontcolor=\"{font}\", style=\"{style}\", penwidth=1.5];".format(
                id=dot_id("n", node["id"]),
                label=dot_quote(str(node.get("label") or node["id"])),
                shape=shape,
                fill=fill,
                color=color,
                font=font,
                style=style,
            )
        )

    for edge in tree["edges"]:
        color, style = edge_style(edge.get("status", "unknown"))
        lines.append(
            "  {src} -> {dst} [label={label}, color=\"{color}\", fontcolor=\"{color}\", style=\"{style}\"];".format(
                src=dot_id("n", edge["src"]),
                dst=dot_id("n", edge["dst"]),
                label=dot_quote(str(edge.get("label") or "")),
                color=color,
                style=style,
            )
        )

    for note in tree["notes"]:
        if not note.get("parent"):
            continue
        color, style = edge_style(note.get("status", "failed"))
        fill, border, font, node_style_text = node_style(note.get("status", "failed"))
        lines.append(
            "  {id} [shape=note, label={label}, fillcolor=\"{fill}\", color=\"{border}\", fontcolor=\"{font}\", style=\"{style}\"];".format(
                id=dot_id("a", note["id"]),
                label=dot_quote(str(note.get("label") or "attempt")),
                fill=fill,
                border=border,
                font=font,
                style=node_style_text,
            )
        )
        lines.append(
            "  {src} -> {dst} [label={label}, color=\"{color}\", fontcolor=\"{color}\", style=\"{style}\"];".format(
                src=dot_id("n", note["parent"]),
                dst=dot_id("a", note["id"]),
                label=dot_quote("failed try"),
                color=color,
                style=style,
            )
        )

    if not tree["nodes"]:
        lines.append('  empty [label="No tree nodes found", fillcolor="#f8fafc", color="#94a3b8"];')
    lines.append("}")
    return "\n".join(lines)


def status_counts(tree: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in tree["nodes"]:
        if not node.get("root"):
            counts[node.get("status", "unknown")] += 1
    return counts


def collect_stats(data: Any, tree: dict[str, Any]) -> dict[str, Any]:
    depths = [node.get("depth") for node in tree["nodes"] if isinstance(node.get("depth"), int)]
    mode = None
    if isinstance(data, dict):
        mode = data.get("planning_mode") or data.get("mode")
        planning_tree = data.get("planning_tree")
        if mode is None and isinstance(planning_tree, dict):
            mode = planning_tree.get("mode")
    return {
        "schema": tree["schema"],
        "nodes": len(tree["nodes"]),
        "edges": len(tree["edges"]),
        "max_depth": max(depths) if depths else 0,
        "attempts": tree.get("attempts", 0),
        "status_counts": status_counts(tree),
        "mode": mode,
        "properties": len(as_list(data.get("properties"))) if isinstance(data, dict) else 0,
        "truncated": bool(tree.get("truncated")),
    }


def html_table(rows: list[tuple[str, Any]]) -> str:
    cells = []
    for key, value in rows:
        if value in (None, "", []):
            continue
        cells.append(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html.escape(str(key)),
                html.escape(truncate(value, 500)),
            )
        )
    if not cells:
        return ""
    return "<table class=\"meta\"><tbody>{}</tbody></table>".format("\n".join(cells))


def stats_strip(stats: dict[str, Any], counts: str) -> str:
    items = [
        ("schema", stats.get("schema")),
        ("nodes", stats.get("nodes")),
        ("edges", stats.get("edges")),
        ("depth", stats.get("max_depth")),
        ("attempts", stats.get("attempts") or None),
        ("results", counts if counts != "none" else None),
    ]
    pills = [
        '<span class="stat-pill"><b>{}</b> {}</span>'.format(
            html.escape(str(label)),
            html.escape(str(value)),
        )
        for label, value in items
        if value not in (None, "", [])
    ]
    if not pills:
        return ""
    return '<div class="summary-strip">{}</div>'.format("".join(pills))


def properties_details(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    props = [prop for prop in as_list(data.get("properties")) if isinstance(prop, dict)]
    if not props:
        return ""
    items = []
    for prop in props:
        label = prop.get("id") or prop.get("name") or "property"
        text = prop.get("natural_language") or prop.get("text") or prop.get("description") or ""
        items.append(f"<li><code>{html.escape(str(label))}</code>: {html.escape(str(text))}</li>")
    return "<details><summary>Properties</summary><ul>{}</ul></details>".format("\n".join(items))


def final_message_details(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    message = first_present(data, MESSAGE_KEYS)
    if not message:
        return ""
    return "<details><summary>Final Message</summary><p>{}</p></details>".format(
        html.escape(truncate(message, 2000))
    )


def accepted_path_summary(tree: dict[str, Any]) -> str:
    accepted = [node for node in tree["nodes"] if node.get("status") == "success"]
    if not accepted:
        return ""
    accepted.sort(key=lambda node: (node.get("depth") if isinstance(node.get("depth"), int) else 9999, str(node["id"])))
    rows = []
    for node in accepted:
        raw = node.get("raw") if isinstance(node.get("raw"), dict) else {}
        action = action_for_node(raw) if isinstance(raw, dict) else ""
        label = action or str(node.get("label") or node["id"]).splitlines()[0]
        changes = summarize_changes(raw) if isinstance(raw, dict) else ""
        suffix = f" <span>{html.escape(changes)}</span>" if changes else ""
        rows.append(
            "<li><code>{}</code> {}</li>".format(
                html.escape(str(node.get("raw_id", node["id"]))),
                html.escape(truncate(label, 160)) + suffix,
            )
        )
    return "<details><summary>Accepted Path</summary><ol>{}</ol></details>".format("\n".join(rows))


def display_title(data: Any, path: Path) -> str:
    if isinstance(data, dict):
        title = first_present(data, TITLE_KEYS)
        if title:
            return truncate(title, 160)
    return path.name


def display_timestamp(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    value = data.get("timestamp_utc") or data.get("timestamp") or data.get("created_at") or data.get("date")
    if value is None:
        return ""
    return str(value)[:19].replace("T", " ")


def file_status(data: Any, tree: dict[str, Any]) -> tuple[str, str]:
    status = None
    if isinstance(data, dict):
        status = data.get("status") or data.get("result")
    kind = normalize_status(status)
    if kind == "unknown":
        counts = status_counts(tree)
        if counts.get("failed"):
            kind = "failed"
        elif counts.get("warning"):
            kind = "warning"
        elif counts.get("success"):
            kind = "success"
    return str(status or kind), kind


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def anchor_for(path: Path, idx: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)
    return f"q{idx:03d}_{stem}"


def section_for_file(path: Path, display_root: Path, idx: int, max_json_nodes: int) -> tuple[str, str, str]:
    display_path = relative_to(path, display_root)
    anchor = anchor_for(path, idx)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report should include broken files.
        safe_path = html.escape(display_path)
        section = (
            f"<section class=\"tree tree-card\" id=\"{anchor}\">"
            f"<div class=\"tree-header\"><span class=\"qnum\">#{idx:03d}</span>"
            f"<span class=\"badge\" style=\"background:{BADGE_COLORS['failed']}\">read error</span>"
            f"<span class=\"qtitle\">{safe_path}</span></div>"
            f"<p class=\"error\">Could not read JSON: {html.escape(str(exc))}</p></section>"
        )
        return section, "read error", "failed"

    tree = extract_tree(data, max_json_nodes)
    stats = collect_stats(data, tree)
    counts = ", ".join(f"{key}: {value}" for key, value in sorted(stats["status_counts"].items())) or "none"
    status_text, status_kind = file_status(data, tree)
    title = display_title(data, path)
    dot_source = graph_dot(tree)
    try:
        graph_html = f"<div class=\"svg-wrap\">{render_svg(dot_source)}</div>"
    except Exception as exc:  # noqa: BLE001 - keep report useful if one graph fails.
        graph_html = (
            "<div class=\"svg-wrap\"><p class=\"error\">Graphviz failed: {}</p><pre>{}</pre></div>".format(
                html.escape(str(exc)),
                html.escape(dot_source),
            )
        )

    rows = [
        ("agent", data.get("agent") if isinstance(data, dict) else None),
        ("model", data.get("model") if isinstance(data, dict) else None),
        ("domain", data.get("domain") if isinstance(data, dict) else None),
        ("mode", stats["mode"]),
        ("properties", stats["properties"] if stats["properties"] else None),
        ("truncated", "yes; increase --max-json-nodes" if stats["truncated"] else None),
    ]

    badge_color = BADGE_COLORS[status_kind]
    meta = display_timestamp(data)
    safe_meta = html.escape(f"{meta} - {display_path}" if meta else display_path)
    safe_title = html.escape(title)
    safe_status = html.escape(status_text)
    details = final_message_details(data) + accepted_path_summary(tree) + properties_details(data)
    section = f"""
  <section class="tree tree-card" id="{anchor}">
    <div class="tree-header">
      <span class="qnum">#{idx:03d}</span>
      <span class="badge" style="background:{badge_color}">{safe_status}</span>
      <span class="qtitle">{safe_title}
        <span class="meta-inline">{safe_meta}</span>
      </span>
    </div>
    {stats_strip(stats, counts)}
    <div class="meta-panel">{html_table(rows)}</div>
    {graph_html}
    <div class="details-panel">{details}</div>
  </section>"""
    return section, status_text, status_kind


def _json_files(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.is_file() and path.suffix.lower() == ".json"]


def _latest(base_dir: Path, count: int, recursive: bool) -> list[Path]:
    if not base_dir.exists():
        return []
    pattern = "**/*.json" if recursive else "*.json"
    paths = _json_files(base_dir.glob(pattern))
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:count]


def _selector_path(selector: str, base_dir: Path) -> Path:
    path = Path(selector).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    base_path = (base_dir / path).resolve()
    return cwd_path if cwd_path.exists() else base_path


def resolve_inputs(selectors: list[str], base_dir: Path, recursive: bool) -> list[Path]:
    if not selectors:
        return _latest(base_dir, 10, recursive)

    if len(selectors) == 1:
        token = selectors[0].strip()
        if token == "latest":
            return _latest(base_dir, 10, recursive)
        if token.isdigit():
            return _latest(base_dir, int(token), recursive)

    if len(selectors) == 2 and selectors[0].strip() == "latest" and selectors[1].isdigit():
        return _latest(base_dir, int(selectors[1]), recursive)

    resolved: list[Path] = []
    for selector in selectors:
        path = _selector_path(selector, base_dir)
        if path.is_file():
            resolved.extend(_json_files([path]))
        elif path.is_dir():
            pattern = "**/*.json" if recursive else "*.json"
            resolved.extend(_json_files(path.glob(pattern)))
        else:
            patterns = [str(path)]
            original = Path(selector).expanduser()
            if not original.is_absolute():
                patterns.append(str(base_dir / selector))
            for pattern in patterns:
                resolved.extend(_json_files(Path(match) for match in glob.glob(pattern, recursive=recursive)))

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in resolved:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        unique_paths.append(real)
    unique_paths.sort(key=lambda path: str(path))
    return unique_paths


def default_output(paths: list[Path], selectors: list[str], base_dir: Path) -> Path:
    if len(paths) == 1:
        return paths[0].with_name(paths[0].stem + "_tree.html")
    if len(selectors) == 1:
        selected = _selector_path(selectors[0], base_dir)
        if selected.is_dir():
            return selected / "planning_trees.html"
    return base_dir / "planning_trees.html"


def default_tree_report_path(json_path: Path | str) -> Path:
    path = Path(json_path)
    return path.with_name(path.stem + "_tree.html")


def render_saved_tree_html(
    json_path: Path | str,
    *,
    output: Path | str | None = None,
    max_json_nodes: int = 160,
    require_tree: bool = True,
    strict: bool = False,
) -> Path | None:
    """Render one saved JSON result/tree file to an adjacent HTML report.

    Returns ``None`` when ``require_tree`` is true and the file only matches the
    generic JSON-structure fallback, or when rendering fails with ``strict`` off.
    """
    path = Path(json_path).expanduser().resolve()
    out = Path(output).expanduser() if output is not None else default_tree_report_path(path)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tree = extract_tree(data, max_json_nodes)
        if require_tree and tree["schema"] == "json-structure":
            return None
        if not tree["nodes"]:
            return None
        report, _statuses = build_report([path], path.parent, max_json_nodes)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        return out
    except Exception:
        if strict:
            raise
        return None


def common_root(paths: list[Path], base_dir: Path) -> Path:
    if not paths:
        return base_dir
    parents = [path.parent for path in paths]
    try:
        common = Path(*Path(*parents[:1]).parts)
    except TypeError:
        common = parents[0]
    try:
        import os

        return Path(os.path.commonpath([str(parent) for parent in parents]))
    except Exception:
        return common


def filter_paths(paths: list[Path], require_tree: bool, require_nodes: bool, max_json_nodes: int) -> list[Path]:
    if not require_tree and not require_nodes:
        return paths
    kept: list[Path] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tree = extract_tree(data, max_json_nodes)
        except Exception:
            if not require_tree and not require_nodes:
                kept.append(path)
            continue
        if require_tree and tree["schema"] == "json-structure":
            continue
        if require_nodes and not tree["nodes"]:
            continue
        kept.append(path)
    return kept


def build_report(paths: list[Path], base_dir: Path, max_json_nodes: int) -> tuple[str, Counter[str]]:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    display_root = common_root(paths, base_dir)
    sections: list[str] = []
    index_links: list[str] = []
    statuses: Counter[str] = Counter()

    for idx, path in enumerate(paths, 1):
        section, status_text, status_kind = section_for_file(path, display_root, idx, max_json_nodes)
        sections.append(section)
        statuses[status_text] += 1
        short = html.escape(truncate(path.name, 48))
        index_links.append(
            f'<a href="#{anchor_for(path, idx)}" class="idx-link" style="background:{BADGE_COLORS[status_kind]}">#{idx:03d} {short}</a>'
        )

    empty = ""
    if not sections:
        empty = '<section class="tree tree-card"><p class="error">No JSON tree files matched.</p></section>'

    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tree Report</title>
<style>
  body { font-family: Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; color: #2c3e50; }
  h1 { text-align: center; padding: 24px 0 8px; margin: 0; letter-spacing: 0; }
  .subtle { text-align: center; margin: 0 12px 10px; color: #5c667a; font-size: 13px; }
  .legend { display: flex; gap: 18px; justify-content: center; padding: 10px; font-size: 13px; color: #555; flex-wrap: wrap; }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .dot { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
  .index { display: flex; flex-wrap: wrap; gap: 6px; max-width: 1200px; margin: 0 auto 24px; padding: 12px 16px; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  .idx-link { padding: 3px 10px; border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: bold; color: white; }
  .tree-card { max-width: 98vw; margin: 0 auto 24px; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12); overflow: hidden; }
  .tree-header { display: flex; align-items: center; gap: 10px; padding: 10px 16px; background: #2c3e50; color: white; }
  .qnum { font-size: 15px; font-weight: bold; min-width: 44px; }
  .badge { font-size: 12px; font-weight: bold; padding: 2px 10px; border-radius: 12px; color: white; white-space: nowrap; }
  .qtitle { font-size: 13px; opacity: .95; min-width: 0; overflow-wrap: anywhere; }
  .meta-inline { opacity: .65; font-size: 11px; margin-left: 8px; }
  .summary-strip { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 16px 0; }
  .stat-pill { display: inline-flex; gap: 5px; align-items: baseline; padding: 3px 8px; border: 1px solid #e2e8f0; border-radius: 999px; background: #f8fafc; color: #334155; font-size: 12px; }
  .stat-pill b { color: #64748b; font-weight: 650; }
  .meta-panel { padding: 12px 16px 0; }
  table.meta { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
  .meta th, .meta td { border-bottom: 1px solid #eef2f7; padding: 5px 8px; vertical-align: top; overflow-wrap: anywhere; }
  .meta th { width: 140px; color: #5c667a; text-align: left; font-weight: 600; }
  .svg-wrap { padding: 12px; background: #f8f9fa; overflow: hidden; cursor: grab; }
  .svg-wrap.dragging { cursor: grabbing; }
  .svg-wrap svg { display: block; width: 100%; height: 72vh; }
  .details-panel { padding: 0 16px 14px; }
  details { margin-top: 12px; padding: 10px 12px; background: #fbfcfe; border: 1px solid #eef2f7; border-radius: 8px; }
  summary { cursor: pointer; font-weight: 650; }
  li { margin: 4px 0; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
  .details-panel span { color: #5c667a; }
  .error { color: #991b1b; background: #fef2f2; border: 1px solid #fecaca; padding: 10px; border-radius: 8px; margin: 12px 16px; }
  pre { overflow-x: auto; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; }
  @media (max-width: 720px) {
    .tree-header { align-items: flex-start; }
    .meta th { width: 104px; }
    .meta-inline { display: block; margin-left: 0; margin-top: 2px; }
  }
</style>
</head>
<body>
<h1>Tree Report</h1>
<p class="subtle">Generated from <code>""" + html.escape(str(display_root)) + """</code> on """ + generated + """ with Graphviz <code>dot</code>.</p>
<div class="legend">
  <span><span class="dot" style="background:#2980b9"></span>root</span>
  <span><span class="dot" style="background:#2ecc71"></span>passed / accepted</span>
  <span><span class="dot" style="background:#e74c3c"></span>failed / rejected</span>
  <span><span class="dot" style="background:#e67e22"></span>backtracked / pending</span>
  <span>scroll to zoom | drag to pan | double-click to reset</span>
</div>
<nav class="index">
""" + "\n".join(index_links) + """
</nav>
""" + "".join(sections) + empty + """
<script>
document.querySelectorAll('.svg-wrap').forEach(function(wrap) {
  var svg = wrap.querySelector('svg');
  if (!svg) return;
  var vb = svg.viewBox.baseVal;
  if (!vb || vb.width === 0) {
    var w = parseFloat(svg.getAttribute('width')) || 800;
    var h = parseFloat(svg.getAttribute('height')) || 600;
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    vb = svg.viewBox.baseVal;
  }
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  var scale = 1, panX = vb.x, panY = vb.y, origW = vb.width, origH = vb.height;
  function apply() { svg.setAttribute('viewBox', panX + ' ' + panY + ' ' + (origW / scale) + ' ' + (origH / scale)); }
  wrap.addEventListener('wheel', function(e) {
    e.preventDefault();
    var r = svg.getBoundingClientRect();
    var cx = (e.clientX - r.left) / r.width, cy = (e.clientY - r.top) / r.height;
    var f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    var ns = Math.max(0.05, Math.min(80, scale * f));
    var cw = origW / scale, ch = origH / scale;
    panX = (panX + cx * cw) - cx * (origW / ns);
    panY = (panY + cy * ch) - cy * (origH / ns);
    scale = ns;
    apply();
  }, {passive: false});
  var drag = null;
  wrap.addEventListener('mousedown', function(e) { drag = {x: e.clientX, y: e.clientY, px: panX, py: panY}; wrap.classList.add('dragging'); });
  window.addEventListener('mousemove', function(e) {
    if (!drag) return;
    var r = svg.getBoundingClientRect();
    panX = drag.px - (e.clientX - drag.x) * (origW / scale) / r.width;
    panY = drag.py - (e.clientY - drag.y) * (origH / scale) / r.height;
    apply();
  });
  window.addEventListener('mouseup', function() { drag = null; wrap.classList.remove('dragging'); });
  wrap.addEventListener('dblclick', function() { scale = 1; panX = vb.x; panY = vb.y; apply(); });
});
</script>
</body>
</html>
"""
    return html_doc, statuses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selectors", nargs="*", help="file, directory, glob, N, or latest N; empty uses latest 10 JSON files")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base directory for latest/default and relative selectors")
    parser.add_argument("--trace-dir", type=Path, default=None, help="compatibility alias for --base-dir")
    parser.add_argument("--repo", type=Path, default=None, help="compatibility: use REPO/agent_traces as the base directory")
    parser.add_argument("-o", "--out", "--output", dest="out", type=Path, default=None, help="output HTML path")
    parser.add_argument("--recursive", action="store_true", help="read JSON files recursively from directory selectors")
    parser.add_argument(
        "--require-tree",
        "--require-planning-tree",
        dest="require_tree",
        action="store_true",
        help="skip files that only render through the generic JSON-structure fallback",
    )
    parser.add_argument("--require-nodes", action="store_true", help="skip files whose detected tree has no nodes")
    parser.add_argument("--max-json-nodes", type=int, default=160, help="cap for generic JSON-structure fallback nodes")
    args = parser.parse_args()

    if args.trace_dir:
        base_dir = args.trace_dir.expanduser().resolve()
    elif args.repo:
        base_dir = (args.repo.expanduser().resolve() / "agent_traces")
    else:
        base_dir = args.base_dir.expanduser().resolve()

    paths = resolve_inputs(args.selectors, base_dir, args.recursive)
    paths = filter_paths(paths, args.require_tree, args.require_nodes, args.max_json_nodes)
    if not paths:
        print("No JSON tree files matched.", file=sys.stderr)
        return 2

    output = args.out.expanduser() if args.out else default_output(paths, args.selectors, base_dir)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report, statuses = build_report(paths, base_dir, args.max_json_nodes)
    output.write_text(report, encoding="utf-8")

    status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())) or "none"
    print(f"Saved: {output}")
    print(f"Trees plotted: {len(paths)}")
    print(f"Status breakdown: {status_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
