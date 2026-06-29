"""Shared agent planning loop for property-checked FSM agents."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from utils.planning_modes import (
    NONBLOCKING_VIOLATION_POLICIES,
    PLANNING_BATCH,
    PLANNING_STEP,
    VIOLATION_IGNORE,
    VIOLATION_RETRY,
    planning_mode_config,
    property_guidance_text,
    property_policy_text,
)
from utils.property_catalog import observe_ap_values
from utils.session import (
    accepted_plan_from_nodes,
    accepted_nodes,
    annotate_node_executed,
    append_node,
    append_result_notice,
    build_tree_summary,
    checkpoint_result,
    make_action,
    make_attempt,
    make_planning_node,
    make_planning_tree,
    make_session,
    make_state_path_entry,
    make_verification,
    mark_feasible,
    start_result_session,
    set_node_outcome,
    write_result,
)
from utils.tla_verifier import verify_fsm_trace


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs shared by FSM-style planning agents."""

    planning_granularity: str = PLANNING_BATCH
    violation_policy: str = VIOLATION_RETRY
    max_plan_steps: int = 10
    max_retries: int = 3


@dataclass(frozen=True)
class AgentFlowSpec:
    """Domain hooks and prompts for the shared FSM agent flow."""

    agent: str
    domain: str
    work_dir: str
    properties: list[dict]
    aps: list[str]
    result_dir: Path | None
    verification_module_name: str
    verification_timeout: int
    observe_initial_state: Callable[[str], dict[str, bool]]
    execute_step: Callable[[dict], str]
    summarize_result: Callable[[str, list[dict], list[str], str], str]
    explain_blocked: Callable[[str, dict[str, bool], list[str], str], str]
    llm_call: Callable[..., tuple[Any, Any]]
    client: Any
    tool_arguments: Callable[[Any, str], Any]
    max_planning_tokens: int
    propose_step_prompt: str
    propose_batch_prompt: str
    predict_ap_prompt: str
    action_proposal_tool: list[dict[str, Any]]
    action_proposal_tool_name: str
    plan_proposal_tool: list[dict[str, Any]]
    plan_proposal_tool_name: str
    ap_prediction_tool: list[dict[str, Any]]
    ap_prediction_tool_name: str
    ap_spec_by_name: dict[str, dict[str, Any]]
    ap_catalog_metadata: dict[str, Any]
    ap_evidence_field: str
    action_prediction_notes: Callable[[str, dict], str]
    observe_execution_ap: Callable[[str], Callable[[str], bool]]
    already_satisfied_response: str = (
        "The request appears already satisfied; no action was executed."
    )


def default_config_from_env(
    *,
    env_var: str,
    retry_default: int,
    max_plan_steps: int = 10,
) -> AgentConfig:
    mode_config = planning_mode_config(
        os.environ.get(env_var),
        retry_default=retry_default,
        invalid="raise",
    )
    return AgentConfig(
        planning_granularity=str(mode_config["planning_granularity"]),
        violation_policy=str(mode_config["violation_policy"]),
        max_plan_steps=max_plan_steps,
        max_retries=int(mode_config["max_retries"]),
    )


def planning_mode_name(config: AgentConfig) -> str:
    return "%s_%s" % (config.planning_granularity, config.violation_policy)


def planning_config_dict(config: AgentConfig) -> dict[str, Any]:
    return {
        "planning_mode": planning_mode_name(config),
        "planning_granularity": config.planning_granularity,
        "violation_policy": config.violation_policy,
        "max_plan_steps": config.max_plan_steps,
        "max_retries": config.max_retries,
        "max_branch_retries": config.max_retries,
    }


def required_tool_choice(name: str) -> dict[str, dict[str, str] | str]:
    return {"type": "function", "function": {"name": name}}


def feedback_json(value: list[dict] | None) -> str:
    if not value:
        return "none"
    return json.dumps(value[-5:], indent=2, sort_keys=True)


def format_action_trace(trace: list[dict]) -> str:
    return "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"


RESULT_SUMMARY_SYSTEM = """\
You are summarising the result of a verified workflow for the user.
Be concise (3-5 sentences). Cover what was done, what changed, and the current state."""


def summarize_result_text(
    goal: str,
    trace: list[dict],
    exec_results: list[str],
    model: str,
    *,
    llm_call: Callable[..., tuple[Any, Any]],
    client: Any,
) -> str:
    trace_text = "\n".join(
        f"  {i}. {step['action_label']}: {step['tool']}({json.dumps(step['args'])})"
        f"\n     result: {exec_results[i-1][:200] if i <= len(exec_results) else 'N/A'}"
        for i, step in enumerate(trace, 1)
    )
    content, _tool_calls = llm_call(client, [
        {"role": "system", "content": RESULT_SUMMARY_SYSTEM},
        {"role": "user", "content": f"Goal: {goal}\n\nExecuted steps:\n{trace_text}"},
    ], model)
    return content.strip()


BLOCKED_REQUEST_SYSTEM = """\
You are explaining to a user why their request could not be safely executed.
The plan was blocked because every candidate action violated a safety property, or retries were exhausted.
Be direct and specific: name the property that blocked the plan, and suggest a safe alternative if one exists."""


def explain_blocked_text(
    goal: str,
    initial_state: dict[str, bool],
    tried_actions: list[str],
    model: str,
    *,
    llm_call: Callable[..., tuple[Any, Any]],
    client: Any,
) -> str:
    state_text = "\n".join(
        f"  {'T' if value else 'F'}  {ap}"
        for ap, value in initial_state.items()
    )
    content, _tool_calls = llm_call(client, [
        {"role": "system", "content": BLOCKED_REQUEST_SYSTEM},
        {"role": "user", "content":
            f"Goal: {goal}\n\n"
            f"Initial state:\n{state_text}\n\n"
            f"Actions tried and rejected: {tried_actions}"},
    ], model)
    return content.strip() or "No feasible property-satisfying plan found."


def propose_next_action(
    *,
    goal: str,
    trace: list[dict],
    tried: list[str],
    model: str,
    config: AgentConfig,
    failed_attempts: list[dict] | None,
    system_prompt: str,
    property_block: str,
    llm_call: Callable[..., tuple[Any, Any]],
    client: Any,
    tools: list[dict[str, Any]],
    tool_name: str,
    tool_arguments: Callable[[Any, str], Any],
    max_tokens: int,
) -> dict | None:
    done_steps = format_action_trace(trace)

    _content, tool_calls = llm_call(client, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"Goal: {goal}\n\n"
            f"Steps done so far:\n{done_steps}\n\n"
            f"{property_block}"
            f"Failed plan attempts and backtrack feedback:\n{feedback_json(failed_attempts)}\n\n"
            f"Banned first actions at this planning point: {tried or 'none'}\n\n"
            f"What is the next action?"},
    ], model,
        tools=tools,
        tool_choice=required_tool_choice(tool_name),
        max_tokens=max_tokens,
    )

    result = tool_arguments(tool_calls, tool_name)
    return result if isinstance(result, dict) else None


def propose_action_plan(
    *,
    goal: str,
    trace: list[dict],
    tried: list[str],
    max_actions: int,
    model: str,
    config: AgentConfig,
    failed_attempts: list[dict] | None,
    system_prompt: str,
    property_block: str,
    llm_call: Callable[..., tuple[Any, Any]],
    client: Any,
    tools: list[dict[str, Any]],
    tool_name: str,
    tool_arguments: Callable[[Any, str], Any],
    max_tokens: int,
) -> dict | None:
    del config
    done_steps = format_action_trace(trace)

    _content, tool_calls = llm_call(client, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"Goal: {goal}\n\n"
            f"Steps already accepted:\n{done_steps}\n\n"
            f"{property_block}"
            f"Failed plan attempts and backtrack feedback:\n{feedback_json(failed_attempts)}\n\n"
            f"Banned first actions at this planning point: {tried or 'none'}\n\n"
            f"Return at most {max_actions} remaining action(s)."},
    ], model,
        tools=tools,
        tool_choice=required_tool_choice(tool_name),
        max_tokens=max_tokens,
    )

    result = tool_arguments(tool_calls, tool_name)
    if not isinstance(result, dict):
        return None
    raw_plan = result.get("plan", [])
    if not isinstance(raw_plan, list):
        return None
    return {
        "plan": [item for item in raw_plan[:max_actions] if isinstance(item, dict)],
        "finish_response": str(result.get("finish_response", "")),
    }


def format_ap_state_for_prompt(state: dict[str, bool], aps: list[str]) -> str:
    true_aps = [ap for ap in aps if state.get(ap, False)]
    false_aps = [ap for ap in aps if not state.get(ap, False)]
    return (
        "TRUE APs:\n"
        + ("\n".join(f"- {ap}" for ap in true_aps) or "- (none)")
        + "\n\nFALSE APs:\n"
        + ("\n".join(f"- {ap}" for ap in false_aps) or "- (none)")
    )


def render_ap_definition(
    ap: str,
    *,
    spec_by_name: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    evidence_field: str,
    evidence_limit_key: str = "observer_evidence_limit",
    evidence_label_key: str = "observer_evidence_label",
) -> str:
    spec = spec_by_name.get(ap) or {}
    description = str(spec.get("description") or ap)
    evidence_items = spec.get(evidence_field) or []
    if not evidence_items:
        return description
    limit = int(metadata.get(evidence_limit_key, 4))
    label = str(metadata.get(evidence_label_key, "Observer evidence"))
    evidence_text = "; ".join(str(item) for item in evidence_items[:limit])
    if len(evidence_items) > limit:
        evidence_text += "; ..."
    return f"{description}\n{label}: {evidence_text}"


def format_recent_action_trace(trace: list[dict] | None, *, limit: int = 3) -> str:
    recent_steps = trace[-limit:] if trace else []
    return "\n".join(
        f"- {step.get('action_label', 'unknown')}: "
        f"{step.get('tool', 'none')}({json.dumps(step.get('args') or {})})"
        for step in recent_steps
    ) or "- (none)"


def predict_ap_value(
    *,
    ap: str,
    current_val: bool,
    action_label: str,
    tool: str,
    args: dict,
    model: str,
    current_state: dict[str, bool] | None,
    trace: list[dict] | None,
    aps: list[str],
    system_prompt: str,
    ap_definition: Callable[[str], str],
    action_prediction_notes: Callable[[str, dict], str],
    llm_call: Callable[..., tuple[Any, Any]],
    client: Any,
    tools: list[dict[str, Any]],
    tool_name: str,
    tool_arguments: Callable[[Any, str], Any],
) -> tuple[bool, str]:
    state_text = (
        format_ap_state_for_prompt(current_state, aps)
        if current_state is not None
        else f"- {ap}: {'TRUE' if current_val else 'FALSE'}"
    )
    _content, tool_calls = llm_call(client, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"Atomic proposition: {ap}\n"
            f"Definition: {ap_definition(ap)}\n"
            f"Current value: {'TRUE' if current_val else 'FALSE'}\n\n"
            f"Full current AP state before this action:\n{state_text}\n\n"
            f"Recently accepted planning steps:\n{format_recent_action_trace(trace)}\n\n"
            f"Action: {action_label}\n"
            f"Command: {tool}({json.dumps(args)})\n\n"
            f"Execution/tool notes:\n{action_prediction_notes(tool, args)}"},
    ], model,
        tools=tools,
        tool_choice=required_tool_choice(tool_name),
    )
    result = tool_arguments(tool_calls, tool_name)
    if isinstance(result, dict) and "value" in result:
        return bool(result["value"]), result.get("reason", "")
    return current_val, "tool call missing — unchanged"


def predict_action_state(
    *,
    action_label: str,
    tool: str,
    args: dict,
    current_state: dict[str, bool],
    model: str,
    trace: list[dict] | None,
    aps: list[str],
    predict_ap: Callable[..., tuple[bool, str]],
) -> dict[str, bool]:
    state_after = dict(current_state)
    for ap in aps:
        current_val = bool(current_state.get(ap, False))
        new_val, _ = predict_ap(
            ap=ap,
            current_val=current_val,
            action_label=action_label,
            tool=tool,
            args=args,
            model=model,
            current_state=current_state,
            trace=trace,
        )
        state_after[ap] = new_val
    state_after["last_action"] = action_label
    return state_after


def observe_execution_ap_state(
    *,
    aps: list[str],
    observe_ap: Callable[[str], bool],
    action_label: str,
) -> dict[str, bool]:
    ap_state = observe_ap_values(aps, observe_ap)
    ap_state["last_action"] = action_label
    return ap_state


def execute_tool_step(
    step: dict,
    *,
    tool_impl: dict[str, Callable[..., str]],
    no_op_result: str = "(no-op)",
) -> str:
    tool = step.get("tool", "none")
    args = step.get("args") or {}
    if tool == "none" or not args:
        return no_op_result
    impl = tool_impl.get(tool)
    if impl is None:
        return f"[error] unknown tool '{tool}'"
    try:
        return impl(**args)
    except Exception as exc:
        return f"[error] {exc}"


def execution_observer(
    *,
    aps: list[str],
    observe_ap: Callable[[str], bool],
) -> Callable[[str], dict[str, bool]]:
    return lambda action_label: observe_execution_ap_state(
        aps=aps,
        observe_ap=observe_ap,
        action_label=action_label,
    )


def trace_verifier(
    *,
    aps: list[str],
    properties: list[dict],
    module_name: str,
    timeout: int,
) -> Callable[..., tuple[bool, str, str, dict]]:
    return lambda s0, trace, is_complete_trace=False: run_verification(
        s0,
        trace,
        aps=aps,
        properties=properties,
        module_name=module_name,
        timeout=timeout,
        is_complete_trace=is_complete_trace,
    )


def state_differ(aps: list[str]) -> Callable[[dict[str, bool], dict[str, bool]], list[dict[str, object]]]:
    return lambda previous, current: diff_ap_states(previous, current, aps)


def diff_ap_states(
    previous: dict[str, bool],
    current: dict[str, bool],
    aps: list[str],
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for name in aps:
        prev_val = previous.get(name)
        curr_val = current.get(name)
        if prev_val is not None and prev_val != curr_val:
            changes.append({"name": name, "before": prev_val, "after": curr_val})
    return changes


def run_verification(
    s0: dict[str, bool],
    trace: list[dict],
    *,
    aps: list[str],
    properties: list[dict],
    module_name: str,
    timeout: int,
    is_complete_trace: bool = False,
) -> tuple[bool, str, str, dict]:
    action_labels = [str(step["action_label"]) for step in trace]
    states_after = [step["state_after"] for step in trace]
    result = verify_fsm_trace(
        s0,
        action_labels,
        states_after,
        aps,
        properties,
        module_name=module_name,
        timeout=timeout,
        is_complete_trace=is_complete_trace,
    )
    tlc_result = result["tlc_result"]
    summary = "; ".join(tlc_result.get("violations", [])) or tlc_result.get("reason", "")
    return result["passed"], result["tla_spec"], summary, result


def planning_action_from_step(step: dict) -> dict:
    return make_action(
        step.get("action_label"),
        step.get("tool", "none"),
        step.get("args") or {},
    )


def property_prompt_block(policy: str, properties: list[dict]) -> str:
    if policy == VIOLATION_IGNORE:
        return ""
    return (
        f"Property policy:\n{property_policy_text(policy)}\n\n"
        f"Properties to avoid violating:\n{property_guidance_text(properties, bullet=True)}\n\n"
    )


def state_path_from_trace(trace: list[dict]) -> list[dict]:
    return [
        make_state_path_entry(
            planning_action_from_step(step),
            step.get("state_after") or {},
        )
        for step in trace
    ]


def branch_retry_budget(config: AgentConfig) -> int:
    if config.violation_policy in NONBLOCKING_VIOLATION_POLICIES:
        return max(1, config.max_retries)
    return config.max_retries


def proposal_finish_response(proposal: dict | None) -> str:
    if not isinstance(proposal, dict):
        return "done"
    return str(
        proposal.get("finish_response")
        or proposal.get("rationale")
        or "done"
    )


def request_plan_bundle(
    *,
    goal: str,
    accepted_trace: list[dict],
    failed_attempts: list[dict],
    banned_first_actions: list[str],
    depth: int,
    model: str,
    config: AgentConfig,
    propose_batch: Callable[..., dict | None],
    propose_step: Callable[..., dict | None],
) -> dict | None:
    if config.planning_granularity == PLANNING_BATCH:
        max_actions = max(1, config.max_plan_steps - depth)
        return propose_batch(
            goal,
            accepted_trace,
            banned_first_actions,
            max_actions,
            model,
            config,
            failed_attempts=failed_attempts,
        )

    proposal = propose_step(
        goal,
        accepted_trace,
        banned_first_actions,
        model,
        config,
        failed_attempts=failed_attempts,
    )
    if proposal is None:
        return None
    if str(proposal.get("action_label", "")) == "goal_satisfied":
        return {
            "plan": [],
            "finish_response": proposal_finish_response(proposal),
        }
    return {
        "plan": [proposal],
        "finish_response": proposal_finish_response(proposal),
    }


def verify_candidate(
    *,
    proposal: dict,
    s0: dict[str, bool],
    s_current: dict[str, bool],
    trace: list[dict],
    model: str,
    predict_state: Callable[..., dict[str, bool]],
    run_trace_verification: Callable[..., tuple[bool, str, str, dict]],
    is_complete_trace: bool = False,
) -> dict:
    import json

    action_label = proposal.get("action_label", "unknown")
    tool = proposal.get("tool", "none")
    args = proposal.get("args") or {}

    print(f"    -> {action_label}: {tool}({json.dumps(args)})")
    s_after = predict_state(action_label, tool, args, s_current, model, trace=trace)
    candidate = {
        "action_label": action_label,
        "tool": tool,
        "args": args,
        "state_before": dict(s_current),
        "state_after": s_after,
    }

    passed, tla_spec, violations_str, verification_result = run_trace_verification(
        s0,
        trace + [candidate],
        is_complete_trace=is_complete_trace,
    )
    violations_list = [v for v in violations_str.split(";") if v.strip()] if violations_str else []

    tlc_result = verification_result.get("tlc_result", {})
    skipped = bool(tlc_result.get("skipped"))
    verif = make_verification(
        passed=passed,
        properties_checked=verification_result.get("properties_checked", []),
        violations=violations_list,
        tla_spec=tla_spec,
        skipped=skipped,
    )
    failure_feedback = None
    if not passed:
        failure_feedback = {
            "type": "verification_skipped" if skipped else "tla_property_violation",
            "action": planning_action_from_step(candidate),
            "violations": violations_list,
            "message": violations_str or (
                "TLC verification was skipped." if skipped else "TLC verification failed."
            ),
        }
    return {
        "candidate": candidate,
        "state_after": s_after,
        "passed": passed,
        "violations_str": violations_str,
        "verification": verif,
        "failure": failure_feedback,
    }


def execute_plan(
    trace: list[dict],
    *,
    s0: dict[str, bool],
    planning_tree: dict | None,
    result_record: dict[str, Any] | None,
    result_path: str | None,
    execute_step: Callable[[dict], str],
    observe_state: Callable[[str], dict[str, bool]],
    run_trace_verification: Callable[..., tuple[bool, str, str, dict]],
    diff_states: Callable[[dict[str, bool], dict[str, bool]], list[dict[str, object]]],
) -> list[str]:
    results: list[str] = []
    if not trace:
        return results

    executed_nodes = accepted_nodes(planning_tree) if planning_tree is not None else []
    observed_trace: list[dict] = []
    previous_ap_state = s0

    print(f"\n\033[35m[Phase 5] Executing {len(trace)} verified step(s):\033[0m")
    for i, step in enumerate(trace, 1):
        tool = step.get("tool", "none")
        args = step.get("args") or {}
        action_label = str(step["action_label"])
        print(f"\n  \033[33m[exec {i}/{len(trace)}] {action_label}: {tool}({json.dumps(args)})\033[0m")

        result = execute_step(step)
        if result != "(no-op)":
            print(f"  \033[2m{result}\033[0m")
        results.append(result)

        ap_state = observe_state(action_label)
        observed_step = dict(step)
        observed_step["state_after"] = ap_state
        observed_trace.append(observed_step)
        _, _, _, tla_result = run_trace_verification(
            s0,
            observed_trace,
            is_complete_trace=(i == len(trace)),
        )
        tlc_result = tla_result.get("tlc_result", {})
        if tlc_result.get("skipped"):
            reason = str(tlc_result.get("reason") or "TLC verification was skipped.")
            print(f"  \033[33mexecution verification SKIPPED: {reason[:120]}\033[0m")
        elif tla_result["passed"]:
            print("  \033[32mexecution verification PASS\033[0m")
        else:
            violations = "; ".join(tlc_result.get("violations", []))
            print(f"  \033[31mexecution verification FAIL: {violations[:120]}\033[0m")

        node_index = i - 1
        if node_index < len(executed_nodes):
            annotate_node_executed(
                executed_nodes[node_index],
                execution_step=node_index,
                execution_result=result,
                ap_state=ap_state,
                ap_changes=diff_states(previous_ap_state, ap_state),
                tla_verification=tla_result,
            )
        previous_ap_state = ap_state
        if result_record is not None:
            checkpoint_result(result_record, result_path)

    return results


def accept_node(node: dict, candidate: dict, check: dict) -> None:
    result = (
        "accepted_with_ignored_violations"
        if check.get("ignored_by_policy")
        else "accepted"
    )
    set_node_outcome(
        node,
        result=result,
        action_label=candidate.get("action_label", "unknown"),
        tool=candidate.get("tool", "none"),
        args=candidate.get("args") or {},
        state_after=check.get("state_after") or {},
        verification=check.get("verification"),
    )


def search_plan(
    *,
    goal: str,
    s0: dict[str, bool],
    current_state: dict[str, bool],
    accepted_trace: list[dict],
    depth: int,
    planning_tree: dict,
    parent_node_id: int | None,
    inherited_failures: list[dict],
    hint_plan: list[dict] | None,
    model: str,
    config: AgentConfig,
    checkpoint: Callable[[], None],
    request_plan: Callable[..., dict | None],
    verify_action: Callable[..., dict],
) -> dict:
    """Search recursively using one planning-tree node per candidate action."""
    import json

    if len(planning_tree["nodes"]) >= config.max_plan_steps:
        return {
            "success": False,
            "failure": {
                "type": "max_tries",
                "depth": depth,
                "nodes_created": len(planning_tree["nodes"]),
                "message": (
                    "Planning exceeded the max node budget of %d."
                    % config.max_plan_steps
                ),
            },
            "finish_response": "No feasible property-satisfying plan found.",
        }

    node_id = len(planning_tree["nodes"])
    node = make_planning_node(
        node_id=node_id,
        parent_node_id=parent_node_id,
        depth=depth,
        state_before=dict(current_state),
        state_path=state_path_from_trace(accepted_trace),
    )
    append_node(planning_tree, node, link_parent=True)
    checkpoint()

    failed_attempts = list(inherited_failures)
    banned_first_actions: list[str] = []
    current_hint = (
        copy.deepcopy(hint_plan)
        if hint_plan and config.planning_granularity == PLANNING_BATCH
        else []
    )

    for child_index in range(branch_retry_budget(config)):
        if current_hint:
            print(f"  [depth {depth+1} attempt {child_index+1}] reusing planned suffix")
            plan_bundle = {
                "plan": copy.deepcopy(current_hint),
                "finish_response": "done",
            }
            attempt = make_attempt(
                retry_index=child_index,
                accepted=False,
                planner_decision=plan_bundle,
                plan_source="hint_plan",
                hint_plan_length=len(current_hint),
            )
            current_hint = []
        else:
            print(f"  [depth {depth+1} attempt {child_index+1}] proposing action...")
            try:
                plan_bundle = request_plan(
                    goal=goal,
                    current_state=current_state,
                    accepted_trace=accepted_trace,
                    failed_attempts=failed_attempts,
                    banned_first_actions=banned_first_actions,
                    depth=depth,
                    model=model,
                    config=config,
                )
            except Exception as exc:
                failure = {
                    "type": "planning_error",
                    "depth": depth,
                    "child_index": child_index,
                    "message": str(exc),
                }
                node["attempts"].append(
                    make_attempt(
                        retry_index=child_index,
                        accepted=False,
                        error=str(exc),
                        failure_feedback=failure,
                    )
                )
                checkpoint()
                failed_attempts.append(failure)
                current_hint = []
                continue

            if plan_bundle is None:
                failure = {
                    "type": "planning_error",
                    "depth": depth,
                    "child_index": child_index,
                    "message": "Planner returned unparsable or invalid JSON.",
                }
                node["attempts"].append(
                    make_attempt(
                        retry_index=child_index,
                        accepted=False,
                        error=failure["message"],
                        failure_feedback=failure,
                    )
                )
                checkpoint()
                failed_attempts.append(failure)
                current_hint = []
                continue

            attempt = make_attempt(
                retry_index=child_index,
                accepted=False,
                planner_decision=plan_bundle,
            )

        plan = plan_bundle.get("plan", [])
        if not isinstance(plan, list):
            plan = []
        if config.planning_granularity == PLANNING_STEP and len(plan) > 1:
            attempt["truncated_to_single_step"] = True
            plan = plan[:1]
            plan_bundle = dict(plan_bundle)
            plan_bundle["plan"] = plan
            attempt["planner_decision"] = plan_bundle

        first_action = plan[0] if plan else None
        if isinstance(first_action, dict) and first_action.get("action_label") == "goal_satisfied":
            plan = []
            plan_bundle = dict(plan_bundle)
            plan_bundle["plan"] = []
            plan_bundle["finish_response"] = proposal_finish_response(first_action)
            attempt["planner_decision"] = plan_bundle

        if not plan:
            finish_response = str(plan_bundle.get("finish_response") or "done")
            attempt["accepted"] = True
            attempt["finish"] = True
            node["attempts"].append(attempt)
            set_node_outcome(
                node,
                result="finish",
                finish_response=finish_response,
            )
            checkpoint()
            return {
                "success": True,
                "plan": [],
                "finish_response": finish_response,
                "node_id": node_id,
            }

        action = plan[0]
        tail = plan[1:] if config.planning_granularity == PLANNING_BATCH else []
        check = verify_action(
            proposal=action,
            s0=s0,
            s_current=current_state,
            trace=accepted_trace,
            model=model,
            is_complete_trace=(
                not tail and config.planning_granularity == PLANNING_BATCH
            ),
        )
        candidate = check["candidate"]
        attempt["action"] = candidate
        attempt["proposal"] = action
        attempt["predicted_state_after"] = check["state_after"]
        attempt["verification"] = check["verification"]

        if not check["passed"]:
            failure = check["failure"]
            can_continue = (
                config.violation_policy in NONBLOCKING_VIOLATION_POLICIES
                and failure is not None
            )
            if can_continue:
                attempt["ignored_property_violation"] = failure
                check["ignored_by_policy"] = True
            else:
                if check["violations_str"]:
                    print(f"    FAIL: {check['violations_str'][:120]}")
                attempt["failure_feedback"] = failure
                node["attempts"].append(attempt)
                checkpoint()
                if failure is not None:
                    failed_attempts.append(failure)
                banned_first_actions.append(str(candidate.get("action_label", "unknown")))
                current_hint = []
                continue

        if check["passed"]:
            print("    PASS")
        elif check["violations_str"]:
            print(f"    {config.violation_policy.upper()} VIOLATION: {check['violations_str'][:120]}")
        else:
            print(f"    {config.violation_policy.upper()} VERIFICATION FAILURE")

        if not tail and config.planning_granularity == PLANNING_BATCH:
            attempt["accepted"] = True
            node["attempts"].append(attempt)
            accept_node(node, candidate, check)
            checkpoint()
            return {
                "success": True,
                "plan": [candidate],
                "finish_response": str(plan_bundle.get("finish_response") or "done"),
                "node_id": node_id,
            }

        child_result = search_plan(
            goal=goal,
            s0=s0,
            current_state=check["state_after"],
            accepted_trace=accepted_trace + [candidate],
            depth=depth + 1,
            planning_tree=planning_tree,
            parent_node_id=node_id,
            inherited_failures=[],
            hint_plan=tail if config.planning_granularity == PLANNING_BATCH else None,
            model=model,
            config=config,
            checkpoint=checkpoint,
            request_plan=request_plan,
            verify_action=verify_action,
        )
        attempt["child_node_id"] = child_result.get("node_id")

        if child_result.get("success"):
            attempt["accepted"] = True
            node["attempts"].append(attempt)
            accept_node(node, candidate, check)
            checkpoint()
            return {
                "success": True,
                "plan": [candidate] + child_result.get("plan", []),
                "finish_response": child_result.get(
                    "finish_response",
                    str(plan_bundle.get("finish_response") or "done"),
                ),
                "node_id": node_id,
            }

        child_failure = child_result.get("failure") or {
            "type": "child_failure",
            "depth": depth + 1,
            "message": "Child subtree exhausted.",
        }
        attempt["child_failure"] = child_failure
        node["attempts"].append(attempt)
        checkpoint()
        if child_failure.get("type") == "max_tries":
            set_node_outcome(node, result="backtracked", failure=child_failure)
            checkpoint()
            return {
                "success": False,
                "failure": child_failure,
                "finish_response": "No feasible property-satisfying plan found.",
                "node_id": node_id,
            }
        failed_attempts.append(child_failure)
        current_hint = []

    exhaustion_failure = {
        "type": "branch_exhausted",
        "depth": depth,
        "node_id": node_id,
        "failed_attempts": failed_attempts,
        "message": (
            "All %d action attempts at this node were exhausted."
            % branch_retry_budget(config)
        ),
    }
    set_node_outcome(node, result="backtracked", failure=exhaustion_failure)
    checkpoint()
    return {
        "success": False,
        "failure": exhaustion_failure,
        "finish_response": "No feasible property-satisfying plan found.",
        "node_id": node_id,
    }


def phase4_plan(
    goal: str,
    s0: dict[str, bool],
    model: str,
    *,
    config: AgentConfig,
    properties: list[dict],
    request_plan: Callable[..., dict | None],
    verify_action: Callable[..., dict],
    planning_tree: dict | None = None,
    result_record: dict[str, Any] | None = None,
    result_path: str | None = None,
) -> tuple[list[dict], dict, bool]:
    """Run the shared planning phase and return ``(trace, tree, feasible)``."""
    planning_tree = planning_tree or make_planning_tree(
        mode=planning_mode_name(config),
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        max_branch_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=properties,
        initial_state=s0,
    )
    planning_tree.setdefault("max_branch_retries", config.max_retries)

    def checkpoint() -> None:
        if result_record is not None:
            checkpoint_result(result_record, result_path)

    print(
        "\n\033[35m[Phase 4] Planning with TLC verification "
        f"({config.planning_granularity}, violations={config.violation_policy}, "
        f"retries={config.max_retries})...\033[0m"
    )

    result = search_plan(
        goal=goal,
        s0=s0,
        current_state=dict(s0),
        accepted_trace=[],
        depth=0,
        planning_tree=planning_tree,
        parent_node_id=None,
        inherited_failures=[],
        hint_plan=None,
        model=model,
        config=config,
        checkpoint=checkpoint,
        request_plan=request_plan,
        verify_action=verify_action,
    )
    feasible = bool(result.get("success"))
    if feasible:
        return result.get("plan", []), planning_tree, True

    if result.get("failure"):
        planning_tree["failure"] = result["failure"]
    return [], planning_tree, False


def rejected_action_labels(planning_tree: dict) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: Any) -> None:
        text = str(label or "").strip()
        if text and text not in seen:
            labels.append(text)
            seen.add(text)

    for node in planning_tree.get("nodes", []):
        if not isinstance(node, dict):
            continue
        action = node.get("action") or {}
        if node.get("result") == "rejected" and isinstance(action, dict):
            add(action.get("label"))
        for attempt in node.get("attempts", []):
            if not isinstance(attempt, dict) or attempt.get("accepted"):
                continue
            proposal = attempt.get("proposal") or attempt.get("action") or {}
            if isinstance(proposal, dict):
                add(proposal.get("action_label") or proposal.get("label"))
    return labels


def make_agent_session(
    *,
    agent: str,
    model: str,
    domain: str,
    request: str,
    work_dir: str,
    properties: list[dict],
    config: AgentConfig,
) -> dict:
    return make_session(
        agent=agent,
        model=model,
        domain=domain,
        request=request,
        work_dir=work_dir,
        properties=properties,
        planning_config=planning_config_dict(config),
    )


def save_agent_session(
    turns: list[dict],
    *,
    model: str,
    agent: str,
    domain: str,
    work_dir: str,
    properties: list[dict],
    config: AgentConfig,
    result_dir: Path | None,
) -> Path:
    latest_turn = turns[-1] if turns else None
    if latest_turn is not None:
        session = dict(latest_turn)
    else:
        session = make_agent_session(
            agent=agent,
            model=model,
            domain=domain,
            request="(multi-turn session)",
            work_dir=work_dir,
            properties=properties,
            config=config,
        )
    if isinstance(session.get("planning_tree"), dict):
        session["planning_tree"]["tree_summary"] = build_tree_summary(session["planning_tree"])
    session["turn_count"] = len(turns)
    session["latest_turn_index"] = len(turns) - 1 if turns else None
    session["turns"] = turns
    path = write_result(session, result_dir)
    if path is None:
        raise RuntimeError("Result saving is disabled")
    return Path(path)


def finalize_successful_plan(planning_tree: dict) -> None:
    mark_feasible(planning_tree, accepted_plan_from_nodes(planning_tree))


def result_notice(response: str, result_path: str | None) -> str:
    return append_result_notice(response, result_path)


def run_agent_flow(
    *,
    goal: str,
    model: str,
    config: AgentConfig,
    spec: AgentFlowSpec,
) -> tuple[str, dict]:
    """Run the standard observe-plan-execute-summarize FSM agent flow."""
    planning_tree = make_planning_tree(
        mode=planning_mode_name(config),
        max_steps=config.max_plan_steps,
        max_retries=config.max_retries,
        max_branch_retries=config.max_retries,
        planning_granularity=config.planning_granularity,
        violation_policy=config.violation_policy,
        properties=spec.properties,
    )
    turn = make_session(
        agent=spec.agent,
        model=model,
        domain=spec.domain,
        request=goal,
        work_dir=spec.work_dir,
        properties=spec.properties,
        planning_config=planning_config_dict(config),
    )
    turn["planning_tree"] = planning_tree
    turn["status"] = "planning"
    result_path = start_result_session(turn, spec.result_dir)

    print(f"\n\033[35m[Phase 3] Observing initial state s0...\033[0m")
    s0 = spec.observe_initial_state(model)
    planning_tree["initial_state"] = s0
    checkpoint_result(turn, result_path)

    def propose_step(
        goal_arg: str,
        trace_arg: list[dict],
        tried: list[str],
        model_arg: str,
        config_arg: AgentConfig,
        failed_attempts: list[dict] | None = None,
    ) -> dict | None:
        return propose_next_action(
            goal=goal_arg,
            trace=trace_arg,
            tried=tried,
            model=model_arg,
            config=config_arg,
            failed_attempts=failed_attempts,
            system_prompt=spec.propose_step_prompt,
            property_block=property_prompt_block(config_arg.violation_policy, spec.properties),
            llm_call=spec.llm_call,
            client=spec.client,
            tools=spec.action_proposal_tool,
            tool_name=spec.action_proposal_tool_name,
            tool_arguments=spec.tool_arguments,
            max_tokens=spec.max_planning_tokens,
        )

    def propose_batch(
        goal_arg: str,
        trace_arg: list[dict],
        tried: list[str],
        max_actions: int,
        model_arg: str,
        config_arg: AgentConfig,
        failed_attempts: list[dict] | None = None,
    ) -> dict | None:
        return propose_action_plan(
            goal=goal_arg,
            trace=trace_arg,
            tried=tried,
            max_actions=max_actions,
            model=model_arg,
            config=config_arg,
            failed_attempts=failed_attempts,
            system_prompt=spec.propose_batch_prompt,
            property_block=property_prompt_block(config_arg.violation_policy, spec.properties),
            llm_call=spec.llm_call,
            client=spec.client,
            tools=spec.plan_proposal_tool,
            tool_name=spec.plan_proposal_tool_name,
            tool_arguments=spec.tool_arguments,
            max_tokens=spec.max_planning_tokens,
        )

    def request_plan(**kwargs: Any) -> dict | None:
        kwargs.pop("current_state", None)
        return request_plan_bundle(
            **kwargs,
            propose_batch=propose_batch,
            propose_step=propose_step,
        )

    def predict_ap(**kwargs: Any) -> tuple[bool, str]:
        return predict_ap_value(
            **kwargs,
            aps=spec.aps,
            system_prompt=spec.predict_ap_prompt,
            ap_definition=lambda name: render_ap_definition(
                name,
                spec_by_name=spec.ap_spec_by_name,
                metadata=spec.ap_catalog_metadata,
                evidence_field=spec.ap_evidence_field,
            ),
            action_prediction_notes=spec.action_prediction_notes,
            llm_call=spec.llm_call,
            client=spec.client,
            tools=spec.ap_prediction_tool,
            tool_name=spec.ap_prediction_tool_name,
            tool_arguments=spec.tool_arguments,
        )

    def predict_state(
        action_label: str,
        tool: str,
        args: dict,
        current_state: dict[str, bool],
        model_arg: str,
        trace: list[dict] | None = None,
    ) -> dict[str, bool]:
        return predict_action_state(
            action_label=action_label,
            tool=tool,
            args=args,
            current_state=current_state,
            model=model_arg,
            trace=trace,
            aps=spec.aps,
            predict_ap=predict_ap,
        )

    run_trace_verification = trace_verifier(
        aps=spec.aps,
        properties=spec.properties,
        module_name=spec.verification_module_name,
        timeout=spec.verification_timeout,
    )

    def verify_action(**kwargs: Any) -> dict:
        return verify_candidate(
            **kwargs,
            predict_state=predict_state,
            run_trace_verification=run_trace_verification,
        )

    trace, planning_tree, feasible = phase4_plan(
        goal=goal,
        s0=s0,
        model=model,
        config=config,
        properties=spec.properties,
        request_plan=request_plan,
        verify_action=verify_action,
        planning_tree=planning_tree,
        result_record=turn,
        result_path=result_path,
    )
    if feasible:
        finalize_successful_plan(planning_tree)
    else:
        planning_tree["feasible"] = False
        planning_tree["accepted_plan"] = []

    if feasible and trace:
        turn["status"] = "executing"
        checkpoint_result(turn, result_path)
        exec_results = execute_plan(
            trace,
            s0=s0,
            planning_tree=planning_tree,
            result_record=turn,
            result_path=result_path,
            execute_step=spec.execute_step,
            observe_state=execution_observer(
                aps=spec.aps,
                observe_ap=spec.observe_execution_ap(model),
            ),
            run_trace_verification=run_trace_verification,
            diff_states=state_differ(spec.aps),
        )
    else:
        exec_results = []

    if feasible:
        response = (
            spec.summarize_result(goal, trace, exec_results, model)
            if trace
            else spec.already_satisfied_response
        )
        turn["status"] = "finished"
    else:
        tried_all = rejected_action_labels(planning_tree)
        response = spec.explain_blocked(goal, s0, tried_all, model)
        turn["status"] = "infeasible"

    planning_tree["tree_summary"] = build_tree_summary(planning_tree)
    turn["final_message"] = response
    result_path = write_result(turn, spec.result_dir, result_path)
    return append_result_notice(response, result_path), turn
