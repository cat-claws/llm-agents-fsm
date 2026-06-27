"""Shared planning-mode vocabulary for llm-agents-fsm agents."""

PLANNING_STEP = "step"
PLANNING_BATCH = "batch"

VIOLATION_RETRY = "retry"
VIOLATION_IGNORE = "ignore"
VIOLATION_ADVISORY = "advisory"

NONBLOCKING_VIOLATION_POLICIES = {VIOLATION_IGNORE, VIOLATION_ADVISORY}
ACCEPTED_NODE_RESULTS = {"accepted", "accepted_with_ignored_violations"}
FINISH_NODE_RESULT = "finish"

PLANNING_MODE_FSM = "fsm"
PLANNING_MODE_PLAN = "plan"
PLANNING_MODE_ADVISORY = "advisory"

PLANNING_MODE_PRESETS = {
    PLANNING_MODE_FSM: (PLANNING_BATCH, VIOLATION_RETRY, None),
    PLANNING_MODE_PLAN: (PLANNING_BATCH, VIOLATION_IGNORE, 1),
    PLANNING_MODE_ADVISORY: (PLANNING_BATCH, VIOLATION_ADVISORY, 1),
}
PLANNING_GRANULARITIES = {PLANNING_STEP, PLANNING_BATCH}
VIOLATION_POLICIES = {VIOLATION_RETRY, VIOLATION_IGNORE, VIOLATION_ADVISORY}

PROPERTY_POLICY_TEXT = {
    VIOLATION_ADVISORY: (
        "Property checks are advisory in this run: use the properties as guidance "
        "and try to avoid violating them. Verification still runs and records "
        "violations, but a property violation does not stop the plan, trigger "
        "retries, or cause backtracking."
    ),
    VIOLATION_RETRY: (
        "Property checks are blocking in this run: if a candidate violates a property, "
        "that branch is rejected and planning retries with a different candidate."
    ),
}


def _choice(name, value, default, allowed, invalid="default"):
    raw = (value or default).strip().lower()
    if raw in allowed:
        return raw
    if invalid == "raise":
        raise ValueError("%s must be one of %s, got %r" % (name, ", ".join(sorted(allowed)), value))
    return default if default in allowed else sorted(allowed)[0]


def planning_preset_config(value, *, retry_default, default=PLANNING_MODE_FSM, invalid="default"):
    preset = _choice("planning preset", value, default, PLANNING_MODE_PRESETS, invalid)
    granularity, policy, retries = PLANNING_MODE_PRESETS[preset]
    return {
        "planning_granularity": granularity,
        "violation_policy": policy,
        "max_retries": retry_default if retries is None else retries,
    }


def normalize_planning_granularity(value, *, default=PLANNING_STEP, invalid="default"):
    return _choice("planning_granularity", value, default, PLANNING_GRANULARITIES, invalid)


def normalize_violation_policy(value, *, default=VIOLATION_RETRY, invalid="default"):
    return _choice("violation_policy", value, default, VIOLATION_POLICIES, invalid)


def property_policy_text(policy):
    return PROPERTY_POLICY_TEXT.get(policy, "")


def property_guidance_text(properties, *, bullet=False):
    if not properties:
        return "(no active properties)"
    prefix = "- " if bullet else ""
    return "\n".join(
        "%s%s: %s" % (prefix, item.get("id", "?"), item.get("natural_language") or item.get("ltl") or "")
        for item in properties
    )
