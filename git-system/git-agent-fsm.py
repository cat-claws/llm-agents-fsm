#!/usr/bin/env python3
"""
git-agent-fsm — property-verified git agent.

Code-driven phases:
  Phase 0: Python — parse inputs (cwd, properties dir)
  Phase 1: Python — load properties, extract + classify APs
  Phase 3: LLM (prompt 3A) — select observation commands
           LLM (prompt 3B) × N_APs — evaluate each AP → s0
  Phase 4: loop until goal reached or retries exhausted:
             LLM (prompt 4A) — propose next action
             LLM (prompt 4B) — predict state_after for all APs
             Python — write TLA+ spec
             Python — run TLC
             Python — parse PASS/FAIL, branch
  Phase 5: Python — execute verified trace
  Response: LLM (prompt 5) — success summary
            LLM (prompt 7) — blocked explanation
  Guard:    LLM (prompt 6) — out-of-scope check (runs first)
"""
from __future__ import annotations

import datetime
import json
import os
import random as _random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import openai

# ── utils path ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.tla_verifier import (   # noqa: E402
    build_tla_spec,
    run_tlc,
    load_properties as _load_properties_file,
    has_liveness,
)
from utils.session import (        # noqa: E402
    make_session,
    make_planning_node,
    make_verification,
    make_execution_step,
    save_session as _save_session_util,
)

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL    = "gpt-4o-mini"
MAX_PLAN_STEPS   = 10
MAX_RETRIES      = 3
MAX_OBS_ROUNDS   = 6    # tool-call rounds in phase 3A
MAX_OUTPUT_CHARS = 4000
CMD_TIMEOUT      = 20

_RESOURCES_DIR      = Path(__file__).resolve().parent / "resources"
_PROPERTIES_FILE    = _RESOURCES_DIR / "GIT_PROPERTIES_AST.json"
_AP_CANDIDATES_FILE = _RESOURCES_DIR / "GIT_AP_CANDIDATES.json"

WORK_DIR = Path.cwd().resolve()

# ── Phase 0: allowlists ───────────────────────────────────────────────────────

ALLOWED_GIT = {
    "status", "log", "diff", "show", "branch", "checkout", "switch",
    "add", "commit", "restore", "reset", "rebase", "merge", "fetch",
    "pull", "push", "remote", "rev-parse", "stash", "tag", "blame",
    "shortlog", "describe", "reflog", "cherry-pick", "revert", "clean",
    "ls-files", "ls-remote", "submodule", "config", "init", "clone",
}

SHELL_BINS: dict[str, str] = {
    "ls":   "/bin/ls",    "pwd":  "/bin/pwd",   "cat":  "/bin/cat",
    "echo": "/bin/echo",  "find": "/usr/bin/find", "grep": "/usr/bin/grep",
    "wc":   "/usr/bin/wc", "head": "/usr/bin/head", "tail": "/usr/bin/tail",
    "stat": "/usr/bin/stat",
}
_SHELL_NAMES = ", ".join(sorted(SHELL_BINS))

# ── Phase 1: load properties + extract APs ───────────────────────────────────

def _collect_aps_from_ast(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "ap":
            out.add(node["name"])
        for v in node.values():
            _collect_aps_from_ast(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_aps_from_ast(v, out)

def _aps_from(props: list[dict]) -> tuple[list[str], list[str]]:
    aps: set[str] = set()
    for p in props:
        _collect_aps_from_ast(p.get("ast", p), aps)
    return (sorted(a for a in aps if not a.startswith("(transition)")),
            sorted(a for a in aps if a.startswith("(transition)")))

# How many properties to sample per session (set to None to use all)
PROPERTY_SAMPLE_SIZE: int | None = 5

_ALL_PROPS: list[dict] = _load_properties_file(_PROPERTIES_FILE)

def _sample_properties(props: list[dict], n: int | None) -> list[dict]:
    if n is None or n >= len(props):
        return props
    return _random.sample(props, n)

PROPERTIES = _sample_properties(_ALL_PROPS, PROPERTY_SAMPLE_SIZE)

STATE_APS, TRANS_APS = _aps_from(PROPERTIES)
ALL_APS = STATE_APS + TRANS_APS

# AP catalog: name → description (for prompting the LLM during observation)
def _load_ap_catalog(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        catalog: dict[str, str] = {}
        for entry in raw.get("current_state_aps", []):
            catalog[entry["name"]] = entry.get("description", "")
        for entry in raw.get("transition_aps", []):
            catalog[entry["name"]] = entry.get("description", "")
        return catalog
    except Exception:
        return {}

AP_CATALOG: dict[str, str] = _load_ap_catalog(_AP_CANDIDATES_FILE)

# ── tool execution (Python, deterministic) ────────────────────────────────────

def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT_CHARS else s[:MAX_OUTPUT_CHARS] + "\n...[truncated]"

def _run(argv: list[str]) -> str:
    try:
        p = subprocess.run(argv, cwd=str(WORK_DIR), capture_output=True,
                           text=True, timeout=CMD_TIMEOUT, check=False)
        merged = "\n".join(filter(None, [(p.stdout or "").strip(), (p.stderr or "").strip()]))
        return _clip(merged or "(no output)") + f"\n[exit {p.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[error] timed out after {CMD_TIMEOUT}s"
    except Exception as e:
        return f"[error] {e}"

def _has_shell_meta(s: str) -> bool:
    return any(c in s for c in ("|", ";", "&", ">", "<", "`", "$"))

def tool_git(command: str) -> str:
    if _has_shell_meta(command):
        return "[error] shell metacharacters (|;&><`$) not allowed in git_cmd"
    try:
        parts = shlex.split(command.strip())
    except ValueError as e:
        return f"[error] {e}"
    if not parts or parts[0] not in ALLOWED_GIT:
        return f"[error] '{parts[0] if parts else ''}' not allowed"
    return _run(["git"] + parts)

def tool_shell(command: str, args: list[str] | None = None) -> str:
    if _has_shell_meta(command):
        return "[error] shell metacharacters not allowed in command name — pass separate args list"
    binary = SHELL_BINS.get(command)
    if not binary:
        return f"[error] '{command}' not allowed. Allowed: {_SHELL_NAMES}"
    return _run([binary] + [str(a) for a in (args or [])])

TOOL_IMPL = {"git_cmd": tool_git, "shell_cmd": tool_shell}

TOOLS = [
    {"type": "function", "function": {
        "name": "git_cmd",
        "description": "Run a read-only git subcommand.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "shell_cmd",
        "description": f"Run one of: {_SHELL_NAMES}.",
        "parameters": {"type": "object",
                       "properties": {
                           "command": {"type": "string"},
                           "args": {"type": "array", "items": {"type": "string"}}},
                       "required": ["command"]}}},
]

# ── AP observability classification ──────────────────────────────────────────
# These APs describe organizational/remote policies that have no local git
# equivalent. They can never be determined from git commands alone, so they
# default to FALSE (closed-world assumption). Prompt 3B is only called for
# the remaining locally-observable APs.

UNOBSERVABLE_APS: frozenset[str] = frozenset({
    "A maintainer-level role.",
    "A non-fast-forward update approval token is present for this action.",
    "A protected-branch classification.",
    "A protected branch requires linear history.",
    "A queue-drain or operator-intervention state is required before execution resumes.",
    "A remote write-protection or permission-denied status is active.",
    "A required immediate force push was missed.",
    "A workflow state in which an immediate force push is required.",
    "An authorized/authenticated status allowing the guarded action.",
    "The action is executed under an approved protected-branch override policy.",
    "The actor is authenticated and authorized for force-push on the target branch.",
    "The current depth of pending git workflow operations in the processing queue exceeds 64.",
    "The number of retry attempts used by the workflow operation exceeds 3.",
    "The repository workflow state indicates that a force push is required before normal publishing can continue.",
    "Updates to this branch must pass through a reviewed integration workflow.",
    "Changes targeting main are integrated only through reviewed merge or rebase workflows.",
    "A network status indicating remote connectivity is available.",
    "Authentication credentials for remote write are valid.",
})

# These APs are structural invariants guaranteed by git's object model or
# desired liveness goals. They are always TRUE in a valid repository and
# should never be used to block normal operations.
ALWAYS_TRUE_APS: frozenset[str] = frozenset({
    "The commit graph remains acyclic and does not introduce reference cycles.",
    "The repository repeatedly returns to a clean synchronized state over time.",
    "The workflow repeatedly reaches a state with no unpublished local commit debt.",
    "Those detached commits are eventually anchored to a named branch reference.",
    "Any new work must be attached to a named branch before commit or push workflows continue.",
})

OBSERVABLE_APS  = [ap for ap in ALL_APS
                   if ap not in UNOBSERVABLE_APS and ap not in ALWAYS_TRUE_APS]

# ── LLM call logger ───────────────────────────────────────────────────────────

_LLM_LOG: list[dict] = []   # accumulated across the entire query; reset per query

def _llm_log_reset() -> None:
    _LLM_LOG.clear()

def _llm_log_snapshot() -> list[dict]:
    return list(_LLM_LOG)

# ── OpenAI client ─────────────────────────────────────────────────────────────

_CLIENT: openai.OpenAI | None = None

def _get_client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return _CLIENT

# ── LLM call helper ───────────────────────────────────────────────────────────

def _llm(messages: list[dict], model: str,
         tools: list | None = None, tag: str = "") -> tuple[str, list]:
    client = _get_client()
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    msg  = resp.choices[0].message
    content = (msg.content or "").strip()

    # Normalise tool_calls to the same dict shape the rest of the code expects
    raw_tcs = msg.tool_calls or []
    tool_calls = []
    for tc in raw_tcs:
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append({
            "id":       tc.id,
            "function": {"name": tc.function.name, "arguments": arguments},
        })

    _LLM_LOG.append({
        "tag":             tag,
        "messages_in":     messages,
        "content_out":     content,
        "tool_calls_out":  tool_calls,
    })
    return content, tool_calls

def _extract_json(text: str) -> Any:
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

# ── Prompt 6: guard — is this query in scope? ─────────────────────────────────

PROMPT_6_SYSTEM = f"""\
You are a scope guard for a git assistant agent.
The agent can only handle: git repository operations and these shell commands: {_SHELL_NAMES}.
Decide if the user's request is in scope.

Output ONLY valid JSON: {{"in_scope": true}} or {{"in_scope": false, "reason": "one sentence"}}"""

def prompt6_guard(query: str, model: str) -> tuple[bool, str]:
    content, _ = _llm([
        {"role": "system", "content": PROMPT_6_SYSTEM},
        {"role": "user",   "content": query},
    ], model, tag="6_guard")
    result = _extract_json(content)
    if isinstance(result, dict) and "in_scope" in result:
        return result["in_scope"], result.get("reason", "")
    return True, ""

# ── Prompt 3A: choose the command to observe one AP ──────────────────────────
# Input:  one AP string
# Output: one tool call (git_cmd or shell_cmd) — no text

PROMPT_3A_SYSTEM = f"""\
You are selecting a single read-only git command to gather evidence for one atomic proposition.
Working directory: {WORK_DIR}

Call exactly ONE tool — the minimal read-only command that best reveals whether the proposition is true.
Allowed git subcommands: status, log, branch, ls-files, remote, rev-parse, diff, show, stash, tag
Allowed shell commands: {_SHELL_NAMES}
No pipes, redirects, or write commands.

Do NOT write any text. Call the tool immediately."""

def prompt3a_choose_cmd(ap: str, model: str) -> tuple[str, str, str]:
    """
    Ask LLM to pick one read-only command for this AP.
    Returns (fn, key, output) — fn is tool name, key is repr, output is stdout.
    """
    content, tool_calls = _llm([
        {"role": "system", "content": PROMPT_3A_SYSTEM},
        {"role": "user",   "content": f"Atomic proposition: {ap}"},
    ], model, tools=TOOLS, tag="3A_choose_cmd")

    if tool_calls:
        tc   = tool_calls[0]
        fn   = tc.get("function", {}).get("name", "")
        args = tc.get("function", {}).get("arguments") or {}
        impl = TOOL_IMPL.get(fn)
        out  = impl(**args) if impl else f"[error] unknown tool {fn}"
        key  = f"{fn}({json.dumps(args)})"
        return fn, key, out

    # Model returned text — fall back to git status
    out = tool_git("status")
    return "git_cmd", 'git_cmd({"command": "status"})', out

# ── Prompt 3B: evaluate one AP from its command output ───────────────────────
# Input:  one AP string + stdout of the chosen command
# Output: {"value": true/false, "reason": "one sentence"}

PROMPT_3B_SYSTEM = """\
You are evaluating a single atomic proposition about a git repository.
You are given the stdout of one git/shell command chosen specifically for this proposition.
Decide TRUE or FALSE based solely on that output.

Output ONLY valid JSON:
{"value": true, "reason": "one sentence"}
{"value": false, "reason": "one sentence"}
If the output is insufficient, output {"value": false, "reason": "insufficient evidence"}."""

def prompt3b_eval_ap(ap: str, cmd_key: str, cmd_out: str, model: str) -> tuple[bool, str]:
    content, _ = _llm([
        {"role": "system", "content": PROMPT_3B_SYSTEM},
        {"role": "user",   "content":
            f"Atomic proposition: {ap}\n\n"
            f"Command output:\n$ {cmd_key}\n{cmd_out}"},
    ], model, tag="3B_eval_ap")
    result = _extract_json(content)
    if isinstance(result, dict) and "value" in result:
        return bool(result["value"]), result.get("reason", "")
    return False, "parse error"

def phase3_build_s0(model: str) -> dict[str, bool]:
    """
    Phase 3: for each observable AP, choose a command (3A), run it with
    output caching so identical commands are only executed once, then
    evaluate the AP from the output (3B).
    """
    n_obs    = len(OBSERVABLE_APS)
    n_unobs  = len(UNOBSERVABLE_APS & set(ALL_APS))
    n_always = len(ALWAYS_TRUE_APS & set(ALL_APS))
    print(f"  \033[36m[Phase 3] Evaluating {n_obs} observable APs "
          f"({n_unobs} unobservable→FALSE, {n_always} invariant→TRUE)...\033[0m")

    cmd_cache: dict[str, str] = {}   # key → output, avoids re-running identical commands

    s0: dict[str, bool] = {}
    for ap in ALL_APS:
        if ap in UNOBSERVABLE_APS:
            s0[ap] = False
            print(f"    ✗ (unobservable) {ap[:70]}")
        elif ap in ALWAYS_TRUE_APS:
            s0[ap] = True
            print(f"    ✓ (invariant)    {ap[:70]}")
        else:
            fn, key, out = prompt3a_choose_cmd(ap, model)
            if key in cmd_cache:
                out = cmd_cache[key]
                print(f"    \033[2m→ {key}  [cached]\033[0m")
            else:
                cmd_cache[key] = out
                print(f"    \033[2m→ {key}\033[0m")
            val, reason = prompt3b_eval_ap(ap, key, out, model)
            s0[ap] = val
            mark = "✓" if val else "✗"
            print(f"    {mark} {ap[:70]}  ({reason[:60]})")

    return s0

# ── Prompt 4A: propose next action ────────────────────────────────────────────

PROMPT_4A_SYSTEM = f"""\
You are proposing the next single git action toward a goal.
Working directory: {WORK_DIR}

Tool call signatures:
  git_cmd(command)         — subcommand + flags as one string, e.g. "add ." or "commit -m 'msg'"
                             Allowed: {", ".join(sorted(ALLOWED_GIT))}
  shell_cmd(command, args) — bare program name + args list, e.g. command="ls", args=["-la"]
                             Allowed programs: {_SHELL_NAMES}

Rules: NO pipes (|), redirects (>/<), semicolons, or backticks in any argument.

Output ONLY valid JSON:
{{
  "action_label": "short_snake_case_label",
  "tool": "git_cmd | shell_cmd | none",
  "args": {{"command": "..."}},
  "rationale": "one sentence"
}}
If the goal is already achieved, output {{"action_label": "goal_satisfied", "tool": "none", "args": {{}}, "rationale": "done"}}"""

def prompt4a_propose(goal: str, s_current: dict[str, bool],
                     trace: list[dict], tried: list[str], model: str) -> dict | None:
    done_steps = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        for i, s in enumerate(trace)
    ) or "  (none yet)"

    content, _ = _llm([
        {"role": "system", "content": PROMPT_4A_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Steps done so far:\n{done_steps}\n\n"
            f"Already tried at this step (rejected): {tried or 'none'}\n\n"
            f"What is the next action?"},
    ], model, tag="4A_propose")

    result = _extract_json(content)
    return result if isinstance(result, dict) else None

# ── Prompt 4B: predict state_after ────────────────────────────────────────────

# ── Prompt 4B: predict one AP's value after an action ────────────────────────
# Input:  one AP string + its current value + action label + command
# Output: {"value": true/false, "reason": "one sentence"}

PROMPT_4B_SYSTEM = """\
You are predicting the value of one atomic proposition after a git action completes.
You are given the proposition, its current value, and the exact git command being run.

Reason about what the command does and whether it would change this proposition.

Output ONLY valid JSON:
{"value": true, "reason": "one sentence"}
{"value": false, "reason": "one sentence"}"""

def prompt4b_predict_ap(ap: str, current_val: bool,
                        action_label: str, tool: str, args: dict,
                        model: str) -> tuple[bool, str]:
    content, _ = _llm([
        {"role": "system", "content": PROMPT_4B_SYSTEM},
        {"role": "user",   "content":
            f"Atomic proposition: {ap}\n"
            f"Current value: {'TRUE' if current_val else 'FALSE'}\n\n"
            f"Action: {action_label}\n"
            f"Command: {tool}({json.dumps(args)})"},
    ], model, tag="4B_predict_ap")
    result = _extract_json(content)
    if isinstance(result, dict) and "value" in result:
        return bool(result["value"]), result.get("reason", "")
    return current_val, "parse error — unchanged"

def prompt4b_predict(action_label: str, tool: str, args: dict,
                     s_current: dict[str, bool], model: str) -> dict[str, bool]:
    """
    Predict state_after by evaluating each observable state AP independently.
    Only APs currently TRUE are checked (they might flip to FALSE).
    Transition APs are set deterministically by Python from action_label.
    """
    s_after = dict(s_current)

    # State APs: only query ones currently TRUE (FALSE APs rarely flip on simple ops)
    for ap in OBSERVABLE_APS:
        if ap.startswith("(transition)"):
            continue   # handled below
        if s_current.get(ap, False):   # only TRUE APs need checking
            new_val, _ = prompt4b_predict_ap(ap, True, action_label, tool, args, model)
            s_after[ap] = new_val

    # Transition APs: set TRUE by matching action_label keywords to AP type.
    # Use explicit AP-type buckets to avoid substring false-positives
    # (e.g. "commits" in a rebase AP must not match a "commit" action).
    def _label_is(label: str, *words: str) -> bool:
        return any(w in label.split("_") for w in words)

    for ap in TRANS_APS:
        al = action_label.lower()
        apl = ap.lower()
        if "force" in apl and "push" in apl:
            s_after[ap] = _label_is(al, "force", "forcepush", "force_push")
        elif "rebase" in apl:
            s_after[ap] = _label_is(al, "rebase")
        elif "merge" in apl:
            s_after[ap] = _label_is(al, "merge")
        elif "push" in apl:
            s_after[ap] = _label_is(al, "push") and "force" not in al
        elif "direct commit" in apl or ("commit" in apl and "rebase" not in apl and "push" not in apl):
            s_after[ap] = _label_is(al, "commit", "stage", "add")
        elif "destructive" in apl or "rewrite" in apl:
            s_after[ap] = _label_is(al, "force", "rebase", "reset", "amend")
        elif "mutating" in apl:
            s_after[ap] = _label_is(al, "commit", "push", "merge", "rebase", "reset", "force")
        else:
            s_after[ap] = False

    s_after["last_action"] = action_label
    return s_after

# ── TLA+ spec + TLC (delegated to utils/tla_verifier) ────────────────────────

def _run_verification(s0: dict[str, bool],
                      trace: list[dict]) -> tuple[bool, str, str]:
    """Build TLA+ spec for s0 + trace and run TLC.

    Returns (passed, tla_spec, violations_summary).
    trace entries: {action_label, state_after: dict[str,bool]}
    """
    ap_trace     = [s0] + [step["state_after"] for step in trace]
    action_labels = [step["action_label"] for step in trace]
    safety_props  = [p for p in PROPERTIES if not has_liveness(p.get("ast", p))]

    tla_spec, cfg, _prop_names = build_tla_spec(
        ap_trace, ALL_APS, safety_props,
        action_labels=action_labels,
        module_name="GitTrace",
    )
    result = run_tlc(tla_spec, cfg, module_name="GitTrace", timeout=CMD_TIMEOUT)
    passed  = result["success"]
    summary = "; ".join(result.get("violations", [])) or result.get("reason", "")
    return passed, tla_spec, summary

# ── Phase 4: plan loop ────────────────────────────────────────────────────────

def phase4_plan(goal: str, s0: dict[str, bool],
                model: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (trace, planning_nodes).
    trace         — accepted candidate dicts {action_label, tool, args, state_before, state_after}
    planning_nodes — canonical planning-tree nodes in utils.session schema
    """
    trace: list[dict] = []
    s_current = dict(s0)
    tried_per_step: list[list[str]] = []
    planning_nodes: list[dict] = []

    print(f"\n\033[35m[Phase 4] Planning with TLC verification...\033[0m")

    goal_done = False   # set when LLM declares goal_satisfied; remaining steps are skipped

    for step_idx in range(MAX_PLAN_STEPS):
        while len(tried_per_step) <= step_idx:
            tried_per_step.append([])
        tried = tried_per_step[step_idx]

        if goal_done:
            print(f"  [step {step_idx+1}] skipped (goal already satisfied)")
            continue

        # Each attempt here represents one TLC-checked action; duplicate proposals
        # and parse errors do not count against the budget.
        attempt = 0
        proposal_misses = 0          # safety cap on consecutive non-TLC rounds
        MAX_PROPOSAL_MISSES = 6

        while attempt < MAX_RETRIES:
            if proposal_misses >= MAX_PROPOSAL_MISSES:
                print(f"  [step {step_idx+1}] Too many repeated/invalid proposals — stopping")
                break

            print(f"  [step {step_idx+1} attempt {attempt+1}] proposing action...")
            proposal = prompt4a_propose(goal, s_current, trace, tried, model)
            if proposal is None:
                print(f"    4A parse error, skipping")
                proposal_misses += 1
                continue

            action_label = proposal.get("action_label", "unknown")
            tool         = proposal.get("tool", "none")
            args         = proposal.get("args") or {}

            if action_label == "goal_satisfied":
                print(f"  [step {step_idx+1}] Goal satisfied — remaining steps will be skipped")
                goal_done = True
                break

            if action_label in tried:
                print(f"    {action_label} already tried, re-asking LLM")
                proposal_misses += 1
                continue

            proposal_misses = 0   # reset on a fresh action
            attempt += 1
            print(f"    → {action_label}: {tool}({json.dumps(args)})")

            # 4B: predict state_after
            s_after = prompt4b_predict(action_label, tool, args, s_current, model)

            # Build candidate trace entry
            candidate = {
                "action_label": action_label,
                "tool":         tool,
                "args":         args,
                "state_before": dict(s_current),
                "state_after":  s_after,
            }

            # TLA+ spec + TLC via utils/tla_verifier
            passed, tla_spec, violations_str = _run_verification(s0, trace + [candidate])
            violations_list = [v for v in violations_str.split(";") if v.strip()] if violations_str else []
            safety_props = [p for p in PROPERTIES if not has_liveness(p.get("ast", p))]

            verif = make_verification(
                passed=passed,
                properties_checked=[p.get("id", "?") for p in safety_props],
                violations=violations_list,
                tla_spec=tla_spec,
            )
            node = make_planning_node(
                node_id=len(planning_nodes),
                parent_node_id=len(planning_nodes) - 1 if planning_nodes else None,
                depth=step_idx,
                action_label=action_label,
                tool=tool,
                args=args,
                state_before=dict(s_current),
                state_after=s_after,
                verification=verif,
                result="accepted" if passed else "rejected",
            )
            planning_nodes.append(node)

            if passed:
                print(f"    \033[32mPASS\033[0m")
                trace.append(candidate)
                s_current = s_after
                break
            else:
                if violations_str:
                    print(f"    \033[31mFAIL: {violations_str[:120]}\033[0m")
                tried.append(action_label)
        else:
            print(f"  [step {step_idx+1}] Exhausted retries — continuing to next step")

    return trace, planning_nodes

# ── Phase 5: execute verified trace (Python) ──────────────────────────────────

def phase5_execute(trace: list[dict]) -> list[str]:
    results = []
    if not trace:
        return results

    print(f"\n\033[35m[Phase 5] Executing {len(trace)} verified step(s):\033[0m")
    for i, step in enumerate(trace, 1):
        tool = step.get("tool", "none")
        args = step.get("args") or {}
        print(f"\n  \033[33m[exec {i}/{len(trace)}] {step['action_label']}: {tool}({json.dumps(args)})\033[0m")

        if tool == "none" or not args:
            results.append("(no-op)")
            continue

        impl = TOOL_IMPL.get(tool)
        if impl is None:
            result = f"[error] unknown tool '{tool}'"
        else:
            try:
                result = impl(**args)
            except Exception as e:
                result = f"[error] {e}"

        print(f"  \033[2m{result}\033[0m")
        results.append(result)

    return results

# ── Prompt 5: success summary ─────────────────────────────────────────────────

PROMPT_5_SYSTEM = """\
You are summarising the result of a verified git workflow for the user.
Be concise (3-5 sentences). Cover: what was done, what changed, current repo state."""

def prompt5_summary(goal: str, trace: list[dict],
                    exec_results: list[str], model: str) -> str:
    trace_text = "\n".join(
        f"  {i+1}. {s['action_label']}: {s['tool']}({json.dumps(s['args'])})"
        f"\n     result: {exec_results[i-1][:200] if i <= len(exec_results) else 'N/A'}"
        for i, s in enumerate(trace, 1)
    )
    content, _ = _llm([
        {"role": "system", "content": PROMPT_5_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\nExecuted steps:\n{trace_text}"},
    ], model, tag="5_summary")
    return content

# ── Prompt 7: blocked explanation ─────────────────────────────────────────────

PROMPT_7_SYSTEM = """\
You are explaining to a user why their git request could not be safely executed.
The plan was blocked — either every candidate action violated a safety property,
or retries were exhausted.
Be direct and specific: name the property that blocked the plan, and suggest a safe alternative if one exists."""

def prompt7_blocked(goal: str, s0: dict[str, bool],
                    tried_actions: list[str], model: str) -> str:
    s0_text = "\n".join(f"  {'T' if v else 'F'}  {ap}" for ap, v in s0.items())
    content, _ = _llm([
        {"role": "system", "content": PROMPT_7_SYSTEM},
        {"role": "user",   "content":
            f"Goal: {goal}\n\n"
            f"Initial state:\n{s0_text}\n\n"
            f"Actions tried and rejected: {tried_actions}"},
    ], model, tag="7_blocked")
    return content

# ── top-level query handler ───────────────────────────────────────────────────

def handle_query(goal: str, model: str) -> tuple[str, dict]:
    """Run one query. Returns (response_text, session_turn_dict)."""
    _llm_log_reset()

    turn = make_session(
        agent="git-agent-fsm",
        model=model,
        domain="git",
        request=goal,
        work_dir=str(WORK_DIR),
        properties=PROPERTIES,
    )

    # Prompt 6: guard
    in_scope, reason = prompt6_guard(goal, model)
    if not in_scope:
        response = f"Out of scope: {reason}"
        turn["status"]        = "out_of_scope"
        turn["final_message"] = response
        turn["llm_log"]       = _llm_log_snapshot()
        return response, turn

    # Phase 3: observe s0
    print(f"\n\033[35m[Phase 3] Observing initial state s0...\033[0m")
    s0 = phase3_build_s0(model)
    turn["planning_tree"]["initial_state"] = s0

    # Phase 4: plan with TLC
    trace, planning_nodes = phase4_plan(goal, s0, model)
    turn["planning_tree"]["nodes"]         = planning_nodes
    turn["planning_tree"]["feasible"]      = bool(trace)
    turn["planning_tree"]["accepted_plan"] = [
        {"label": s["action_label"], "tool": s["tool"], "args": s["args"]}
        for s in trace
    ]

    # Phase 5: execute
    exec_results = phase5_execute(trace)
    turn["execution"]["steps"] = [
        make_execution_step(
            step_index=i,
            action_label=s["action_label"],
            tool=s["tool"],
            args=s["args"],
            result=exec_results[i] if i < len(exec_results) else "(no result)",
        )
        for i, s in enumerate(trace)
    ]

    # Response
    if trace:
        response = prompt5_summary(goal, trace, exec_results, model)
        turn["status"] = "finished"
    else:
        tried_all = [
            n["action"]["label"]
            for n in planning_nodes
            if n.get("result") == "rejected"
        ]
        response   = prompt7_blocked(goal, s0, tried_all, model)
        turn["status"] = "infeasible"

    turn["final_message"] = response
    turn["llm_log"]       = _llm_log_snapshot()
    return response, turn

# ── session save ──────────────────────────────────────────────────────────────

SESSIONS_DIR = WORK_DIR / ".git-agent-sessions"

def save_session(turns: list[dict], model: str) -> Path:
    session = make_session(
        agent="git-agent-fsm",
        model=model,
        domain="git",
        request="(multi-turn session)",
        work_dir=str(WORK_DIR),
        properties=PROPERTIES,
    )
    session["turns"] = turns   # embed per-turn dicts for multi-turn sessions
    return _save_session_util(session, SESSIONS_DIR, filename_prefix="session_fsm")

# ── REPL ──────────────────────────────────────────────────────────────────────

HELP_TEXT = """\
git-agent-fsm commands:
  /help          show this message
  /props         show property counts
  /model <name>  switch OpenAI model (current: {model})
  /cwd           show working directory
  /exit  /quit   exit

Each query runs: guard → Phase 3 (observe s0) → Phase 4 (plan+TLC) → Phase 5 (execute).
"""

def _is_git_repo() -> bool:
    r = subprocess.run(["git", "rev-parse", "--git-dir"],
                       cwd=str(WORK_DIR), capture_output=True, text=True)
    return r.returncode == 0

def repl() -> None:
    model   = DEFAULT_MODEL
    session_log: list[dict] = []

    in_repo     = _is_git_repo()
    repo_notice = "" if in_repo else "  \033[33m(not a git repo)\033[0m"
    sample_note = (f"sampled {len(PROPERTIES)}/{len(_ALL_PROPS)}"
                   if PROPERTY_SAMPLE_SIZE else f"all {len(PROPERTIES)}")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    print(f"\033[1mgit-agent-fsm\033[0m  model={model}  base_url={base_url}  cwd={WORK_DIR}{repo_notice}")
    print(f"Properties: {sample_note} | {len(STATE_APS)} state APs | {len(TRANS_APS)} transition APs")
    print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("\033[1mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            if session_log:
                path = save_session(session_log, model)
                print(f"\nSession saved → {path}")
            print("\nBye.")
            return

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            if session_log:
                path = save_session(session_log, model)
                print(f"Session saved → {path}")
            print("Bye.")
            return
        if user_input == "/help":
            print(HELP_TEXT.format(model=model))
            continue
        if user_input == "/props":
            print(f"{len(PROPERTIES)} properties | {len(STATE_APS)} state APs | {len(TRANS_APS)} transition APs\n")
            continue
        if user_input == "/cwd":
            print(f"{WORK_DIR}\n")
            continue
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            model = parts[1].strip() if len(parts) == 2 else model
            print(f"Model: {model}\n")
            continue
        if user_input.startswith("/"):
            print("Unknown command. Type /help.\n")
            continue

        try:
            answer, detail = handle_query(user_input, model)
        except openai.OpenAIError as e:
            answer  = f"[openai error] {e}"
            detail  = {"query": user_input, "response": answer, "llm_log": _llm_log_snapshot()}
        except Exception as e:
            import traceback; traceback.print_exc()
            answer  = f"[error] {e}"
            detail  = {"query": user_input, "response": answer, "llm_log": _llm_log_snapshot()}

        print(f"\n\033[1mAgent>\033[0m {answer}\n")
        session_log.append(detail)

if __name__ == "__main__":
    repl()
