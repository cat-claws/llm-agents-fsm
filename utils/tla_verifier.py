"""Shared TLA+/TLC verification utilities for llm-agents-fsm.

The strict cross-domain verification path is:

  initial_state : {ap_name: bool}
  action_labels : [transition/action label, ...]
  states_after  : [{ap_name: bool}, ...]

verify_fsm_trace() combines these into an AP trace, writes the action-labelled
FSM into TLA+ with the active properties, then runs TLC.  AP prediction and AP
observation stay domain-specific; TLA generation and TLC execution live here.

build_tla_spec() also retains the older plain-state trace mode for compatibility.
In both modes every AP name is slugified to a valid TLA+ identifier, so long
natural-language AP strings are handled cleanly.

Property format (shared, SHRDLU-originated):
  {
    "id":               "prop.git.01",
    "natural_language": "...",
    "ltl":              "G(...)",
    "ast": { "type": "globally", "operand": ... }
  }

run_tlc() returns a structured result dict; callers check result["success"].
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "build_tla_spec",
    "build_tla_cfg",
    "run_tlc",
    "verify_fsm_trace",
    "verify_ap_trace",
    "load_properties",
    "has_liveness",
]

_JAR_SEARCH = [
    "/usr/local/lib/tla2tools.jar",
    "/usr/lib/tla2tools.jar",
    str(Path.home() / "tla2tools.jar"),
    str(Path(__file__).resolve().parent.parent / "tla2tools.jar"),
    "/common/home/users/r/rhzhang/tools/tla/tla2tools.jar",
]


def _find_jar() -> Optional[str]:
    env = os.environ.get("TLA2TOOLS_JAR")
    if env and os.path.isfile(env):
        return env
    for p in _JAR_SEARCH:
        if os.path.isfile(p):
            return p
    return None


def _find_java() -> str:
    env = os.environ.get("JAVA_BIN")
    if env:
        return env
    legacy = "/common/home/users/r/rhzhang/.conda/envs/rh1/bin/java"
    if os.path.isfile(legacy):
        return legacy
    return "java"

def _ap_slug(name: str, index: int) -> str:
    """Stable short TLA+ variable name: AP_<index>_<sanitised_prefix>."""
    sanitised = re.sub(r"[^a-zA-Z0-9]", "_", name)[:30].strip("_")
    return f"AP_{index:03d}_{sanitised}"


def _make_slug_map(ap_names: List[str]) -> Dict[str, str]:
    """Return {ap_name: tla_var} for every AP in the list."""
    return {name: _ap_slug(name, i) for i, name in enumerate(ap_names)}

def has_liveness(node: Dict) -> bool:
    """Return True if the LTL AST contains any finally/until operator."""
    t = node.get("type", "")
    if t in ("finally", "until", "eventually"):
        return True
    if t in ("globally", "not", "next"):
        return has_liveness(node.get("operand", {}))
    if t in ("and", "or"):
        return any(has_liveness(a) for a in node.get("args", []))
    if t == "implies":
        return has_liveness(node.get("left", {})) or has_liveness(node.get("right", {}))
    return False

def _ast_to_tla(node: Dict, slug_map: Dict[str, str], ap_slugs: List[str]) -> str:
    """Recursively translate an LTL AST node to a TLA+ formula string.

    G(A => X(P)) is encoded as [][A => P']_vars (the TLC-safe encoding for
    action-next properties) when the outer node type is 'globally' and the
    inner right child is 'next'.  All other 'next' operators outside that
    specific shape are approximated as G (safe over-approximation on finite traces).
    """
    t = node.get("type", "")

    if t == "ap":
        name = node["name"]
        var = slug_map.get(name)
        if var is None:
            # AP not in current property set — treat as TRUE (open-world safe default)
            return "TRUE"
        return var

    if t == "not":
        return "~(%s)" % _ast_to_tla(node["operand"], slug_map, ap_slugs)

    if t == "and":
        parts = [_ast_to_tla(a, slug_map, ap_slugs) for a in node["args"]]
        return "(%s)" % " /\\ ".join(parts)

    if t == "or":
        parts = [_ast_to_tla(a, slug_map, ap_slugs) for a in node["args"]]
        return "(%s)" % " \\/ ".join(parts)

    if t == "implies":
        l = _ast_to_tla(node["left"],  slug_map, ap_slugs)
        r = _ast_to_tla(node["right"], slug_map, ap_slugs)
        return "(%s => %s)" % (l, r)

    if t == "globally":
        inner = node["operand"]
        if (inner.get("type") == "implies"
                and inner["right"].get("type") == "next"):
            antecedent = _ast_to_tla(inner["left"], slug_map, ap_slugs)
            consequent = _ast_to_tla(inner["right"]["operand"], slug_map, ap_slugs)
            return "[][((%s) => %s')]_<<%s>>" % (
                antecedent, consequent, ", ".join(ap_slugs)
            )
        return "[](%s)" % _ast_to_tla(inner, slug_map, ap_slugs)

    if t in ("finally", "eventually"):
        return "<>(%s)" % _ast_to_tla(node["operand"], slug_map, ap_slugs)

    if t == "next":
        return "[](%s)" % _ast_to_tla(node["operand"], slug_map, ap_slugs)

    if t == "until":
        l = _ast_to_tla(node["left"],  slug_map, ap_slugs)
        r = _ast_to_tla(node["right"], slug_map, ap_slugs)
        return "(%s ~> %s)" % (l, r)

    return "TRUE"

def build_tla_spec(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: List[Dict],
    *,
    action_labels: Optional[List[str]] = None,
    module_name: str = "AgentTrace",
) -> tuple[str, str, List[str]]:
    """Return (tla_module_string, cfg_string, prop_names) for the given trace.

    ap_trace        list of {ap_name: bool}, length N+1 (s0 … sN) for
                    action-labelled mode, or length N for plain-state mode
                    (where s0 is treated as both init and step-0 state).
    ap_names        ordered list of all AP name strings used as variable names
    properties      list of property dicts with 'id', 'natural_language', 'ast'
    action_labels   if provided, list of N action label strings (len == len(ap_trace)-1)
                    triggering action-labelled TLA+ spec style; otherwise plain-state style
    """
    if not ap_trace:
        raise ValueError("ap_trace must be non-empty")

    slug_map  = _make_slug_map(ap_names)
    ap_slugs  = [slug_map[n] for n in ap_names]

    def bool_tla(v: bool) -> str:
        return "TRUE" if v else "FALSE"

    def state_conj(state: Dict[str, bool], prime: bool = False) -> List[str]:
        p = "'" if prime else ""
        return [
            "     /\\ %s%s = %s" % (slug_map[n], p, bool_tla(state.get(n, False)))
            for n in ap_names
        ]

    lines: List[str] = []

    if action_labels is not None:
        n = len(action_labels)
        assert len(ap_trace) == n + 1, \
            "action_labels length must be len(ap_trace)-1"

        all_action_labels = sorted({"none"} | set(action_labels))
        action_set = "{" + ", ".join('"%s"' % a for a in all_action_labels) + "}"
        all_vars = ap_slugs + ["last_action", "step"]

        lines += [
            "---- MODULE %s ----" % module_name,
            "EXTENDS TLC, Sequences, Naturals",
            "",
            "VARIABLES %s" % ", ".join(all_vars),
            "",
            "AllActions == %s" % action_set,
            "",
            "TypeOK ==",
        ]
        for v in ap_slugs:
            lines.append("  /\\ %s \\in {TRUE, FALSE}" % v)
        lines += [
            "  /\\ last_action \\in AllActions",
            "  /\\ step \\in 1..%d" % (n + 1),
            "",
            "Init ==",
        ]
        lines += state_conj(ap_trace[0])
        lines += [
            '     /\\ last_action = "none"',
            "     /\\ step = 1",
            "",
            "Next ==",
        ]
        for k, (label, s_after) in enumerate(zip(action_labels, ap_trace[1:]), 1):
            lines.append("  \\/ /\\ step = %d" % k)
            lines += state_conj(s_after, prime=True)
            lines += [
                '     /\\ last_action\' = "%s"' % label,
                "     /\\ step' = %d" % (k + 1),
            ]
        lines += [
            "  \\/ /\\ step = %d" % (n + 1),
            "     /\\ UNCHANGED <<%s>>" % ", ".join(all_vars),
            "",
            "Spec == Init /\\ [][Next]_<<%s>>" % ", ".join(all_vars),
            "",
        ]

    else:
        n = len(ap_trace)
        all_vars_str = "<<step, %s>>" % ", ".join(ap_slugs)

        lines += [
            "---- MODULE %s ----" % module_name,
            "EXTENDS Naturals",
            "",
            "VARIABLES %s, step" % ", ".join(ap_slugs),
            "",
        ]
        for i, state in enumerate(ap_trace):
            conj = " /\\ ".join(
                "%s = %s" % (slug_map[n2], bool_tla(state.get(n2, False)))
                for n2 in ap_names
            )
            lines.append("State_%d == %s" % (i, conj))
        lines += [
            "",
            "Init ==",
            "  /\\ State_0",
            "  /\\ step = 0",
            "",
            "Next ==",
        ]
        for i in range(1, n):
            conj_p = " /\\ ".join(
                "%s' = %s" % (slug_map[n2], bool_tla(ap_trace[i].get(n2, False)))
                for n2 in ap_names
            )
            lines += [
                "  \\/ /\\ step = %d" % (i - 1),
                "     /\\ step' = %d" % i,
                "     /\\ %s" % conj_p,
            ]
        lines += [
            "  \\/ /\\ step = %d" % (n - 1),
            "     /\\ step' = step",
        ]
        for ap in ap_slugs:
            lines.append("     /\\ %s' = %s" % (ap, ap))
        lines += [
            "",
            "Spec == Init /\\ [][Next]_%s" % all_vars_str,
            "",
        ]

    prop_names: List[str] = []
    for i, prop in enumerate(properties, 1):
        pid  = prop.get("id", "prop_%02d" % i)
        tla_id = "Property_%s" % re.sub(r"[^a-zA-Z0-9]", "_", pid)
        nl   = prop.get("natural_language", "")
        ast  = prop.get("ast", prop)
        if has_liveness(ast):
            lines.append("(* %s [liveness — skipped on finite trace] *)" % pid)
            lines.append("%s == TRUE" % tla_id)
        else:
            formula = _ast_to_tla(ast, slug_map, ap_slugs)
            lines.append("(* %s: %s *)" % (pid, nl))
            lines.append("%s == %s" % (tla_id, formula))
            prop_names.append(tla_id)
        lines.append("")

    lines.append("=" * 20)
    tla_str = "\n".join(lines)
    has_type_ok = action_labels is not None
    cfg_str = build_tla_cfg(prop_names, has_type_ok=has_type_ok)
    return tla_str, cfg_str, prop_names


def build_tla_cfg(prop_names: List[str], *, has_type_ok: bool = False) -> str:
    """Return a TLC .cfg string for the given list of property TLA+ identifiers."""
    lines = ["SPECIFICATION Spec"]
    if has_type_ok:
        lines.append("INVARIANT TypeOK")
    for pn in prop_names:
        lines.append("PROPERTY %s" % pn)
    return "\n".join(lines)

def run_tlc(
    tla_spec: str,
    cfg: str,
    *,
    module_name: str = "AgentTrace",
    timeout: int = 60,
) -> Dict:
    """Run TLC on tla_spec + cfg strings.  Returns a structured result dict.

    Result keys:
      success    bool       True iff TLC found no errors
      skipped    bool       True iff TLC was not run (jar/java missing)
      reason     str        set when skipped=True
      returncode int
      stdout     str
      stderr     str
      violations list[str]
    """
    jar  = _find_jar()
    java = _find_java()

    if jar is None:
        return {
            "success":    False,
            "skipped":    True,
            "reason":     "tla2tools.jar not found; set TLA2TOOLS_JAR env var or place it at %s"
                          % _JAR_SEARCH[-1],
            "violations": [],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tla_path = os.path.join(tmpdir, "%s.tla" % module_name)
        cfg_path = os.path.join(tmpdir, "%s.cfg" % module_name)
        with open(tla_path, "w") as f:
            f.write(tla_spec)
        with open(cfg_path, "w") as f:
            f.write(cfg)

        try:
            proc = subprocess.run(
                [java, "-jar", jar, "-config", cfg_path, tla_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return {
                "success":    False,
                "skipped":    True,
                "reason":     "java not found at '%s'" % java,
                "violations": [],
            }
        except subprocess.TimeoutExpired:
            return {
                "success":    False,
                "skipped":    False,
                "reason":     "TLC timed out after %ds" % timeout,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "violations": [],
            }

        stdout = proc.stdout
        stderr = proc.stderr
        passed = "No error has been found" in stdout
        violations = [
            line.strip()
            for line in stdout.splitlines()
            if line.strip().startswith("Error") or "violated" in line.lower()
        ]
        return {
            "success":     passed,
            "skipped":     False,
            "returncode":  proc.returncode,
            "stdout":      stdout,
            "stderr":      stderr,
            "violations":  violations,
        }


def verify_ap_trace(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: List[Dict],
    *,
    action_labels: Optional[List[str]] = None,
    module_name: str = "AgentTrace",
    timeout: int = 60,
    is_complete_trace: bool = True,
) -> Dict:
    """Build TLA+ spec, run TLC, return combined result dict.

    When is_complete_trace=False, liveness properties are additionally filtered
    out (they can never be satisfied by a partial trace that stutters at the end).
    """
    active = properties
    if not is_complete_trace:
        active = [p for p in properties if not has_liveness(p.get("ast", p))]

    tla_spec, cfg, prop_names = build_tla_spec(
        ap_trace, ap_names, active,
        action_labels=action_labels,
        module_name=module_name,
    )
    tlc_result = run_tlc(tla_spec, cfg, module_name=module_name, timeout=timeout)
    return {
        "tla_spec":           tla_spec,
        "tla_cfg":            cfg,
        "tlc_result":         tlc_result,
        "trace_length":       len(ap_trace),
        "properties_checked": [p.get("id", "?") for p in active],
    }


def verify_fsm_trace(
    initial_state: Dict[str, bool],
    action_labels: List[str],
    states_after: List[Dict[str, bool]],
    ap_names: List[str],
    properties: List[Dict],
    *,
    module_name: str = "AgentTrace",
    timeout: int = 60,
    is_complete_trace: bool = True,
) -> Dict:
    """Verify one explicit FSM trace using the shared action-labelled TLA path.

    The caller is responsible for domain-specific AP prediction/observation.
    This function is deliberately domain-neutral: it only assembles
    ``s0 --action--> s1 ...`` into TLA+ and runs TLC.
    """
    if len(action_labels) != len(states_after):
        raise ValueError("action_labels length must equal states_after length")

    active = properties
    if not is_complete_trace:
        active = [p for p in properties if not has_liveness(p.get("ast", p))]

    ap_trace = [initial_state] + states_after
    tla_spec, cfg, prop_names = build_tla_spec(
        ap_trace,
        ap_names,
        active,
        action_labels=action_labels,
        module_name=module_name,
    )
    tlc_result = run_tlc(tla_spec, cfg, module_name=module_name, timeout=timeout)
    passed = bool(tlc_result.get("success") or tlc_result.get("skipped"))
    return {
        "passed":             passed,
        "tla_spec":           tla_spec,
        "tla_cfg":            cfg,
        "tlc_result":         tlc_result,
        "trace_length":       len(ap_trace),
        "transition_count":   len(action_labels),
        "action_labels":      list(action_labels),
        "properties_checked": [p.get("id", "?") for p in active],
        "tla_properties":     prop_names,
    }


def load_properties(path: str | Path) -> List[Dict]:
    """Load a properties JSON file.

    Accepts both formats:
      - SHRDLU catalog: {"properties": [ {id, natural_language, ltl, ast}, ... ]}
      - bare list:      [ {id, natural_language, ltl, ast}, ... ]
      - legacy bare AST (git property_NN.json): wraps each file with synthetic id/nl
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "properties" in data:
        return data["properties"]
    if isinstance(data, list):
        return data
    stem = Path(path).stem
    return [{
        "id":               stem,
        "natural_language": stem,
        "ltl":              "",
        "ast":              data,
    }]
