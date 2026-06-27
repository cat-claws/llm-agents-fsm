"""Shared property-catalog helpers for agent FSMs."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Iterable, List

from utils.tla_verifier import load_properties


def select_properties(
    properties: Iterable[dict],
    *,
    active_ids: Iterable[str] | None = None,
    sample_size: int | None = None,
    rng=random,
) -> List[dict]:
    """Filter and optionally sample a property list."""
    selected = list(properties)
    if active_ids is not None:
        active = set(active_ids)
        selected = [prop for prop in selected if prop.get("id") in active]
    if sample_size is not None and sample_size < len(selected):
        selected = rng.sample(selected, sample_size)
    return selected


def load_property_catalog(
    path: str | Path,
    *,
    active_ids: Iterable[str] | None = None,
    sample_size: int | None = None,
    rng=random,
) -> List[dict]:
    """Load a property catalog and apply the common filter/sample options."""
    return select_properties(
        load_properties(path),
        active_ids=active_ids,
        sample_size=sample_size,
        rng=rng,
    )


def collect_aps_from_ast(node: Any, out: set[str]) -> None:
    """Collect AP names from an LTL AST into ``out``."""
    if isinstance(node, dict):
        if node.get("type") == "ap" and "name" in node:
            out.add(node["name"])
        for value in node.values():
            collect_aps_from_ast(value, out)
    elif isinstance(node, list):
        for value in node:
            collect_aps_from_ast(value, out)


def aps_from_properties(
    properties: Iterable[dict],
    *,
    transition_prefix: str = "(transition)",
) -> tuple[list[str], list[str]]:
    """Return ``(state_aps, transition_aps)`` used by a property set."""
    aps: set[str] = set()
    for prop in properties:
        collect_aps_from_ast(prop.get("ast", prop), aps)
    transition = sorted(ap for ap in aps if ap.startswith(transition_prefix))
    state = sorted(ap for ap in aps if not ap.startswith(transition_prefix))
    return state, transition


def observe_ap_values(
    ap_names: Iterable[str],
    observe_ap: Callable[[str], bool],
) -> dict[str, bool]:
    """Evaluate each AP by calling a domain-provided observer."""
    values: dict[str, bool] = {}
    for name in ap_names:
        values[name] = bool(observe_ap(name))
    return values
