"""Extract scalar metrics from lmms-eval ``simple_evaluate`` result dicts."""
from __future__ import annotations

from typing import Any


def extract_task_metric(
    results: dict[str, Any],
    task_name: str,
    metric_substring: str,
) -> float:
    """
    Pick a single float metric from lmms-eval style results.

    Typical layout: ``results["results"][task_name]`` is a dict whose keys look like
    ``"mmmu_acc,none"`` or ``"exact_match,flexible-extract"``. We match the first key
    that contains ``metric_substring`` and has a numeric value.
    """
    if not results:
        raise ValueError("Empty results dict.")
    block = results.get("results", {}).get(task_name)
    if block is None:
        raise KeyError(f"No results for task {task_name!r}. Top keys: {list(results.keys())}")
    if not isinstance(block, dict):
        raise TypeError(f"Task block for {task_name!r} must be dict, got {type(block)}")

    candidates: list[tuple[str, float]] = []
    for key, val in block.items():
        if metric_substring not in key:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            candidates.append((key, float(val)))
        elif isinstance(val, dict) and "acc" in val:  # rare nesting
            v = val.get("acc")
            if isinstance(v, (int, float)):
                candidates.append((key, float(v)))

    if not candidates:
        raise KeyError(
            f"No metric key containing {metric_substring!r} under task {task_name!r}. "
            f"Available keys: {list(block.keys())}"
        )
    # Prefer shorter / canonical-looking keys (e.g. exact match on substring at start)
    candidates.sort(key=lambda t: (len(t[0]), t[0]))
    return candidates[0][1]
