"""Locating the transition on a trajectory."""

from __future__ import annotations

from typing import Dict, List, Optional


def crossing_step(rows: List[Dict], key: str, threshold: float,
                  rising: bool = True) -> Optional[float]:
    """First step at which `key` crosses `threshold`, linearly interpolated.

    Returns None if it never crosses, which is a real outcome worth recording
    (some configurations simply do not grok within the budget) rather than
    something to paper over with a fallback value.
    """
    prev = None
    for r in rows:
        v, s = r.get(key), r["step"]
        if v is None:
            continue
        hit = v >= threshold if rising else v <= threshold
        if hit:
            if prev is None:
                return float(s)
            (ps, pv) = prev
            if v == pv:
                return float(s)
            frac = (threshold - pv) / (v - pv)
            return float(ps + frac * (s - ps))
        prev = (s, v)
    return None


def phase_boundaries(rows: List[Dict], memorise_acc: float = 0.99,
                     generalise_acc: float = 0.9) -> Dict[str, Optional[float]]:
    """The three-phase summary of a run.

    memorisation_step   train accuracy first reaches `memorise_acc`
    grokking_step       test accuracy first reaches `generalise_acc`
    plateau_length      the gap between them, i.e. how long the run spends
                        looking like a failed, overfitted model
    """
    mem = crossing_step(rows, "train_acc", memorise_acc)
    grok = crossing_step(rows, "test_acc", generalise_acc)
    out = {"memorisation_step": mem, "grokking_step": grok,
           "plateau_length": (grok - mem) if (mem is not None and grok is not None) else None}
    for extra, thr in (("restricted_loss_block", 1.0), ("logit_var_a+b", 0.5)):
        if rows and extra in rows[0]:
            out[f"{extra}_crossing"] = crossing_step(
                rows, extra, thr, rising=(extra != "restricted_loss_block"))
    return out
