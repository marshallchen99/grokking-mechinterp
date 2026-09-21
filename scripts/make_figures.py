#!/usr/bin/env python3
"""Render every figure from the results files.

Each figure is skipped, with a note, when its inputs are missing, so a
partially-finished experiment still produces the figures it can.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.timing import crossing_step, phase_boundaries   # noqa: E402
from grokking.report import is_finished                                # noqa: E402
from grokking.viz import save, use_style                               # noqa: E402
from grokking.viz.figures import (                                     # noqa: E402
    ablation_bars, grokking_curve, operation_panels, phase_diagram,
    progress_panel, spectrum_before_after,
)

OP_LABELS = {
    "add": "(a + b) mod p", "sub": "(a - b) mod p", "mul": "(a * b) mod p",
    "sq_sum": "(a^2 + b^2) mod p", "sq_sum_cross": "(a^2 + ab + b^2) mod p",
    "cube_add": "(a^3 + b) mod p", "cube_cross": "(a^3 + ab) mod p",
}


def load(root, name):
    p = root / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_add_s0")
    ap.add_argument("--op-tags", nargs="*", default=None)   # default: report_blocks.OP_TAGS
    ap.add_argument("--modes", nargs="+", default=["light"])
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    figs = root / "figures"
    from grokking.report_blocks import OP_TAGS
    args.op_tags = args.op_tags or OP_TAGS

    hist = load(root, f"{args.tag}_history.json")
    analysis = load(root, f"{args.tag}_analysis.json")
    mech = load(root, f"{args.tag}_mechanism.json")
    red = load(root, f"{args.tag}_redundancy.json")
    sweep = load(root, "sweep_summary.json")

    for mode in args.modes:
        use_style(mode)
        made, skipped = [], []

        # ---- 1. the grokking curve -------------------------------------
        if hist:
            pb = phase_boundaries(hist["history"])
            g = int(pb["grokking_step"]) if pb["grokking_step"] else None
            p_ = hist["data"]["p"]
            fig = grokking_curve(hist["history"], mode=mode, grok_step=g,
                                 title=f"Grokking on (a + b) mod {p_}")
            save(fig, "fig1_grokking_curve", figs, mode=mode); made.append("fig1")
        else:
            skipped.append("fig1 (no history)")

        # ---- 2. the spectrum, before and after ---------------------------
        if analysis and analysis.get("complete"):
            last = analysis["rows"][-1]["step"]
            picks = [s for s in (1000, 9000, 14500, last) if s <= last]
            fig = spectrum_before_after(analysis["rows"], analysis["key_freqs"],
                                        picks, mode=mode)
            save(fig, "fig2_embedding_spectrum", figs, mode=mode); made.append("fig2")
        else:
            skipped.append("fig2 (trajectory analysis incomplete)")

        # ---- 3. progress measures ---------------------------------------
        if analysis and analysis.get("complete") and hist:
            pb = phase_boundaries(hist["history"])
            g = int(pb["grokking_step"]) if pb["grokking_step"] else None
            mem = int(pb["memorisation_step"]) if pb["memorisation_step"] else 200
            fig = progress_panel(analysis["rows"], mode=mode, grok_step=g,
                                 phases={"memorisation": (0, mem),
                                         "circuit formation": (mem, int(g * 0.9) if g else 13000),
                                         "cleanup": (int(g * 0.9) if g else 13000,
                                                     int(g * 1.15) if g else 16500)})
            save(fig, "fig3_progress_measures", figs, mode=mode); made.append("fig3")
        else:
            skipped.append("fig3 (trajectory analysis incomplete)")

        # ---- 4. ablations -------------------------------------------------
        if mech:
            fa, K = mech["ablation_frequency"], mech["key_freqs"]["used"]
            ctrl = fa.get("_control_freqs", {}).get("freqs", [])
            entries = [
                {"label": "no edit (baseline)", "value": fa["baseline"]["test_acc"], "survives": True},
                {"label": f"keep only the key frequencies {K}",
                 "value": fa["keep_key_freqs"]["test_acc"], "survives": True}]
            if red and red.get("minimal_subsets"):
                mset = red["minimal_subsets"][0]
                v = next((r["test_acc"] for r in red["rows"] if r["subset"] == mset), None)
                if v is not None:
                    entries.append({"label": f"keep only the minimal subset {mset}",
                                    "value": v, "survives": True})
            entries += [
                {"label": f"delete {len(ctrl)} control frequencies {ctrl}",
                 "value": fa["drop_control_freqs"]["test_acc"], "survives": True},
                {"label": f"delete only the key frequencies {K}",
                 "value": fa["drop_key_freqs"]["test_acc"], "survives": False},
                {"label": f"keep only the control frequencies {ctrl}",
                 "value": fa["keep_control_freqs"]["test_acc"], "survives": False},
                {"label": "remove the MLP",
                 "value": mech["ablation_component"]["no_mlp"]["test_acc"], "survives": False},
                {"label": "remove attention",
                 "value": mech["ablation_component"]["no_attention"]["test_acc"], "survives": False}]
            fig = ablation_bars(entries, mode=mode, chance=1.0 / mech["p"],
                                title="Which edits the network survives")
            save(fig, "fig4_ablations", figs, mode=mode); made.append("fig4")
        else:
            skipped.append("fig4 (no mechanism file)")

        # ---- 5. operations ------------------------------------------------
        runs = []
        for tag in args.op_tags:
            h = load(root, f"{tag}_history.json")
            if not is_finished(h):
                continue            # partial histories render as finished runs
            label = OP_LABELS.get(h["data"]["op"], h["data"]["op"])
            runs.append({"label": f"{label}   p={h['data']['p']}",
                         "hist": h["history"],
                         "grok_step": crossing_step(h["history"], "test_acc", 0.90),
                         "budget": h["train_cfg"]["steps"]})
        if len(runs) >= 2:
            fig = operation_panels(runs, mode=mode, chance=1.0 / runs[0]["hist"][0].get("p", 113)
                                   if False else None)
            save(fig, "fig5_operations", figs, mode=mode); made.append("fig5")
        else:
            skipped.append(f"fig5 (only {len(runs)} operation runs finished)")

        # ---- 6. phase diagram ----------------------------------------------
        if sweep:
            fig = phase_diagram(sweep["cells"], "weight_decay", "train_frac",
                                "grok_step", mode=mode,
                                value_label="steps to 90% test accuracy",
                                title=f"When grokking happens (p = {sweep['cells'][0]['p']})")
            save(fig, "fig6_phase_diagram", figs, mode=mode); made.append("fig6")
        else:
            skipped.append("fig6 (sweep not finished)")

        print(f"[{mode}] made {made}")
        for s in skipped:
            print(f"[{mode}] skipped {s}")


if __name__ == "__main__":
    main()
