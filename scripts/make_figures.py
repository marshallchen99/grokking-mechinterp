#!/usr/bin/env python3
"""Render every figure from the results files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.timing import phase_boundaries      # noqa: E402
from grokking.viz import save, use_style                   # noqa: E402
from grokking.viz.figures import (                         # noqa: E402
    grokking_curve, progress_panel, spectrum_bars,
)


def fig_grokking_curve(root: Path, tag: str, mode: str):
    hist = json.loads((root / "results" / f"{tag}_history.json").read_text())["history"]
    pb = phase_boundaries(hist)
    fig = grokking_curve(hist, mode=mode, grok_step=int(pb["grokking_step"]) if pb["grokking_step"] else None,
                         title="Grokking on (a + b) mod 113")
    return save(fig, "fig1_grokking_curve", root / "figures", mode=mode), pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_add_s0")
    ap.add_argument("--modes", nargs="+", default=["light"])
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    for mode in args.modes:
        use_style(mode)
        paths, pb = fig_grokking_curve(root, args.tag, mode)
        print(f"[{mode}] fig1 -> {[p.name for p in paths]}")
        print(f"        phases: {json.dumps({k: (round(v,1) if v else v) for k,v in pb.items()})}")

        apath = root / "results" / f"{args.tag}_analysis.json"
        if apath.exists():
            a = json.loads(apath.read_text())
            rows = a["rows"]
            fig = spectrum_bars(rows[-1]["emb_power_per_freq"], a["key_freqs"], mode=mode,
                                title=f"Embedding spectrum after training (step {rows[-1]['step']:,})")
            print(f"[{mode}] fig2 -> {[p.name for p in save(fig, 'fig2_embedding_spectrum', root/'figures', mode=mode)]}")
            fig = progress_panel(rows, mode=mode,
                                 grok_step=int(pb["grokking_step"]) if pb["grokking_step"] else None)
            print(f"[{mode}] fig3 -> {[p.name for p in save(fig, 'fig3_progress_measures', root/'figures', mode=mode)]}")


if __name__ == "__main__":
    main()
