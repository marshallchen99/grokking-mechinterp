#!/usr/bin/env python3
"""Summarise a sweep: which configurations grok, when, and with what mechanism.

Censoring is the thing to be careful about here.  A configuration that would
grok at step 60,000 and a configuration that never groks look identical under a
20,000-step budget, so every cell records `censored` and the budget alongside
the step, and no cell is ever described as "does not grok" without the budget
in the same breath.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.core import checkpoint_paths, load_snapshot    # noqa: E402
from grokking.analysis.spectra import key_freqs_consensus             # noqa: E402
from grokking.analysis.structure import additive_structure            # noqa: E402
from grokking.analysis.timing import crossing_step                    # noqa: E402
from grokking.runinfo import provenance, run_dataset                  # noqa: E402
from grokking.fourier import make_fourier_basis                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="S_p59_")
    ap.add_argument("--out", default="sweep_summary.json")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--no-mechanism", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = Path(args.root)
    files = sorted((root / "results").glob(f"{args.pattern}*_history.json"))
    if not files:
        raise SystemExit(f"no histories matching {args.pattern!r}")
    print(f"{len(files)} runs", flush=True)

    bases = {}
    cells = []
    for f in files:
        h = json.loads(f.read_text())
        tag = h["tag"]
        hist = h["history"]
        budget = h["train_cfg"]["steps"]
        grok = crossing_step(hist, "test_acc", 0.90)
        cell = {
            "tag": tag,
            "p": h["data"]["p"],
            "op": h["data"]["op"],
            "weight_decay": h["train_cfg"]["weight_decay"],
            "train_frac": h["data"]["train_frac"],
            "n_train": h["data"]["n_train"],
            "budget": budget,
            "grok_step": grok,
            "censored": grok is None,
            "final_test_acc": hist[-1]["test_acc"],
            "final_train_acc": hist[-1]["train_acc"],
            "max_test_acc": max(r["test_acc"] for r in hist),
            "memorisation_step": crossing_step(hist, "train_acc", 0.99),
            "final_weight_norm": hist[-1]["weight_norm"],
            "wall_clock_s": hist[-1]["elapsed"],
        }

        if not args.no_mechanism:
            paths = checkpoint_paths(root / "checkpoints" / tag)
            if paths:
                p = h["data"]["p"]
                if p not in bases:
                    bases[p] = make_fourier_basis(p, dtype=torch.float64)[0]
                F = bases[p]
                data = run_dataset(root, tag)
                snap = load_snapshot(paths[-1][1], data)
                cons = key_freqs_consensus(snap.model.W_E.detach(), snap.neuron_acts, F, p)
                K = cons["consensus"] if cons["agree"] else cons["union"]
                cell["n_key_freqs"] = len(K)
                cell["key_freqs"] = K
                cell["key_rules_agree"] = cons["agree"]
                cell["gini_W_E"] = cons["spectrum"].gini
                cell["frac_power_in_key"] = \
                    cons["spectrum"].sparsity_report()["frac_power_in_key_freqs"]
                cell["logit_var_a+b"] = additive_structure(snap.logits, p)["a+b"]
                cell["neuron_frac_above_85"] = cons["rule_b"]["frac_above"]

        cells.append(cell)
        flag = "censored" if cell["censored"] else f"grok at {cell['grok_step']:,.0f}"
        print(f"  {tag:28s} wd={cell['weight_decay']:<5} frac={cell['train_frac']:<5} "
              f"{flag:20s} final test acc {cell['final_test_acc']:.4f}"
              + (f"  gini {cell.get('gini_W_E', float('nan')):.3f}" if "gini_W_E" in cell else ""),
              flush=True)

    out = {"_provenance": provenance(root), "cells": cells,
           "n_censored": sum(c["censored"] for c in cells),
           "n_total": len(cells)}
    path = root / "results" / args.out
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}  ({out['n_censored']}/{out['n_total']} censored)", flush=True)


if __name__ == "__main__":
    main()
