#!/usr/bin/env python3
"""Everything measured on a run's FINAL checkpoint, written to one results file.

Separate from `analyze_run.py`, which walks the whole trajectory: this is the
end-state mechanistic report -- what the finished model computes, and the
causal tests of whether it is really using it.

Headline numbers come from the final checkpoint and never from the transition,
because the representation keeps compressing for thousands of steps after the
accuracy moves.  The step they came from is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.ablation import (                       # noqa: E402
    ablate_neurons, component_ablation_study, evaluate_model,
    frequency_ablation_study,
)
from grokking.analysis.core import checkpoint_paths, load_snapshot   # noqa: E402
from grokking.analysis.progress import (                       # noqa: E402
    excluded_loss, per_frequency_excluded_loss, restricted_loss,
)
from grokking.analysis.spectra import key_freqs_consensus      # noqa: E402
from grokking.analysis.structure import (                      # noqa: E402
    additive_structure, readout_budget, trig_identity_report,
)
from grokking.data import make_dataset                         # noqa: E402
from grokking.fourier import make_fourier_basis                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--op", default="add")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dlog", action="store_true",
                    help="also analyse in the discrete-log (multiplicative) basis")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = Path(args.root)
    p = args.p
    data = make_dataset(p=p, op=args.op, train_frac=args.train_frac, seed=args.data_seed)
    F, _ = make_fourier_basis(p, dtype=torch.float64)

    paths = checkpoint_paths(root / "checkpoints" / args.tag)
    if not paths:
        raise SystemExit(f"no checkpoints for {args.tag!r}")
    snap = load_snapshot(paths[-1][1], data)
    print(f"{args.tag}: final checkpoint step {snap.step}", flush=True)

    out = {"tag": args.tag, "op": args.op, "p": p, "final_step": snap.step,
           "baseline": snap.losses()}

    # ---- which frequencies, by three independent rules ---------------------
    cons = key_freqs_consensus(snap.model.W_E.detach(), snap.neuron_acts, F, p)
    K = cons["consensus"] if cons["agree"] else cons["union"]
    spec = cons["spectrum"]
    out["key_freqs"] = {
        "by_rule": cons["by_rule"], "used": K,
        "jaccard": cons["jaccard"], "agree": cons["agree"],
        "gini_W_E": spec.gini,
        "frac_power_in_key": spec.sparsity_report()["frac_power_in_key_freqs"],
        "power_per_freq": [float(x) for x in spec.power_per_freq],
    }
    print(f"  key frequencies {K}  (Jaccard {cons['jaccard']:.3f})", flush=True)

    # ---- neuron census -----------------------------------------------------
    rb = cons["rule_b"]
    dom = cons["rule_b"]  # counts recomputed below from the full rule-b output
    from grokking.analysis.spectra import key_freqs_rule_b
    full_b = key_freqs_rule_b(snap.neuron_acts, F, p)
    domf, domfrac = full_b["dominant_freq"], full_b["dominant_frac"]
    counts = {int(k): int((domf == k).sum()) for k in sorted(set(domf.tolist()))}
    out["neurons"] = {
        "counts_by_freq": counts,
        "n_total": int(domf.numel()),
        "frac_above_85": float((domfrac > 0.85).float().mean()),
        "mean_dominant_frac": float(domfrac.mean()),
        "min_dominant_frac": float(domfrac.min()),
        "all_on_key_freqs": set(counts) <= set(K),
    }

    # ---- what does it compute ---------------------------------------------
    out["structure"] = additive_structure(snap.logits, p)
    trig = trig_identity_report(snap.logits, F, p, K)
    out["trig"] = {
        "mean_sum_frac": trig["mean_sum_frac"],
        "mean_readout_frac": trig["mean_readout_frac"],
        "per_freq": {str(k): {"sum_frac": v["sum_frac"],
                              "readout_frac": v["readout_frac"]}
                     for k, v in trig["per_freq"].items()},
    }
    # The readout is read in whichever direction this model actually uses;
    # reading it in the wrong one returns noise, which is why both are recorded.
    out["readout_budget"] = {
        "sum": readout_budget(snap.logits, F, p, K, sign=+1),
        "diff": readout_budget(snap.logits, F, p, K, sign=-1),
    }
    best = max(("sum", "diff"), key=lambda kk: out["readout_budget"][kk]["own"])
    out["readout_budget"]["direction_used"] = best
    print(f"  readout budget ({best}): own {out['readout_budget'][best]['own']:.4f} "
          f"dc {out['readout_budget'][best]['dc']:.4f} "
          f"cross {out['readout_budget'][best]['cross']:.4f}", flush=True)
    print(f"  (a+b) variance {out['structure']['a+b']:.4f}; "
          f"trig sum-fraction {trig['mean_sum_frac']:.4f}", flush=True)

    # ---- progress measures at the end -------------------------------------
    prog = {}
    for mode in ("sum", "block"):
        for split in ("all", "train"):
            r = restricted_loss(snap, F, K, mode=mode, split=split)
            prog[f"restricted_{mode}_{split}"] = r
        prog[f"excluded_{mode}_train"] = excluded_loss(snap, F, K, mode=mode, split="train")
    out["progress"] = prog
    out["per_frequency_excluded_loss"] = per_frequency_excluded_loss(snap, F)

    # ---- causal ------------------------------------------------------------
    print("  running ablations...", flush=True)
    out["ablation_frequency"] = frequency_ablation_study(snap.model, data, F, K)
    out["ablation_component"] = component_ablation_study(snap.model, data)

    mean_act = snap.neuron_acts.reshape(-1, snap.neuron_acts.shape[-1]).mean(0)
    per_freq_neuron = {}
    for k in K:
        per_freq_neuron[f"drop_freq_{k}"] = evaluate_model(
            ablate_neurons(snap.model, domf != k, mean_act), data)
        per_freq_neuron[f"keep_only_freq_{k}"] = evaluate_model(
            ablate_neurons(snap.model, domf == k, mean_act), data)
    out["ablation_neuron_per_freq"] = per_freq_neuron

    # ---- optional: the multiplicative basis --------------------------------
    if args.dlog:
        from grokking.analysis.dlog import dlog_view, star_embedding, zero_element_report
        print("  analysing in the discrete-log basis...", flush=True)
        sv = dlog_view(snap)
        n = sv.p
        Fs, _ = make_fourier_basis(n, dtype=torch.float64)
        cons_s = key_freqs_consensus(star_embedding(snap.model.W_E.detach(), p),
                                     sv.neuron_acts, Fs, n)
        Ks = cons_s["consensus"] if cons_s["agree"] else cons_s["union"]
        star = {
            "n": n, "primitive_root": sv.g,
            "key_freqs": {"by_rule": cons_s["by_rule"], "used": Ks,
                          "jaccard": cons_s["jaccard"], "agree": cons_s["agree"],
                          "gini_W_E": cons_s["spectrum"].gini,
                          "frac_power_in_key":
                              cons_s["spectrum"].sparsity_report()["frac_power_in_key_freqs"]},
            "baseline": sv.losses(),
            "structure": additive_structure(sv.logits, n),
        }
        if Ks:
            star["progress"] = {
                "restricted_sum_all": restricted_loss(sv, Fs, Ks, mode="sum", split="all"),
                "excluded_sum_train": excluded_loss(sv, Fs, Ks, mode="sum", split="train"),
            }
            tr = trig_identity_report(sv.logits, Fs, n, Ks)
            star["trig"] = {"mean_sum_frac": tr["mean_sum_frac"],
                            "mean_readout_frac": tr["mean_readout_frac"]}
        star["zero_element"] = zero_element_report(snap)
        out["dlog"] = star

    path = root / "results" / f"{args.tag}_mechanism.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
