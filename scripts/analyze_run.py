#!/usr/bin/env python3
"""Walk a run's checkpoints and measure what is happening inside the model.

For every checkpoint this computes:

  * the ordinary train/test loss and accuracy (so the analysis file is
    self-contained and does not have to be joined against the training log);
  * the Fourier spectrum of the number embeddings, and how concentrated it is;
  * restricted and excluded loss, using the key frequencies of the FINAL model
    held fixed across the whole trajectory -- which is the point, since the
    question is when the final circuit starts to exist;
  * how much of the logits is a function of (a+b), against controls;
  * the split of each key frequency's energy between (a+b) and (a-b);
  * how concentrated each MLP neuron is on a single frequency.

Usage:
    python scripts/analyze_run.py --tag main_add_s0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.core import checkpoint_paths, load_snapshot   # noqa: E402
from grokking.analysis.progress import progress_measures             # noqa: E402
from grokking.analysis.spectra import (                              # noqa: E402
    embedding_spectrum, key_freqs_consensus, neuron_frequencies,
)
from grokking.analysis.structure import (                            # noqa: E402
    additive_structure, trig_identity_report,
)
from grokking.data import make_dataset                               # noqa: E402
from grokking.fourier import make_fourier_basis
from grokking.runinfo import run_config, run_dataset                      # noqa: E402


def analyse_one(snap, F, key_freqs, p, with_neurons=True):
    spec = embedding_spectrum(snap.model.W_E.detach(), F, p)
    row = progress_measures(snap, F, key_freqs, emb_spectrum=spec)

    # The measures above use the FINAL checkpoint's key frequencies -- right for
    # asking when the final circuit appears, wrong for forecasting, where it
    # smuggles in information from after the transition.  So the key-frequency-
    # dependent measures are also computed with the frequencies this checkpoint
    # itself would pick (the parameter-free gap rule on its own embedding).
    live = embedding_spectrum(snap.model.W_E.detach(), F, p, method="gap").key_freqs
    row["key_freqs_live"] = live
    if live:
        from grokking.analysis.progress import excluded_loss, restricted_loss
        row["restricted_loss_sum_all_live"] = restricted_loss(snap, F, live, mode="sum", split="all")["loss"]
        row["excluded_loss_sum_live"] = excluded_loss(snap, F, live, mode="sum", split="train")["loss"]
        pw = spec.power_per_freq[1:]
        row["emb_key_frac_live"] = float(sum(pw[k - 1] for k in live) / pw.sum())

    struct = additive_structure(snap.logits, p)
    for k, v in struct.items():
        row[f"logit_var_{k}"] = v

    trig = trig_identity_report(snap.logits, F, p, key_freqs)
    row["trig_sum_frac"] = trig["mean_sum_frac"]
    row["trig_readout_frac"] = trig["mean_readout_frac"]

    row["emb_key_freqs"] = spec.key_freqs
    row["emb_power_per_freq"] = [float(x) for x in spec.power_per_freq]

    if with_neurons:
        nf = neuron_frequencies(snap.neuron_acts, F, p)
        dom, frac = nf["dominant_freq"], nf["dominant_frac"]
        row["neuron_mean_dominant_frac"] = float(frac.mean())
        row["neuron_frac_above_50pct"] = float((frac > 0.5).float().mean())
        row["neuron_frac_above_85pct"] = float((frac > 0.85).float().mean())
        counts = torch.bincount(dom, minlength=(p - 1) // 2 + 1).tolist()
        row["neuron_freq_histogram"] = counts
        in_key = sum(counts[k] for k in key_freqs)
        row["neuron_frac_on_key_freqs"] = in_key / int(dom.numel())
    return row


def _write(out_path, args, key_freqs, cons, final, final_spec, rows, complete=False):
    """Write the analysis file.

    `complete` is False on the incremental writes made while the walk is still
    running.  Consumers must check it: a partially written trajectory looks
    perfectly well-formed and silently produces nonsense (a "lead time" of four
    steps rather than four thousand).
    """
    out_path.write_text(json.dumps({
        "complete": complete,
        "n_checkpoints_analysed": len(rows),
        "tag": args.tag, "op": args.op, "p": args.p,
        "key_freqs": key_freqs,
        "key_freq_rules": cons["by_rule"],
        "key_freq_jaccard": cons["jaccard"],
        "key_freq_rules_agree": cons["agree"],
        "final_checkpoint_step": final.step,
        "final_emb_report": final_spec.sparsity_report(),
        "neuron_report": cons["rule_b"],
        "rows": rows,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--op", default="add")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--key-method", default="gap", choices=["gap", "topk", "zscore"])
    ap.add_argument("--key-k", type=int, default=5)
    ap.add_argument("--no-neurons", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = Path(args.root)
    data = run_dataset(root, args.tag)   # the run's own split, never a default
    cfg_ = run_config(root, args.tag)
    args.p, args.op, args.train_frac = cfg_["p"], cfg_["op"], cfg_["train_frac"]
    F, _ = make_fourier_basis(args.p, dtype=torch.float64)

    paths = checkpoint_paths(root / "checkpoints" / args.tag)
    if not paths:
        raise SystemExit(f"no checkpoints found for tag {args.tag!r}")
    paths = paths[:: args.every]
    print(f"{len(paths)} checkpoints, steps {paths[0][0]} .. {paths[-1][0]}", flush=True)

    # Key frequencies are read off the FINAL checkpoint and then held fixed
    # across the whole trajectory.  That is the entire point of the progress
    # measures: the question is *when the final circuit starts to exist*, so
    # re-deriving the set per checkpoint would ask a different question at every
    # step -- and at early steps the spectrum is dense, so the rules just pick an
    # arbitrary frequency.
    #
    # Three independent rules are computed and their agreement is reported.  If
    # they disagree the write-up has to say so rather than quietly pick one.
    all_paths = checkpoint_paths(root / "checkpoints" / args.tag)
    final = load_snapshot(all_paths[-1][1], data)
    cons = key_freqs_consensus(final.model.W_E.detach(), final.neuron_acts, F, args.p)
    final_spec = cons["spectrum"]
    key_freqs = cons["consensus"] if cons["agree"] else cons["union"]
    print(f"key frequencies from the FINAL checkpoint (step {final.step}):", flush=True)
    for rule, v in cons["by_rule"].items():
        print(f"    {rule:24s} {v}", flush=True)
    print(f"    {'-> used':24s} {key_freqs}   Jaccard {cons['jaccard']:.3f} "
          f"agree={cons['agree']}", flush=True)
    if not cons["agree"]:
        print("    NOTE: rules disagree; the union is used and this must be "
              "stated in the write-up", flush=True)
    print(f"  key frequencies carry "
          f"{final_spec.sparsity_report()['frac_power_in_key_freqs']:.4f} of embedding "
          f"power; Gini(W_E) {final_spec.gini:.4f}; "
          f"{cons['rule_b']['frac_above']:.4f} of neurons above 0.85 explained", flush=True)
    if not key_freqs:
        print("  WARNING: no key frequencies found -- the model may not have "
              "developed a sparse representation", flush=True)

    rows, t0 = [], time.time()
    out_path = root / "results" / f"{args.tag}_analysis.json"
    for i, (step, path) in enumerate(paths):
        snap = load_snapshot(path, data)
        rows.append(analyse_one(snap, F, key_freqs, args.p,
                                with_neurons=not args.no_neurons))
        if i % 10 == 0 or i == len(paths) - 1:
            print(f"  [{i+1}/{len(paths)}] step {step:6d} "
                  f"test_acc {rows[-1]['test_acc']:.3f} "
                  f"restr {rows[-1]['restricted_loss_sum_all']:.2e} "
                  f"excl {rows[-1]['excluded_loss_sum']:.3f} "
                  f"a+b {rows[-1]['logit_var_a+b']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _write(out_path, args, key_freqs, cons, final, final_spec, rows)
    _write(out_path, args, key_freqs, cons, final, final_spec, rows, complete=True)
    print(f"wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
