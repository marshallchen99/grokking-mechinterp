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
    embedding_spectrum, neuron_frequencies,
)
from grokking.analysis.structure import (                            # noqa: E402
    additive_structure, trig_identity_report,
)
from grokking.data import make_dataset                               # noqa: E402
from grokking.fourier import make_fourier_basis                      # noqa: E402


def analyse_one(snap, F, key_freqs, p, with_neurons=True):
    spec = embedding_spectrum(snap.model.W_E.detach(), F, p)
    row = progress_measures(snap, F, key_freqs, emb_spectrum=spec)

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
    data = make_dataset(p=args.p, op=args.op, train_frac=args.train_frac,
                        seed=args.data_seed)
    F, _ = make_fourier_basis(args.p, dtype=torch.float64)

    paths = checkpoint_paths(root / "checkpoints" / args.tag)
    if not paths:
        raise SystemExit(f"no checkpoints found for tag {args.tag!r}")
    paths = paths[:: args.every]
    print(f"{len(paths)} checkpoints, steps {paths[0][0]} .. {paths[-1][0]}", flush=True)

    # Key frequencies are read off the FINAL model and then held fixed, so that
    # "restricted loss at step 500" means "how good is the circuit that will
    # eventually exist", not "how good is whatever happens to be strongest now".
    final = load_snapshot(paths[-1][1], data)
    final_spec = embedding_spectrum(final.model.W_E.detach(), F, args.p,
                                    method=args.key_method, k=args.key_k)
    key_freqs = final_spec.key_freqs
    print(f"key frequencies from final model ({args.key_method}): {key_freqs}", flush=True)
    print(f"  they carry {final_spec.sparsity_report()['frac_power_in_key_freqs']:.4f} "
          f"of embedding power; spectrum gini {final_spec.gini:.4f}", flush=True)
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
                  f"restr {rows[-1]['restricted_loss_block']:.4f} "
                  f"excl {rows[-1]['excluded_loss_block']:.4f} "
                  f"a+b {rows[-1]['logit_var_a+b']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            out_path.write_text(json.dumps(
                {"tag": args.tag, "op": args.op, "p": args.p,
                 "key_freqs": key_freqs, "key_method": args.key_method,
                 "final_emb_report": final_spec.sparsity_report(),
                 "rows": rows}))
    out_path.write_text(json.dumps(
        {"tag": args.tag, "op": args.op, "p": args.p,
         "key_freqs": key_freqs, "key_method": args.key_method,
         "final_emb_report": final_spec.sparsity_report(),
         "rows": rows}))
    print(f"wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
