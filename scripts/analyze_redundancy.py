#!/usr/bin/env python3
"""Which of the learned frequencies are actually needed?

The model settles on a small set of frequencies, but that does not mean it
needs all of them.  This keeps every subset of the key set in the embedding,
deletes everything else, and re-runs the network -- so it reports not just
"these frequencies matter" but which combinations suffice, and whether the set
the model found is minimal.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.ablation import (                      # noqa: E402
    ablate_embedding_frequencies, evaluate_model,
)
from grokking.analysis.core import checkpoint_paths, load_snapshot  # noqa: E402
from grokking.analysis.spectra import key_freqs_consensus     # noqa: E402
from grokking.data import make_dataset
from grokking.runinfo import run_config, run_dataset  # noqa: E402                        # noqa: E402
from grokking.fourier import make_fourier_basis               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--op", default=None)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--max-subsets", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = Path(args.root)
    # The run's own split, read from its record -- never a command-line default.
    cfg_ = run_config(root, args.tag)
    for _k, _v in (("p", args.p), ("op", args.op), ("train_frac", args.train_frac)):
        if _v is not None and _v != cfg_[_k]:
            raise SystemExit(f"--{_k.replace('_', '-')} {_v} contradicts the run's record ({cfg_[_k]}); "
                             "the data configuration is read from the run and cannot be overridden")
    args.p, args.op, args.train_frac = cfg_["p"], cfg_["op"], cfg_["train_frac"]
    data = run_dataset(root, args.tag)
    F, _ = make_fourier_basis(args.p, dtype=torch.float64)
    snap = load_snapshot(checkpoint_paths(root / "checkpoints" / args.tag)[-1][1], data)
    cons = key_freqs_consensus(snap.model.W_E.detach(), snap.neuron_acts, F, args.p)
    K = cons["consensus"] if cons["agree"] else cons["union"]
    print(f"{args.tag}: key frequencies {K} at step {snap.step}", flush=True)

    if 2 ** len(K) > args.max_subsets:
        print(f"  {2**len(K)} subsets exceeds the cap; testing singletons and "
              f"leave-one-out only", flush=True)
        subsets = [()] + [(k,) for k in K] + \
                  [tuple(x for x in K if x != k) for k in K] + [tuple(K)]
    else:
        subsets = [s for r in range(len(K) + 1) for s in itertools.combinations(K, r)]

    rows = []
    for sub in subsets:
        m = ablate_embedding_frequencies(snap.model, F, args.p, list(sub), keep=True)
        rows.append({"subset": list(sub), "size": len(sub), **evaluate_model(m, data)})
        print(f"  keep {str(list(sub)):<24s} test acc {rows[-1]['test_acc']:.4f} "
              f"loss {rows[-1]['test_loss']:.4f}", flush=True)

    good = [r for r in rows if r["test_acc"] >= args.threshold]
    minimal = min((r["size"] for r in good), default=None)
    out = {
        "tag": args.tag, "step": snap.step, "key_freqs": K,
        "threshold": args.threshold,
        "rows": rows,
        "sufficient_subsets": [r["subset"] for r in good],
        "minimal_size": minimal,
        "minimal_subsets": [r["subset"] for r in good if r["size"] == minimal],
        "redundant_frequencies": sorted(
            set(K) - set().union(*[set(r["subset"]) for r in good if r["size"] == minimal])
        ) if good else [],
        "chance_acc": 1.0 / args.p,
    }
    path = root / "results" / f"{args.tag}_redundancy.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  minimal sufficient size {minimal}; redundant: "
          f"{out['redundant_frequencies']}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
