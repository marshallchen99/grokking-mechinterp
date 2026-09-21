#!/usr/bin/env python3
"""What does a model that plateaus at 50% on a^2+ab+b^2 actually compute?

"Did not grok" is a description of the accuracy, not of the model.  A run that
stops at exactly half is doing something specific, and the interesting question
is what.  This tests one concrete hypothesis -- that the circuit has lost the
sign of b -- against the obvious alternatives, and reports how much of the
error it accounts for rather than declaring it the answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.core import checkpoint_paths, load_snapshot   # noqa: E402
from grokking.data import make_dataset                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)

    out = []
    for tag in args.tags:
        mani = json.loads((root / "checkpoints" / tag / "manifest.json").read_text())
        p = mani["data"]["p"]
        d = make_dataset(p=p, op="sq_sum_cross", train_frac=args.train_frac, seed=0)
        snap = load_snapshot(checkpoint_paths(root / "checkpoints" / tag)[-1][1], d)
        pred = snap.logits.argmax(-1)
        test = ~d.train_mask()

        a = torch.arange(p)[:, None].expand(p, p)
        b = torch.arange(p)[None, :].expand(p, p)
        # Candidate functions the model might have settled on instead.
        cands = {
            "true  a^2+ab+b^2": (a * a + a * b + b * b) % p,
            "sign-flipped b  a^2-ab+b^2": (a * a - a * b + b * b) % p,
            "(a+b)^2": ((a + b) ** 2) % p,
            "a^2+b^2": (a * a + b * b) % p,
            "a*b": (a * b) % p,
        }
        true, flip = cands["true  a^2+ab+b^2"], cands["sign-flipped b  a^2-ab+b^2"]
        coincide = true == flip

        rec = {"tag": tag, "p": p, "p_mod_3": p % 3,
               "form_splits": p % 3 == 1,
               "train_acc": float((pred == true)[~test].float().mean()),
               "test_acc": float((pred == true)[test].float().mean()),
               "chance": 1.0 / p,
               "forms_differ_on": float((~coincide).float().mean()),
               "matches": {n: float((pred == v)[test].float().mean())
                           for n, v in cands.items()},
               "sign_flip_enrichment": None,
               "acc_where_forms_coincide": float((pred == true)[test & coincide].float().mean()),
               "n_coincide": int((test & coincide).sum()),
               "either_form": float(((pred == true) | (pred == flip))[test].float().mean()),
               }
        rec["sign_flip_enrichment"] = (
            float(((pred == flip) & ~coincide)[test].float().mean()) * p)
        out.append(rec)
        print(f"p={p} ({rec['p_mod_3']} mod 3, form "
              f"{'splits' if rec['form_splits'] else 'irreducible'}): "
              f"test acc {rec['test_acc']:.4f}", flush=True)
        for n, v in rec["matches"].items():
            print(f"    prediction matches {n:30s} {v:.4f}", flush=True)
        print(f"    one of the two forms: {rec['either_form']:.4f}; "
              f"sign-flip enrichment over chance: {rec['sign_flip_enrichment']:.1f}x", flush=True)
        print(f"    accuracy where the two forms coincide: "
              f"{rec['acc_where_forms_coincide']:.4f} ({rec['n_coincide']} pairs)", flush=True)

    path = root / "results" / "quadratic_form.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
