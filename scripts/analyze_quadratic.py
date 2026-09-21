#!/usr/bin/env python3
"""Why does a model trained on a^2+ab+b^2 plateau near 50%?

The form is symmetric in (a, b) while the train/test split is over ORDERED
pairs, so many held-out pairs have their transpose in the training set.  A
model that memorised the training table and learned only that it is symmetric
would score exactly on those and nothing else.  This measures that directly,
and also how the remaining errors are structured, rather than reading a
mechanism into the accuracy.
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
from grokking.data import make_dataset
from grokking.runinfo import run_config, run_dataset  # noqa: E402                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)

    out = []
    for tag in args.tags:
        mani = json.loads((root / "checkpoints" / tag / "manifest.json").read_text())
        p = mani["data"]["p"]
        d = run_dataset(root, tag)   # the run's own split, not a hard-coded seed
        snap = load_snapshot(checkpoint_paths(root / "checkpoints" / tag)[-1][1], d)
        snap_model = snap.model
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

        rec = {"tag": tag, "p": p, "p_mod_3": p % 3, "budget": mani["train_cfg"]["steps"],
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

        # The form is symmetric in (a, b) but the split is over ORDERED pairs,
        # so a held-out (a, b) often has its transpose (b, a) in the training
        # set.  A model that memorised the training pairs and learned only that
        # the table is symmetric answers exactly those correctly and nothing
        # else -- which would put test accuracy near the fraction of test pairs
        # whose transpose was seen.  Check that before calling it anything more.
        train = d.train_mask()
        seen_T = train.T & test                    # (a,b) held out, (b,a) trained on
        rec["frac_test_with_transpose_in_train"] = float(seen_T.sum() / test.sum())
        rec["acc_transpose_in_train"] = float((pred == true)[seen_T].float().mean())
        rec["acc_transpose_held_out"] = float((pred == true)[test & ~train.T].float().mean())
        rec["predictions_symmetric"] = float((pred == pred.T).float().mean())
        unseen = test & ~train.T & ~coincide
        rec["n_unseen"] = int(unseen.sum())
        rec["unseen_predicts_flipped"] = float((pred == flip)[unseen].float().mean())
        rec["unseen_flip_enrichment"] = rec["unseen_predicts_flipped"] * p
        # f(a,-b) = f(-a,b) = f(b,-a) = f(-b,a).  Is a sign-flip prediction just a
        # memorised answer retrieved from one of these partners?
        neg = lambda x: (-x) % p
        partner = torch.zeros(p, p, dtype=torch.bool)
        for i, j in ((a, neg(b)), (neg(a), b), (b, neg(a)), (neg(b), a)):
            partner |= train[i, j]
        w, wo = unseen & partner, unseen & ~partner
        rec["unseen_with_flipped_partner_trained"] = {
            "n": int(w.sum()), "predicts_flipped": float((pred == flip)[w].float().mean())}
        # A single sign flip changes the value; a double flip (-a,-b) or (-b,-a)
        # keeps it.  If the representation simply put x near -x, double-flip
        # partners would be retrieved as readily -- and would give the RIGHT
        # answer.  Measure both, and measure the embedding similarity directly.
        dbl = torch.zeros(p, p, dtype=torch.bool)
        for i, j in ((neg(a), neg(b)), (neg(b), neg(a))):
            dbl |= train[i, j]
        dbl_only = test & ~train.T & dbl & ~partner & ~coincide
        rec["unseen_with_only_double_flip_partner"] = {
            "n": int(dbl_only.sum()),
            "correct": float((pred == true)[dbl_only].float().mean()) if int(dbl_only.sum()) else None}
        E = snap_model.W_E.detach()[:p]
        En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
        xs = torch.arange(1, p)
        cos_neg = (En[xs] * En[(-xs) % p]).sum(1)
        g = torch.Generator().manual_seed(0)
        ys = (xs + torch.randint(1, p - 1, (p - 1,), generator=g)) % p
        ys = torch.where(ys == 0, torch.ones_like(ys), ys)
        cos_rand = (En[xs] * En[ys]).sum(1)
        rec["embedding_cos_x_minus_x"] = float(cos_neg.mean())
        rec["embedding_cos_x_random"] = float(cos_rand.mean())
        rec["unseen_without_flipped_partner"] = {
            "n": int(wo.sum()),
            "predicts_flipped": float((pred == flip)[wo].float().mean()) if int(wo.sum()) else None}
        rec["explained_by_memorisation_plus_symmetry"] = (
            rec["frac_test_with_transpose_in_train"] * rec["acc_transpose_in_train"]
            + (1 - rec["frac_test_with_transpose_in_train"]) * rec["acc_transpose_held_out"])
        out.append(rec)
        print(f"p={p} ({rec['p_mod_3']} mod 3, form "
              f"{'splits' if rec['form_splits'] else 'irreducible'}): "
              f"test acc {rec['test_acc']:.4f}", flush=True)
        for n, v in rec["matches"].items():
            print(f"    prediction matches {n:30s} {v:.4f}", flush=True)
        print(f"    one of the two forms: {rec['either_form']:.4f}; "
              f"sign-flip enrichment over chance: {rec['sign_flip_enrichment']:.1f}x", flush=True)
        print(f"    test pairs whose transpose was trained on: "
              f"{rec['frac_test_with_transpose_in_train']:.3f} -> accuracy "
              f"{rec['acc_transpose_in_train']:.4f}; the rest {rec['acc_transpose_held_out']:.4f}",
              flush=True)
        print(f"    accuracy where the two forms coincide: "
              f"{rec['acc_where_forms_coincide']:.4f} ({rec['n_coincide']} pairs)", flush=True)

    path = root / "results" / "quadratic_form.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
