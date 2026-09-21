#!/usr/bin/env python3
"""Do the original run and the corrected run share an initialisation?

The write-up once said they "do not share an initialisation at all", because
changing W_U's shape changes the random draw.  That is only true for W_U: the
generator draws the tensors in a fixed order and W_U comes last, so every other
tensor is identical.  This measures it rather than asserting it.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grokking.analysis.core import checkpoint_paths   # noqa: E402
from grokking.runinfo import provenance               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="main_add_s0")
    ap.add_argument("--b", default="B_add_s0")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    (sa, pa), (sb, pb) = (checkpoint_paths(root / "checkpoints" / t)[0] for t in (args.a, args.b))
    A = torch.load(pa, weights_only=True)["state_dict"]
    B = torch.load(pb, weights_only=True)["state_dict"]
    rows = {}
    for name in A:
        same_shape = A[name].shape == B[name].shape
        rows[name] = {"shape_a": list(A[name].shape), "shape_b": list(B[name].shape),
                      "identical": bool(same_shape and torch.equal(A[name], B[name]))}
        if not same_shape:
            k = min(A[name].shape[-1], B[name].shape[-1])
            rows[name]["shared_columns_identical"] = bool(
                torch.equal(A[name][..., :k], B[name][..., :k]))
    out = {"a": args.a, "b": args.b, "step_a": sa, "step_b": sb,
           "_provenance": provenance(root), "tensors": rows,
           "n_tensors": len(rows), "n_identical": sum(r["identical"] for r in rows.values())}

    # Run a's extra output column (for '=') was sliced off before the loss.  If
    # it really got no gradient, AdamW only decays it: every element is its
    # step-0 value times (1 - lr * wd) ** step.  Check that at every checkpoint.
    p = json.loads((root / "results" / f"{args.b}_history.json").read_text())["data"]["p"]
    if A["W_U"].shape[1] > p:
        ha = json.loads((root / "results" / f"{args.a}_history.json").read_text())
        decay = 1.0 - ha["train_cfg"]["lr"] * ha["train_cfg"]["weight_decay"]
        col0 = A["W_U"][:, p].double()
        worst = 0.0
        ckpts = checkpoint_paths(root / "checkpoints" / args.a)
        for s, path in ckpts[1:]:
            col = torch.load(path, weights_only=True)["state_dict"]["W_U"][:, p].double()
            pred = col0 * decay ** s
            worst = max(worst, float(((col - pred).norm() / pred.norm()).item()))
        out["extra_column"] = {"index": p, "checkpoints_checked": len(ckpts) - 1,
                               "decay_per_step": decay,
                               "max_relative_deviation_from_pure_decay": worst}
        print(f"extra column {p}: max relative deviation from pure decay {worst:.2e} "
              f"over {len(ckpts) - 1} checkpoints")
    (root / "results" / "controls_init.json").write_text(json.dumps(out, indent=1))
    for n, r in rows.items():
        print(f"  {n:6s} {str(r['shape_a']):14s} {str(r['shape_b']):14s} identical={r['identical']}"
              + (f"  shared columns identical={r['shared_columns_identical']}"
                 if "shared_columns_identical" in r else ""))
    print(f"{out['n_identical']} of {out['n_tensors']} tensors identical at step {sa} / {sb}")


if __name__ == "__main__":
    main()
