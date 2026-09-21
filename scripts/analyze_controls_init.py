#!/usr/bin/env python3
"""Do the original run and the corrected run share an initialisation?

The write-up once said they "do not share an initialisation at all", because
changing W_U's shape changes the random draw.  That is only true for W_U: the
generator draws the tensors in a fixed order and W_U comes last, so every other
tensor is identical.  This measures it rather than asserting it.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grokking.analysis.core import checkpoint_paths   # noqa: E402

a_tag, b_tag = (sys.argv[1:3] if len(sys.argv) >= 3 else ("main_add_s0", "B_add_s0"))
A = torch.load(checkpoint_paths(ROOT / "checkpoints" / a_tag)[0][1], weights_only=True)["state_dict"]
B = torch.load(checkpoint_paths(ROOT / "checkpoints" / b_tag)[0][1], weights_only=True)["state_dict"]
rows = {}
for name in A:
    same_shape = A[name].shape == B[name].shape
    rows[name] = {"shape_a": list(A[name].shape), "shape_b": list(B[name].shape),
                  "identical": bool(same_shape and torch.equal(A[name], B[name]))}
    if not same_shape:
        k = min(A[name].shape[-1], B[name].shape[-1])
        rows[name]["shared_columns_identical"] = bool(torch.equal(A[name][..., :k], B[name][..., :k]))
out = {"a": a_tag, "b": b_tag, "tensors": rows,
       "n_tensors": len(rows), "n_identical": sum(r["identical"] for r in rows.values())}
(ROOT / "results" / "controls_init.json").write_text(json.dumps(out, indent=1))
for n, r in rows.items():
    print(f"  {n:6s} {str(r['shape_a']):14s} {str(r['shape_b']):14s} identical={r['identical']}"
          + (f"  shared columns identical={r['shared_columns_identical']}" if "shared_columns_identical" in r else ""))
print(f"{out['n_identical']} of {out['n_tensors']} tensors identical at step 0")
