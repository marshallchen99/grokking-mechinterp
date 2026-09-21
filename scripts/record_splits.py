#!/usr/bin/env python3
"""Record every run's train/test split fingerprint, and check it is the split trained on.

The split is not stored: it is rebuilt from (p, op, train_frac, seed) with
torch.randperm.  results/split_hashes.json holds a fingerprint of each run's
split, and the analysis checks every rebuilt split against it, so a torch
version that shuffled differently stops the analysis instead of silently using
another split.  An existing fingerprint is never replaced: a rebuilt split that
disagrees with it is an error, and the file is left as it was.

A fingerprint only shows the rebuild is stable.  To show it is the split the
run was *trained* on, the final checkpoint's train and test loss on the rebuilt
split are compared with the losses the run logged while training.  Accuracy
cannot do this -- a model at 100% train and test accuracy scores 100% on any
split -- but the mean loss over a different subset of pairs is different.  As
a check on the check, the same losses are computed on a deliberately different
split, which must disagree with the log.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as tF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grokking.analysis.core import checkpoint_paths, load_snapshot   # noqa: E402
from grokking.data import make_dataset                          # noqa: E402
from grokking.model import final_logits                         # noqa: E402
from grokking.runinfo import SPLIT_HASHES, provenance, run_config    # noqa: E402

LOSS_RTOL = 1e-3          # logged vs recomputed loss on the same split
OTHER_SEED_OFFSET = 1000  # the deliberately different split


def losses(model, d, in_float32):
    """Train and test loss computed the way the run logged them."""
    out = {}
    with torch.no_grad():
        for name, (x, y) in (("train", d.train()), ("test", d.test())):
            lg = final_logits(model(x, last_only=True), d.p)
            out[name] = float(tF.cross_entropy(lg if in_float32 else lg.double(), y))
    return out


def rel_diff(logged, got):
    return max(abs(logged[k] - got[k]) / max(abs(logged[k]), 1e-300) for k in logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    table_path = root / "results" / SPLIT_HASHES
    existing = json.loads(table_path.read_text())["runs"] if table_path.exists() else {}

    runs, changed, failed = {}, [], []
    for f in sorted((root / "results").glob("*_history.json")):
        tag = f.name[: -len("_history.json")]
        c = run_config(root, tag)
        d = make_dataset(p=c["p"], op=c["op"], train_frac=c["train_frac"], seed=c["seed"])
        entry = {"split_hash": d.split_hash()}
        if tag in existing and existing[tag]["split_hash"] != entry["split_hash"]:
            changed.append(tag)
        paths = checkpoint_paths(root / "checkpoints" / tag)
        last = json.loads(f.read_text())["history"][-1]
        if paths and paths[-1][0] == last["step"]:
            model = load_snapshot(paths[-1][1], d).model
            logged = {"train": last["train_loss"], "test": last["test_loss"]}
            # a loss logged in float32 is exactly a float32 number
            f32 = all(float(torch.tensor(v, dtype=torch.float32)) == v for v in logged.values())
            other = make_dataset(p=c["p"], op=c["op"], train_frac=c["train_frac"],
                                 seed=c["seed"] + OTHER_SEED_OFFSET)
            entry.update({
                "checked_at_step": last["step"], "logged_in_float32": f32,
                "loss_rel_diff": rel_diff(logged, losses(model, d, f32)),
                "other_split_loss_rel_diff": rel_diff(logged, losses(model, other, f32))})
            entry["matches"] = (entry["loss_rel_diff"] < LOSS_RTOL
                                and entry["other_split_loss_rel_diff"] > LOSS_RTOL)
            if not entry["matches"]:
                failed.append(tag)
        runs[tag] = entry
        print(f"  {tag:24s} {entry['split_hash']}  "
              + ("NOT CHECKED (no final checkpoint)" if "matches" not in entry else
                 f"own split {entry['loss_rel_diff']:.1e}, other split "
                 f"{entry['other_split_loss_rel_diff']:.1e}"
                 + ("" if entry["matches"] else "  FAILED")), flush=True)
    if changed:
        raise SystemExit(f"rebuilt splits differ from the recorded fingerprints for {changed}; "
                         f"{table_path} left unchanged (has torch's shuffling changed?)")
    if failed:
        raise SystemExit(f"the logged losses do not identify the split for {failed}; "
                         f"{table_path} left unchanged")
    table_path.write_text(json.dumps(
        {"_provenance": provenance(root), "loss_rtol": LOSS_RTOL, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
