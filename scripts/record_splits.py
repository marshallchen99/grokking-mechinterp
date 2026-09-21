#!/usr/bin/env python3
"""Record every run's train/test split fingerprint, and check it is the split trained on.

The split is not stored: it is rebuilt from (p, op, train_frac, seed) with
torch.randperm.  This writes results/split_hashes.json, which the analysis
checks every rebuilt split against, so that a torch version that shuffled
differently stops the analysis instead of silently using another split.

A fingerprint of the rebuilt split only shows that the rebuild is stable.  To
show it is the split the run was *trained* on, the final checkpoint is
evaluated on the rebuilt split and compared with the train and test accuracy
the run logged while training: on any other split they would not agree.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grokking.analysis.ablation import evaluate_model          # noqa: E402
from grokking.analysis.core import checkpoint_paths, load_snapshot   # noqa: E402
from grokking.data import make_dataset                          # noqa: E402
from grokking.runinfo import SPLIT_HASHES, provenance, run_config    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    runs = {}
    for f in sorted((root / "results").glob("*_history.json")):
        tag = f.name[: -len("_history.json")]
        c = run_config(root, tag)
        d = make_dataset(p=c["p"], op=c["op"], train_frac=c["train_frac"], seed=c["seed"])
        entry = {"split_hash": d.split_hash()}
        paths = checkpoint_paths(root / "checkpoints" / tag)
        last = json.loads(f.read_text())["history"][-1]
        if paths and paths[-1][0] == last["step"]:
            ev = evaluate_model(load_snapshot(paths[-1][1], d).model, d)
            entry["checked_at_step"] = last["step"]
            # accuracies are fractions of a fixed count, so they must agree exactly
            # up to float formatting; losses are only logged to float precision
            entry["train_acc_logged_vs_rebuilt"] = [last["train_acc"], ev["train_acc"]]
            entry["test_acc_logged_vs_rebuilt"] = [last["test_acc"], ev["test_acc"]]
            entry["matches"] = (abs(last["train_acc"] - ev["train_acc"]) < 1e-6
                                and abs(last["test_acc"] - ev["test_acc"]) < 1e-6)
        runs[tag] = entry
        print(f"  {tag:24s} {entry['split_hash']}  "
              + ("matches its training log" if entry.get("matches") else
                 "NOT CHECKED (no final checkpoint)" if "matches" not in entry else "MISMATCH"),
              flush=True)
    bad = [t for t, e in runs.items() if e.get("matches") is False]
    (root / "results" / SPLIT_HASHES).write_text(json.dumps(
        {"_provenance": provenance(root), "runs": runs}, indent=1))
    if bad:
        raise SystemExit(f"rebuilt split disagrees with the training log for {bad}")


if __name__ == "__main__":
    main()
