"""Where a run's data configuration comes from: the run itself.

The analysis scripts used to take the modulus, operation, training fraction and
data seed as command-line flags with defaults.  A run analysed without the
right `--data-seed` was then silently evaluated on the seed-0 split, where
about half of its "training" pairs are really held-out pairs -- and nothing
crashed, because every split is a perfectly valid split.  Seventeen runs were
analysed that way before an external review caught it.

So the configuration is read from the run's own history file, and a run
without one is an error rather than a default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .data import ModularDataset, make_dataset


def run_config(root: Path, tag: str) -> Dict:
    """p, op, train_frac and data seed exactly as the run was trained."""
    path = Path(root) / "results" / f"{tag}_history.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no history for run {tag!r} at {path}; its data configuration "
            f"cannot be recovered, and guessing it is how the wrong split gets used")
    d = json.loads(path.read_text())["data"]
    missing = [k for k in ("p", "op", "train_frac", "seed") if k not in d]
    if missing:
        raise KeyError(f"history for {tag!r} lacks {missing}")
    return {"p": int(d["p"]), "op": d["op"],
            "train_frac": float(d["train_frac"]), "seed": int(d["seed"])}


SPLIT_HASHES = "split_hashes.json"   # fingerprints of the shipped runs' splits


def recorded_split_hash(root: Path, tag: str):
    """The split fingerprint from the run's record, else from results/split_hashes.json."""
    h = json.loads((Path(root) / "results" / f"{tag}_history.json").read_text())
    if "split_hash" in h.get("data", {}):
        return h["data"]["split_hash"]
    table = Path(root) / "results" / SPLIT_HASHES
    return json.loads(table.read_text()).get(tag) if table.exists() else None


def run_dataset(root: Path, tag: str) -> ModularDataset:
    """The dataset -- including the exact train/test split -- a run was trained on."""
    c = run_config(root, tag)
    d = make_dataset(p=c["p"], op=c["op"], train_frac=c["train_frac"], seed=c["seed"])
    want = recorded_split_hash(root, tag)
    if want is not None and d.split_hash() != want:
        raise RuntimeError(
            f"the split rebuilt for {tag!r} is not the one it was trained on "
            f"({d.split_hash()} != {want}); torch's randperm has probably changed. "
            f"The shipped runs used torch 2.12.")
    return d


def provenance(root: Path = None) -> Dict:
    """What produced a results file: code commit, command line, threads, versions.

    Written into every derived results file.  Without it, files made by two
    versions of an analysis script look identical, which is how a redefined
    field once sat unnoticed in two meanings across the shipped results.
    """
    import platform
    import subprocess
    import sys

    import torch

    repo = Path(__file__).resolve().parents[2]

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    # local absolute paths do not belong in shipped files
    places = sorted({str(Path(x).resolve()) for x in (repo, root) if x is not None},
                    key=len, reverse=True)

    def rel(a):
        for pre in places:
            a = a.replace(pre + "/", "").replace(pre, ".")
        return a

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "code_modified": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "argv": [rel(a) for a in sys.argv],
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


# ------------------------------------------------------------ training choices

# Three runs were trained before the history file recorded the loss precision.
# main_add_s0 records neither that nor the warmup; B_add_s0 and B_mul_s0 record
# the warmup (10 steps) but not the precision.  The values below come from the
# git history and the launch commands, not from the results:
#   main_add_s0  launched 2026-09-20 11:18 with `run_train.py --tag main_add_s0
#                --op add --steps 40000 --threads 6`, from a working tree first
#                committed 19 minutes later as c147ab2.  That commit's train.py
#                uses F.cross_entropy on float32 logits and no LR scheduler, and
#                it reproduces the run's initial weights and first steps.
#   B_add_s0,    launched 13:01 from a working tree committed six minutes later
#   B_mul_s0     as f381b15 (the run_train.py committed at 6a213ef and b5b7cfe
#                does not run); train.py is identical from 6a213ef to e25caae
#                and calls cross_entropy_f64 in the training step.
# Independent check: float64 runs' training loss settles near 9e-8 and the
# float32 runs' near 1e-5, and these two settle near 9e-8.
# A run missing from both its record and this table is reported as "not
# recorded" -- never given a default, which is how B_add_s0 was once labelled
# float32.
LEGACY_TRAINING = {
    # num_threads rests on the launch command alone; no shipped file records it.
    "main_add_s0": {"loss_dtype": "float32", "warmup_steps": None, "num_threads": 6},
    "B_add_s0": {"loss_dtype": "float64"},
    "B_mul_s0": {"loss_dtype": "float64"},
}


def training_choice(root: Path, tag: str, key: str):
    """A training choice from the run's record, else from LEGACY_TRAINING, else None."""
    path = Path(root) / "results" / f"{tag}_history.json"
    tc = json.loads(path.read_text()).get("train_cfg", {}) if path.exists() else {}
    if key in tc:
        return tc[key]
    mani = Path(root) / "checkpoints" / tag / "manifest.json"
    if mani.exists():
        m = json.loads(mani.read_text())
        if key in m:
            return m[key]
    return LEGACY_TRAINING.get(tag, {}).get(key)
