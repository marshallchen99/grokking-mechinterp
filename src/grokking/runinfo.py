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


def run_dataset(root: Path, tag: str) -> ModularDataset:
    """The dataset -- including the exact train/test split -- a run was trained on."""
    c = run_config(root, tag)
    return make_dataset(p=c["p"], op=c["op"], train_frac=c["train_frac"], seed=c["seed"])


# ------------------------------------------------------------ training choices

# Three runs were trained before the history file recorded the loss precision
# and the warmup.  Their values are established from the git history, not
# inferred from the results:
#   main_add_s0  launched 2026-09-20 11:18, before commit 6a213ef introduced the
#                float64 loss; the train.py of that time (c147ab2) uses
#                F.cross_entropy on float32 logits and no LR scheduler.
#   B_add_s0,    launched 13:01, after 6a213ef (float64 loss, 10-step warmup) and
#   B_mul_s0     before fda9115 added the loss_dtype field; train.py at e25caae
#                calls cross_entropy_f64 in the training step.
# A run missing from both its record and this table is reported as "not
# recorded" -- never given a default, which is how B_add_s0 was once labelled
# float32.
LEGACY_TRAINING = {
    # num_threads: the run was launched with --threads 6, visible in the process list
    # at the time; the run did not record it and no shipped file does, so this
    # entry rests on the launch command alone.
    "main_add_s0": {"loss_dtype": "float32", "warmup_steps": None, "num_threads": 6},
    "B_add_s0": {"loss_dtype": "float64", "warmup_steps": 10},
    "B_mul_s0": {"loss_dtype": "float64", "warmup_steps": 10},
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
