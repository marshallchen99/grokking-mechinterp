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
