"""Full-batch training loop with log-spaced checkpointing.

Two things make this different from an ordinary training script.

1.  Full batch.  The whole training set is 3830 examples of 3 tokens; there is
    no reason to minibatch, and the grokking literature trains full batch.  One
    "step" is therefore one gradient step on the entire training set.

2.  Checkpoints matter more than the final model.  The interesting object of
    study is the *trajectory* -- the circuit forms silently over tens of
    thousands of steps while every externally visible metric is flat.  So we
    save a log-spaced set of checkpoints, dense at the start and at the point
    where test accuracy moves, and we log metrics every step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import ModularDataset
from .model import ModelConfig, OneLayerTransformer, final_logits


@dataclass
class TrainConfig:
    steps: int = 40_000
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 1.0
    optimizer: str = "adamw"
    log_every: int = 10
    n_checkpoints: int = 160
    seed: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def checkpoint_steps(total: int, n: int) -> List[int]:
    """Log-spaced step indices, always including 0 and `total`.

    Log spacing is the right choice here: the memorisation phase is over within
    a few hundred steps, while cleanup happens tens of thousands of steps later,
    so linear spacing would waste almost every checkpoint on the flat middle.
    """
    if n >= total + 1:
        return list(range(total + 1))
    import math
    lo, hi = 1.0, float(total)
    raw = [0] + [int(round(math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 2))))
                 for i in range(n - 1)]
    out = sorted(set(raw) | {0, total})
    return [s for s in out if 0 <= s <= total]


@torch.no_grad()
def evaluate(model: OneLayerTransformer, tokens: torch.Tensor, labels: torch.Tensor,
             n_answer: int) -> Dict[str, float]:
    logits = final_logits(model(tokens, last_only=True), n_answer)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()
    return {"loss": float(loss), "acc": float(acc)}


class Trainer:
    def __init__(
        self,
        model: OneLayerTransformer,
        data: ModularDataset,
        cfg: TrainConfig,
        out_dir: Path,
        tag: str = "run",
    ):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.ckpt_dir = self.out_dir / "checkpoints" / tag
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "results").mkdir(parents=True, exist_ok=True)

        self.train_x, self.train_y = data.train()
        self.test_x, self.test_y = data.test()
        self.n_answer = data.p

        if cfg.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas),
                eps=cfg.eps, weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps,
            )
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

        self.ckpt_at = set(checkpoint_steps(cfg.steps, cfg.n_checkpoints))
        self.history: List[Dict] = []

    # ------------------------------------------------------------------

    def _save_ckpt(self, step: int):
        torch.save(
            {"step": step, "state_dict": self.model.state_dict()},
            self.ckpt_dir / f"step_{step:06d}.pt",
        )

    def _write_history(self):
        path = self.out_dir / "results" / f"{self.tag}_history.json"
        payload = {
            "tag": self.tag,
            "model_cfg": self.model.cfg.to_dict(),
            "train_cfg": self.cfg.to_dict(),
            "data": {"p": self.data.p, "op": self.data.op,
                     "train_frac": self.data.train_frac, "seed": self.data.seed,
                     "n_train": self.data.n_train, "n_test": self.data.n_test},
            "n_params": self.model.n_params(),
            "history": self.history,
        }
        path.write_text(json.dumps(payload))

    def run(self, verbose: bool = True, flush_every: int = 2000) -> List[Dict]:
        cfg = self.cfg
        t0 = time.time()
        for step in range(cfg.steps + 1):
            if step in self.ckpt_at:
                self._save_ckpt(step)

            # ---- metrics (before the update, so step 0 is the init) --------
            if step % cfg.log_every == 0 or step == cfg.steps:
                tr = evaluate(self.model, self.train_x, self.train_y, self.n_answer)
                te = evaluate(self.model, self.test_x, self.test_y, self.n_answer)
                self.history.append({
                    "step": step,
                    "train_loss": tr["loss"], "train_acc": tr["acc"],
                    "test_loss": te["loss"], "test_acc": te["acc"],
                    "weight_norm": self.model.param_norm(),
                    "elapsed": time.time() - t0,
                })
                if verbose and (step % (cfg.log_every * 100) == 0 or step == cfg.steps):
                    print(f"step {step:6d} | train {tr['loss']:.4f}/{tr['acc']:.3f} "
                          f"| test {te['loss']:.4f}/{te['acc']:.3f} "
                          f"| |w| {self.history[-1]['weight_norm']:.1f} "
                          f"| {time.time()-t0:.0f}s", flush=True)
                if step % flush_every == 0:
                    self._write_history()

            if step == cfg.steps:
                break

            # ---- one full-batch gradient step ------------------------------
            logits = final_logits(self.model(self.train_x, last_only=True), self.n_answer)
            loss = F.cross_entropy(logits, self.train_y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        self._write_history()
        return self.history
