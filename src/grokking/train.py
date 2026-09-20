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


def cross_entropy_at(logits: torch.Tensor, labels: torch.Tensor,
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Cross-entropy at a chosen precision -- float64 by default.

    The dtype is a parameter rather than a constant so that the cost of the
    float32 floor can be measured rather than asserted.
    """
    return torch.nn.functional.cross_entropy(logits.to(dtype), labels)


def cross_entropy_f64(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with the logits upcast to float64.

    This matters more than it looks.  Once the model has memorised, the gap
    between the correct logit and the rest grows past about 16, and in float32
    `log_softmax` quantises at 2^-23 = 1.2e-7: the reported loss bottoms out at
    that value and the gradient of the correct class degrades towards zero.
    Since the whole phenomenon lives in the tens of thousands of steps *after*
    the training loss is nominally zero, a loss floor is exactly the wrong
    artefact to have.  Parameters stay float32; only the logits are upcast, and
    the measured cost is inside run-to-run noise.
    """
    return torch.nn.functional.cross_entropy(logits.to(torch.float64), labels)


@dataclass
class TrainConfig:
    steps: int = 40_000
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 1.0
    optimizer: str = "adamw"
    warmup_steps: int = 10
    loss_dtype: str = "float64"
    log_every: int = 10
    n_log_checkpoints: int = 120
    dense_from: int = 5_000
    dense_to: int = 20_000
    dense_every: int = 250
    seed: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def checkpoint_steps(total: int, n_log: int = 120, dense: tuple = (5_000, 20_000, 250)
                     ) -> List[int]:
    """A hybrid log + dense checkpoint schedule.

    The trajectory is the object of study, and neither spacing works alone:
    log spacing puts almost everything in the first few hundred steps and
    almost nothing across the transition; linear spacing does the reverse.  So:

      * a log band of `n_log` geometric points across the whole run, which
        resolves the memorisation phase (over by step ~200) and gives cheap
        coverage of the long plateau;
      * a dense band of evenly spaced points across the window where the
        transition is expected, which is where the interesting derivative is.

    If a run's transition falls outside the dense band, widen the band and
    re-run rather than interpolating across it.
    """
    import math
    S = {0, total}
    if total >= 1:
        for i in range(n_log):
            S.add(int(round(math.exp(math.log(total) * i / max(n_log - 1, 1)))))
    lo, hi, step = dense
    for s in range(lo, min(hi, total) + 1, step):
        S.add(s)
    return sorted(x for x in S if 0 <= x <= total)


DTYPES = {"float64": torch.float64, "float32": torch.float32}


@torch.no_grad()
def evaluate(model: OneLayerTransformer, tokens: torch.Tensor, labels: torch.Tensor,
             n_answer: int) -> Dict[str, float]:
    logits = final_logits(model(tokens, last_only=True), n_answer)
    loss = cross_entropy_f64(logits, labels)
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

        # Linear warmup over the first `warmup_steps` optimizer steps.  Note the
        # multiplier at step 0 is exactly 0, so the first step updates nothing
        # -- including the decoupled weight decay, which AdamW scales by lr.
        # Both reference implementations behave this way; it is not an
        # off-by-one to be "fixed".
        w = max(cfg.warmup_steps, 1)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda step: min(step / w, 1.0))

        self._loss_dtype = DTYPES[cfg.loss_dtype]
        self.ckpt_at = set(checkpoint_steps(
            cfg.steps, cfg.n_log_checkpoints,
            (cfg.dense_from, cfg.dense_to, cfg.dense_every)))
        self._write_manifest()
        self.history: List[Dict] = []

    # ------------------------------------------------------------------

    def _write_manifest(self):
        """Record exactly what was run, next to the checkpoints it produced."""
        import torch as _t
        (self.ckpt_dir / "manifest.json").write_text(json.dumps({
            "tag": self.tag,
            "model_cfg": self.model.cfg.to_dict(),
            "train_cfg": self.cfg.to_dict(),
            "data": {"p": self.data.p, "op": self.data.op,
                     "train_frac": self.data.train_frac, "seed": self.data.seed,
                     "n_train": self.data.n_train, "n_test": self.data.n_test},
            "n_params": self.model.n_params(),
            "checkpoint_steps": sorted(self.ckpt_at),
            "torch_version": _t.__version__,
            "num_threads": _t.get_num_threads(),
        }, indent=1))

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
            loss = cross_entropy_at(logits, self.train_y, self._loss_dtype)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.sched.step()

        self._write_history()
        return self.history
