"""Loading a checkpoint and running the whole input table through it.

Every analysis in this package is a function of the model's behaviour on all
p * p possible inputs, not on a sample, so the natural unit of work is: load a
checkpoint, run the complete table once, keep the intermediates.  At p = 113
that is 12769 rows, which costs a fraction of a second, so nothing here needs
to be clever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from ..data import ModularDataset, make_dataset
from ..model import ModelConfig, OneLayerTransformer

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


@dataclass
class Snapshot:
    """One checkpoint, with the model's behaviour on the full p x p table.

    All grids are indexed [a, b, ...] so that `grid[a, b]` is the value for the
    input pair (a, b).
    """

    step: int
    model: OneLayerTransformer
    data: ModularDataset

    logits: torch.Tensor        # (p, p, p)   answer logits, "=" column dropped
    neuron_acts: torch.Tensor   # (p, p, d_mlp)  post-ReLU at the final position
    attn: torch.Tensor          # (p, p, n_heads, n_ctx)  final-position pattern
    resid_mid: torch.Tensor     # (p, p, d_model) after attention, before MLP

    @property
    def p(self) -> int:
        return self.data.p

    def flat_logits(self) -> torch.Tensor:
        return self.logits.reshape(self.p * self.p, self.p)

    def losses(self, logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Cross-entropy and accuracy on train and test, for (possibly edited) logits."""
        lg = self.flat_logits() if logits is None else logits.reshape(self.p * self.p, self.p)
        y = self.data.labels
        out = {}
        for name, idx in (("train", self.data.train_idx), ("test", self.data.test_idx)):
            sub, lab = lg[idx], y[idx]
            out[f"{name}_loss"] = float(torch.nn.functional.cross_entropy(sub, lab))
            out[f"{name}_acc"] = float((sub.argmax(-1) == lab).float().mean())
        return out


def _model_cfg_from_state(state: Dict[str, torch.Tensor], seed: int = 0) -> ModelConfig:
    d_vocab, d_model = state["W_E"].shape
    n_ctx = state["W_pos"].shape[0]
    n_heads, _, d_head = state["W_Q"].shape
    d_mlp = state["W_in"].shape[1]
    return ModelConfig(d_vocab=d_vocab, n_ctx=n_ctx, d_model=d_model,
                       n_heads=n_heads, d_head=d_head, d_mlp=d_mlp, seed=seed)


@torch.no_grad()
def load_snapshot(path: str | Path, data: Optional[ModularDataset] = None) -> Snapshot:
    """Load a checkpoint and run the full input table through it."""
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state, step = blob["state_dict"], int(blob["step"])

    cfg = _model_cfg_from_state(state)
    model = OneLayerTransformer(cfg)
    model.load_state_dict(state)
    model.eval()

    if data is None:
        data = make_dataset(p=cfg.d_vocab - 1)
    p = data.p

    _, cache = model.run_with_cache(data.inputs)
    logits = cache["logits"][:, -1, :p].reshape(p, p, p)
    neuron = cache["mlp_post"][:, -1, :].reshape(p, p, -1)
    attn = cache["attn_pattern"][:, :, -1, :].reshape(p, p, cfg.n_heads, cfg.n_ctx)
    resid = cache["resid_mid"][:, -1, :].reshape(p, p, -1)

    return Snapshot(step=step, model=model, data=data, logits=logits,
                    neuron_acts=neuron, attn=attn, resid_mid=resid)


def checkpoint_paths(ckpt_dir: str | Path) -> List[Tuple[int, Path]]:
    """All checkpoints in a directory, sorted by step."""
    out = []
    for p in Path(ckpt_dir).glob("step_*.pt"):
        m = _STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def iter_checkpoints(ckpt_dir: str | Path, data: Optional[ModularDataset] = None,
                     every: int = 1) -> Iterator[Snapshot]:
    """Yield snapshots in step order (every `every`-th checkpoint)."""
    for i, (_, path) in enumerate(checkpoint_paths(ckpt_dir)):
        if i % every == 0:
            yield load_snapshot(path, data=data)
