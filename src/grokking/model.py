"""A one-layer transformer, written out explicitly so every tensor can be read.

This deliberately avoids torch.nn.MultiheadAttention and any library that
fuses operations: the whole point of the project is to look at the individual
weight matrices and intermediate activations, so they all have to be reachable
by name.

Architecture (following the configuration used in the grokking literature):

    x        = W_E[tokens] + W_pos                      (B, n_ctx, d_model)
    x        = x + Attention(x)                         residual
    x        = x + MLP(x)                               residual
    logits   = x @ W_U                                  (B, n_ctx, d_vocab)

There is deliberately NO LayerNorm and there are NO biases.  Both are omitted
in the canonical grokking setup, and both would make the Fourier analysis
messier: LayerNorm makes every readout depend on the norm of the whole residual
stream, and biases add constant terms that show up as spurious "const"
components in the spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    d_vocab: int = 114          # p + 1, the extra token is "="
    n_ctx: int = 3              # [a, b, =]
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32            # d_model / n_heads
    d_mlp: int = 512
    act: str = "relu"
    causal: bool = True
    init_std: float = 0.0       # 0 -> use init_range / sqrt(d_model)
    init_range: float = 0.8
    seed: int = 0

    def __post_init__(self):
        if self.d_head * self.n_heads != self.d_model:
            raise ValueError(
                f"n_heads * d_head ({self.n_heads} * {self.d_head}) must equal "
                f"d_model ({self.d_model})"
            )

    def std(self) -> float:
        return self.init_std if self.init_std > 0 else self.init_range / (self.d_model ** 0.5)

    def to_dict(self) -> Dict:
        return asdict(self)


ACTS = {"relu": torch.relu, "gelu": torch.nn.functional.gelu}


class OneLayerTransformer(nn.Module):
    """One attention layer + one MLP layer, no normalisation, no biases."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        std = cfg.std()

        def param(*shape):
            return nn.Parameter(torch.randn(*shape, generator=g) * std)

        self.W_E = param(cfg.d_vocab, cfg.d_model)
        self.W_pos = param(cfg.n_ctx, cfg.d_model)

        self.W_Q = param(cfg.n_heads, cfg.d_model, cfg.d_head)
        self.W_K = param(cfg.n_heads, cfg.d_model, cfg.d_head)
        self.W_V = param(cfg.n_heads, cfg.d_model, cfg.d_head)
        self.W_O = param(cfg.n_heads, cfg.d_head, cfg.d_model)

        self.W_in = param(cfg.d_model, cfg.d_mlp)
        self.W_out = param(cfg.d_mlp, cfg.d_model)

        self.W_U = param(cfg.d_model, cfg.d_vocab)

        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    # ------------------------------------------------------------------ fwd

    def forward(
        self,
        tokens: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """tokens: (B, n_ctx) int64 -> logits: (B, T_out, d_vocab).

        If `cache` is a dict it is filled in place with every named
        intermediate, detached, for analysis.

        `last_only=True` computes only the final position downstream of
        attention.  This is mathematically identical for that position -- with
        causal attention nothing at the last position depends on the MLP or the
        unembedding at earlier positions -- but skips two thirds of the MLP and
        unembedding work, which dominate the cost.  Used for training; the full
        path is used for analysis.  `tests/test_model.py` asserts the two agree.
        """
        cfg = self.cfg

        def put(name, t):
            if cache is not None:
                cache[name] = t.detach()

        embed = self.W_E[tokens]                          # (B, T, d_model)
        resid_pre = embed + self.W_pos                    # (B, T, d_model)
        put("embed", embed)
        put("resid_pre", resid_pre)

        # ---- attention -----------------------------------------------------
        q_src = resid_pre[:, -1:] if last_only else resid_pre
        q = torch.einsum("btm,hmd->bhtd", q_src, self.W_Q)
        k = torch.einsum("btm,hmd->bhtd", resid_pre, self.W_K)
        v = torch.einsum("btm,hmd->bhtd", resid_pre, self.W_V)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (cfg.d_head ** 0.5)
        if cfg.causal and not last_only:
            # the last query row is unmasked anyway, so last_only needs no mask
            scores = scores.masked_fill(~self.causal_mask, float("-inf"))
        pattern = scores.softmax(dim=-1)                  # (B, H, Tq, Tk)
        z = torch.einsum("bhqk,bhkd->bhqd", pattern, v)   # (B, H, Tq, d_head)
        attn_out = torch.einsum("bhtd,hdm->btm", z, self.W_O)
        put("attn_scores", scores)
        put("attn_pattern", pattern)
        put("attn_z", z)
        put("attn_out", attn_out)

        resid_mid = q_src + attn_out
        put("resid_mid", resid_mid)

        # ---- MLP -----------------------------------------------------------
        pre_act = resid_mid @ self.W_in                   # (B, T, d_mlp)
        post_act = ACTS[cfg.act](pre_act)
        mlp_out = post_act @ self.W_out
        put("mlp_pre", pre_act)
        put("mlp_post", post_act)
        put("mlp_out", mlp_out)

        resid_post = resid_mid + mlp_out
        put("resid_post", resid_post)

        logits = resid_post @ self.W_U
        put("logits", logits)
        return logits

    # ------------------------------------------------------------- helpers

    def run_with_cache(self, tokens: torch.Tensor, last_only: bool = False):
        cache: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logits = self(tokens, cache=cache, last_only=last_only)
        return logits.detach(), cache

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_norm(self) -> float:
        with torch.no_grad():
            return float(torch.sqrt(sum(p.pow(2).sum() for p in self.parameters())))


def final_logits(logits: torch.Tensor, n_answer: int) -> torch.Tensor:
    """Take the last position and drop the "=" column.

    The answer is always a residue in 0..p-1, so column p (the "=" token) can
    never be correct.  Keeping it would add a meaningless direction to every
    Fourier decomposition of the logits.
    """
    return logits[:, -1, :n_answer]
