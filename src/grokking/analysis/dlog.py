"""Viewing a multiplication model in the multiplicative-character basis.

Modular multiplication looks nothing like modular addition in the ordinary
coordinate, and a Fourier analysis in that coordinate finds a dense, apparently
structureless spectrum.  But the nonzero residues mod p form a cyclic group of
order p - 1 under multiplication, so re-indexing them by discrete logarithm
turns the task into addition mod p - 1:

    a * b = g^(dlog[a] + dlog[b] mod (p-1))

`dlog_view` performs exactly that re-indexing on a trained model's activations,
producing an object the ordinary Section-5 toolkit can consume unchanged, with
n = p - 1 (which is even, hence the Nyquist basis row).

Two things this is for.

1.  *Causal* tests in the multiplicative basis -- restricted and excluded loss,
    frequency ablations -- rather than the observational spectra that have been
    reported.  The reduction itself is published; whether the circuit is
    load-bearing in that basis is what gets measured here.

2.  The absorbing element.  Zero is not in the multiplicative group: with 0
    included the structure is a monoid, not a group, and the character story
    does not apply to it.  Comparing the analysis with and without the zero row
    and column says whether an apparently dense spectrum is an artifact of
    including an element the algebra excludes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..data import ModularDataset
from ..fourier import discrete_log_table


@dataclass
class StarView:
    """A multiplication model's behaviour, re-indexed by discrete logarithm.

    The fields carry the same meaning and shapes as on `Snapshot`, with p
    replaced by n = p - 1, so the analysis functions work on it unchanged.
    """

    step: int
    model: object
    data: ModularDataset       # a synthetic (n x n) addition dataset
    logits: torch.Tensor       # (n, n, n)
    neuron_acts: torch.Tensor  # (n, n, d_mlp)
    attn: torch.Tensor
    resid_mid: torch.Tensor
    g: int
    original_p: int

    @property
    def p(self) -> int:
        return self.data.p

    def flat_logits(self) -> torch.Tensor:
        n = self.p
        return self.logits.reshape(n * n, n)

    def losses(self, logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
        n = self.p
        lg = self.flat_logits() if logits is None else logits.reshape(n * n, n)
        y = self.data.labels
        out = {}
        for name, idx in (("train", self.data.train_idx), ("test", self.data.test_idx)):
            sub, lab = lg[idx], y[idx]
            out[f"{name}_loss"] = float(
                torch.nn.functional.cross_entropy(sub.to(torch.float64), lab))
            out[f"{name}_acc"] = float((sub.argmax(-1) == lab).float().mean())
        return out


def dlog_view(snapshot) -> StarView:
    """Re-index a multiplication snapshot by discrete logarithm.

    Rows and columns for the input value 0 are dropped (0 is not in the
    multiplicative group), as is the output class 0, which can only be produced
    by an input containing 0.
    """
    p = snapshot.p
    dlog, exp_table, g = discrete_log_table(p)
    n = p - 1
    e = exp_table                                  # (n,) values, indexed by exponent

    logits = snapshot.logits[e][:, e][:, :, e]     # (n, n, n)
    neuron = snapshot.neuron_acts[e][:, e]         # (n, n, d_mlp)
    attn = snapshot.attn[e][:, e]
    resid = snapshot.resid_mid[e][:, e]

    # Re-express the original split in the new coordinates.
    orig_train = torch.zeros(p * p, dtype=torch.bool)
    orig_train[snapshot.data.train_idx] = True
    orig_train = orig_train.view(p, p)
    star_train = orig_train[e][:, e].reshape(-1)

    labels = ((torch.arange(n).repeat_interleave(n) + torch.arange(n).repeat(n)) % n).long()
    a = torch.arange(n).repeat_interleave(n)
    b = torch.arange(n).repeat(n)
    inputs = torch.stack([a, b, torch.full((n * n,), n)], dim=1).long()

    data = ModularDataset(
        p=n, op="add(dlog)", inputs=inputs, labels=labels,
        train_idx=star_train.nonzero().flatten(),
        test_idx=(~star_train).nonzero().flatten(),
        seed=snapshot.data.seed, train_frac=float(star_train.float().mean()),
    )

    # Sanity: the re-indexed labels must match the model's own task.
    orig_labels = snapshot.data.label_grid()[e][:, e]          # (n, n) values in 1..p-1
    assert torch.equal(dlog[orig_labels].reshape(-1), labels), \
        "discrete-log re-indexing does not reproduce the task"

    return StarView(step=snapshot.step, model=snapshot.model, data=data,
                    logits=logits, neuron_acts=neuron, attn=attn,
                    resid_mid=resid, g=g, original_p=p)


def star_embedding(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """W_E re-indexed by discrete logarithm: row e is the embedding of g^e."""
    _, exp_table, _ = discrete_log_table(p)
    return W_E[exp_table]


def zero_element_report(snapshot) -> Dict[str, float]:
    """Is the absorbing element 0 handled by a separate sub-circuit?

    Checks whether the embedding of 0 is an outlier in norm, and whether any
    neurons behave like detectors for `a == 0` or `b == 0`.
    """
    p = snapshot.p
    W = snapshot.model.W_E.detach()
    norms = W[:p].norm(dim=1)
    acts = snapshot.neuron_acts                                   # (p, p, d_mlp)
    flat = acts.reshape(p * p, -1).double()
    flat = flat - flat.mean(dim=0, keepdim=True)

    a_is_zero = torch.zeros(p, p); a_is_zero[0, :] = 1.0
    b_is_zero = torch.zeros(p, p); b_is_zero[:, 0] = 1.0
    out = {"zero_row_norm": float(norms[0]),
           "mean_other_row_norm": float(norms[1:].mean()),
           "zero_row_norm_zscore": float(
               (norms[0] - norms[1:].mean()) / (norms[1:].std() + 1e-12))}
    for name, ind in (("a_is_zero", a_is_zero), ("b_is_zero", b_is_zero)):
        x = (ind.reshape(-1).double() - ind.mean())
        x = x / (x.norm() + 1e-30)
        r = (flat * x[:, None]).sum(dim=0) / (flat.norm(dim=0) + 1e-30)
        out[f"max_neuron_corr_{name}"] = float(r.abs().max())
        out[f"n_neurons_corr_{name}_above_0.5"] = int((r.abs() > 0.5).sum())
    return out
