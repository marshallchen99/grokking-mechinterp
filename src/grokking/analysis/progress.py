"""Progress measures: metrics that see inside the plateau.

The whole difficulty of grokking is that every externally visible number --
train loss, train accuracy, test loss, test accuracy -- is flat for tens of
thousands of steps while something decisive happens inside.  A progress measure
is a metric computed from the model's internals that moves during that flat
stretch.

Two of them, both defined by editing the logits in the Fourier basis and
re-measuring the loss:

restricted loss
    Keep ONLY the components the claimed algorithm is supposed to use, throw
    away everything else, and see how well the model still does.  If it does
    well, those components are sufficient.  Crucially, this can fall long
    before test accuracy moves -- the generalising machinery is already there,
    buried under the memorised component.

excluded loss
    The complement: delete exactly those components and keep the rest.  This
    measures how much the model still leans on everything *other* than the
    claimed algorithm, so it rises as memorisation is cleaned up.

Both are computed on the training set, because the interesting claim is about
what the model is doing on data it has already fit.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as tF

from ..fourier import fourier_2d, inverse_fourier_2d, gini
from .spectra import block_indices


def _freq_block_mask(p: int, freqs: List[int], include_direct: bool) -> torch.Tensor:
    """(p, p) bool mask over 2D basis indices for the given frequencies."""
    mask = torch.zeros(p, p, dtype=torch.bool)
    mask[0, 0] = True                                   # the overall mean
    for k in freqs:
        ck, sk = block_indices(k, p)
        idx = torch.tensor([ck, sk])
        mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True  # the 2x2 product block
        if include_direct:
            mask[idx, 0] = True                          # f(a) alone
            mask[0, idx] = True                          # f(b) alone
    return mask


def _keep_sum_directions(coeffs: torch.Tensor, p: int, freqs: List[int]) -> torch.Tensor:
    """Zero everything except the cos/sin(w(a+b)) parts of the given frequencies.

    Within frequency k's 2x2 block (cc, cs, sc, ss), the (a+b) part has
    amplitudes A = (cc - ss)/2 and B = (sc + cs)/2, and re-expressing *only*
    that part in the product basis gives (A, B, B, -A).
    """
    out = torch.zeros_like(coeffs)
    out[0, 0] = coeffs[0, 0]
    for k in freqs:
        ck, sk = block_indices(k, p)
        cc, cs, sc, ss = coeffs[ck, ck], coeffs[ck, sk], coeffs[sk, ck], coeffs[sk, sk]
        A = (cc - ss) / 2
        B = (sc + cs) / 2
        out[ck, ck], out[sk, sk] = A, -A
        out[sk, ck], out[ck, sk] = B, B
    return out


def filter_logits(logits: torch.Tensor, F: torch.Tensor, p: int, freqs: List[int],
                  keep: bool = True, mode: str = "block",
                  include_direct: bool = True) -> torch.Tensor:
    """Edit the logits in the 2D Fourier basis over (a, b).

    keep=True   -> restricted: retain only the named components
    keep=False  -> excluded:   delete exactly the named components
    mode="block"  the full 2x2 product block per frequency (plus, optionally,
                  the terms depending on a or b alone)
    mode="sum"    only the cos/sin(w(a+b)) directions -- the strict form of the
                  algorithmic claim
    """
    coeffs = fourier_2d(logits.to(F.dtype), F)
    if mode == "sum":
        kept = _keep_sum_directions(coeffs, p, freqs)
    elif mode == "block":
        mask = _freq_block_mask(p, freqs, include_direct)
        kept = coeffs * mask.unsqueeze(-1)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    new = kept if keep else (coeffs - kept + (coeffs[0, 0] * 0))
    if not keep:
        new = coeffs - kept
        new[0, 0] = coeffs[0, 0]        # always keep the mean; removing it is meaningless
    return inverse_fourier_2d(new, F)


def _loss_acc(logits: torch.Tensor, data, idx: torch.Tensor) -> Dict[str, float]:
    p = data.p
    flat = logits.reshape(p * p, p).float()
    sub, lab = flat[idx], data.labels[idx]
    return {"loss": float(tF.cross_entropy(sub, lab)),
            "acc": float((sub.argmax(-1) == lab).float().mean())}


def restricted_loss(snapshot, F: torch.Tensor, freqs: List[int],
                    mode: str = "block", split: str = "train") -> Dict[str, float]:
    edited = filter_logits(snapshot.logits, F, snapshot.p, freqs, keep=True, mode=mode)
    idx = snapshot.data.train_idx if split == "train" else snapshot.data.test_idx
    return _loss_acc(edited, snapshot.data, idx)


def excluded_loss(snapshot, F: torch.Tensor, freqs: List[int],
                  mode: str = "block", split: str = "train") -> Dict[str, float]:
    edited = filter_logits(snapshot.logits, F, snapshot.p, freqs, keep=False, mode=mode)
    idx = snapshot.data.train_idx if split == "train" else snapshot.data.test_idx
    return _loss_acc(edited, snapshot.data, idx)


def progress_measures(snapshot, F: torch.Tensor, freqs: List[int],
                      emb_spectrum=None) -> Dict[str, float]:
    """Every scalar we track along the trajectory, for one checkpoint."""
    base = snapshot.losses()
    out = {"step": snapshot.step, **base}
    for mode in ("block", "sum"):
        r = restricted_loss(snapshot, F, freqs, mode=mode)
        e = excluded_loss(snapshot, F, freqs, mode=mode)
        out[f"restricted_loss_{mode}"] = r["loss"]
        out[f"restricted_acc_{mode}"] = r["acc"]
        out[f"excluded_loss_{mode}"] = e["loss"]
        out[f"excluded_acc_{mode}"] = e["acc"]
    out["weight_norm"] = snapshot.model.param_norm()
    if emb_spectrum is not None:
        out["emb_gini"] = emb_spectrum.gini
        out["emb_key_frac"] = emb_spectrum.sparsity_report()["frac_power_in_key_freqs"]
    return out
