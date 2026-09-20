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


def _loss_acc(logits: torch.Tensor, data, idx: Optional[torch.Tensor]) -> Dict[str, float]:
    """Loss and accuracy of edited logits, in float64.

    Restricted loss reaches the 1e-4 range and below once the circuit is
    formed, which float32 cross-entropy cannot resolve; the whole measure is
    about small numbers, so it is computed in double throughout.
    """
    p = data.p
    flat = logits.reshape(p * p, p).to(torch.float64)
    if idx is None:
        sub, lab = flat, data.labels
    else:
        sub, lab = flat[idx], data.labels[idx]
    return {"loss": float(tF.cross_entropy(sub, lab)),
            "acc": float((sub.argmax(-1) == lab).float().mean())}


def _split_idx(snapshot, split: str):
    if split == "train":
        return snapshot.data.train_idx
    if split == "test":
        return snapshot.data.test_idx
    if split == "all":
        return None
    raise ValueError(f"unknown split {split!r}")


def restricted_loss(snapshot, F: torch.Tensor, freqs: List[int],
                    mode: str = "sum", split: str = "all") -> Dict[str, float]:
    edited = filter_logits(snapshot.logits, F, snapshot.p, freqs, keep=True, mode=mode)
    return _loss_acc(edited, snapshot.data, _split_idx(snapshot, split))


def excluded_loss(snapshot, F: torch.Tensor, freqs: List[int],
                  mode: str = "sum", split: str = "train") -> Dict[str, float]:
    edited = filter_logits(snapshot.logits, F, snapshot.p, freqs, keep=False, mode=mode)
    return _loss_acc(edited, snapshot.data, _split_idx(snapshot, split))


def per_frequency_excluded_loss(snapshot, F: torch.Tensor,
                                split: str = "train") -> Dict[int, float]:
    """Excluded loss with a single frequency removed, for every frequency.

    Useful because it needs no key-frequency identification at all: whichever
    frequencies the model is relying on will stand out as the ones whose
    removal hurts.  That makes it the right tool for runs whose key set is
    ambiguous, and an independent check on the key set for runs where it is not.
    """
    p = snapshot.p
    idx = _split_idx(snapshot, split)
    out = {}
    for k in range(1, (p - 1) // 2 + 1):
        edited = filter_logits(snapshot.logits, F, p, [k], keep=False, mode="sum")
        out[k] = _loss_acc(edited, snapshot.data, idx)["loss"]
    return out


def progress_measures(snapshot, F: torch.Tensor, freqs: List[int],
                      emb_spectrum=None) -> Dict[str, float]:
    """Every scalar we track along the trajectory, for one checkpoint."""
    base = snapshot.losses()
    out = {"step": snapshot.step, **base}
    # "sum" is the strict form of the algorithmic claim (2 directions per
    # frequency); "block" also keeps the (a-b) content, so it is a looser
    # upper bound.  The literature is inconsistent about which it used, so
    # both are reported rather than one being silently chosen.
    for mode in ("sum", "block"):
        for split in ("all", "train"):
            r = restricted_loss(snapshot, F, freqs, mode=mode, split=split)
            out[f"restricted_loss_{mode}_{split}"] = r["loss"]
            out[f"restricted_acc_{mode}_{split}"] = r["acc"]
        e = excluded_loss(snapshot, F, freqs, mode=mode, split="train")
        out[f"excluded_loss_{mode}"] = e["loss"]
        out[f"excluded_acc_{mode}"] = e["acc"]
    # back-compat aliases used by the figure code
    out["restricted_loss_block"] = out["restricted_loss_block_all"]
    out["restricted_loss_sum"] = out["restricted_loss_sum_all"]
    out["weight_norm"] = snapshot.model.param_norm()
    with torch.no_grad():
        out["sum_sq_weights"] = float(sum(q.pow(2).sum() for q in snapshot.model.parameters()))
    if emb_spectrum is not None:
        out["emb_gini"] = emb_spectrum.gini
        out["emb_key_frac"] = emb_spectrum.sparsity_report()["frac_power_in_key_freqs"]
    return out
