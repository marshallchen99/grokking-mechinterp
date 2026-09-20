"""Fourier spectra of the model's weights and activations.

The central claim to be tested is that the trained network represents each
input number as a point on a circle, using only a handful of angular
frequencies.  If that is true, then the Fourier transform of the embedding
matrix -- and of almost everything downstream of it -- should be *sparse*:
nearly all of its energy sitting on a few frequencies, and nearly none on the
other fifty.  That is a strong, falsifiable prediction, and these are the
instruments for checking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..fourier import fourier_1d, fourier_2d, index_frequencies, gini


def block_indices(k: int, p: Optional[int] = None) -> Tuple[int, int]:
    """Basis indices (cos k, sin k) for frequency k.

    Over Z_p the distinct frequencies are 1 .. (p-1)/2; above that they alias
    back onto lower ones, so asking for one is always a mistake rather than a
    harmless no-op.
    """
    if k < 1:
        raise ValueError("frequency 0 is the constant term and has no sin/cos pair")
    if p is not None and k > p // 2:
        raise ValueError(
            f"frequency {k} does not exist for n={p}; the distinct frequencies "
            f"are 1..{p // 2} (higher ones alias back onto these)"
        )
    if p is not None and p % 2 == 0 and k == p // 2:
        # Nyquist: cos(pi x) = (-1)^x has no sine partner, so the "block" for
        # this frequency is one index, not two.  Callers deduplicate.
        return p - 1, p - 1
    return 2 * k - 1, 2 * k


# ------------------------------------------------------------------ weights

@dataclass
class Spectrum:
    coeffs: torch.Tensor        # (p, d) Fourier coefficients per basis index
    norm_per_index: torch.Tensor  # (p,)
    power_per_freq: torch.Tensor  # (n_freq,) index 0 is the constant term
    key_freqs: List[int]
    gini: float

    def sparsity_report(self) -> Dict[str, float]:
        pw = self.power_per_freq[1:]           # drop the constant
        top = pw[torch.tensor(self.key_freqs) - 1].sum() if self.key_freqs else torch.tensor(0.0)
        return {
            "n_key_freqs": len(self.key_freqs),
            "frac_power_in_key_freqs": float(top / pw.sum()) if float(pw.sum()) else 0.0,
            "gini": self.gini,
        }


def embedding_spectrum(W_E: torch.Tensor, F: torch.Tensor, p: int,
                       **key_kwargs) -> Spectrum:
    """Fourier-transform the number embeddings (rows 0..p-1 of W_E).

    Row p of W_E is the "=" token, which is not a number and has no place in a
    basis over Z_p, so it is dropped.
    """
    emb = W_E[:p].to(F.dtype)
    coeffs = fourier_1d(emb, F, dim=0)               # (p, d_model)
    norm_per_index = coeffs.norm(dim=1)
    power = _group_power(norm_per_index.pow(2), p)
    keys = key_frequencies(power, **key_kwargs)
    return Spectrum(coeffs=coeffs, norm_per_index=norm_per_index,
                    power_per_freq=power, key_freqs=keys,
                    gini=float(gini(power[1:])))


def _group_power(per_index: torch.Tensor, p: int) -> torch.Tensor:
    """Sum a per-basis-index quantity into per-frequency buckets."""
    freqs = index_frequencies(p)
    out = torch.zeros(p // 2 + 1, dtype=per_index.dtype)
    out.index_add_(0, freqs, per_index)
    return out


def key_frequencies(power_per_freq: torch.Tensor, method: str = "gap",
                    k: int = 5, z: float = 3.0,
                    min_frac: float = 0.01) -> List[int]:
    """Which frequencies is the model actually using?

    Three selection rules, because "the model uses 5 frequencies" should not
    depend on a threshold chosen after seeing the answer:

    gap        sort frequencies by power, cut at the largest multiplicative gap
               between consecutive values.  Parameter-free.
    topk       take the k strongest.  Only for reproducing a published count.
    zscore     take everything more than `z` robust standard deviations above
               the median.

    Frequencies carrying less than `min_frac` of the total are never selected,
    which keeps the gap rule from finding a "gap" inside the noise floor of an
    untrained model.
    """
    pw = power_per_freq[1:].clone()                 # frequencies 1 .. (p-1)/2
    total = float(pw.sum())
    if total <= 0:
        return []
    order = torch.argsort(pw, descending=True)
    sorted_pw = pw[order]

    if method == "topk":
        chosen = order[:k]
    elif method == "zscore":
        med = sorted_pw.median()
        mad = (pw - med).abs().median() * 1.4826 + 1e-30
        chosen = order[(sorted_pw - med) / mad > z]
    elif method == "gap":
        # largest ratio between consecutive sorted powers, searched only over
        # the head of the distribution (a gap in the tail is just noise)
        head = min(len(sorted_pw) - 1, max(2, len(sorted_pw) // 2))
        ratios = sorted_pw[:head] / (sorted_pw[1:head + 1] + 1e-30)
        cut = int(torch.argmax(ratios)) + 1
        chosen = order[:cut]
    else:
        raise ValueError(f"unknown method {method!r}")

    keys = [int(i) + 1 for i in chosen if float(pw[i]) / total >= min_frac]
    return sorted(keys)


# -------------------------------------------------------------- activations

def freq_block_power(coeffs2d: torch.Tensor, k: int, p: Optional[int] = None) -> torch.Tensor:
    """Power of the 2D coefficients belonging purely to frequency k.

    That is the 3x3 block {const, cos k, sin k} x {const, cos k, sin k}, minus
    the constant-constant term (which is the overall mean and belongs to no
    frequency).
    """
    ck, sk = block_indices(k, p)
    idx = torch.tensor(sorted({0, ck, sk}))
    block = coeffs2d[idx][:, idx]
    total = block.pow(2).sum(dim=(0, 1))
    const = coeffs2d[0, 0].pow(2)
    return total - const


def neuron_frequencies(neuron_acts: torch.Tensor, F: torch.Tensor, p: int
                       ) -> Dict[str, torch.Tensor]:
    """Assign each MLP neuron to the frequency that explains it best.

    `neuron_acts` is (p, p, d_mlp): the post-activation value of every neuron
    for every input pair.  If the "each neuron is a wave at one frequency"
    story is right, one frequency should account for nearly all of a neuron's
    variance.
    """
    acts = neuron_acts.to(F.dtype)
    coeffs = fourier_2d(acts, F)                                 # (p, p, d_mlp)
    total = coeffs.pow(2).sum(dim=(0, 1)) - coeffs[0, 0].pow(2)  # variance, minus mean
    n_freq = p // 2
    per_freq = torch.stack([freq_block_power(coeffs, k, p) for k in range(1, n_freq + 1)])
    frac = per_freq / (total + 1e-30)
    best = frac.argmax(dim=0) + 1
    return {
        "coeffs": coeffs,
        "power_per_freq": per_freq,          # (n_freq, d_mlp)
        "explained_frac": frac,              # (n_freq, d_mlp)
        "dominant_freq": best,               # (d_mlp,)
        "dominant_frac": frac.max(dim=0).values,
        "total_variance": total,
    }


def logit_frequency_power(logits: torch.Tensor, F: torch.Tensor, p: int
                          ) -> Dict[str, torch.Tensor]:
    """Per-frequency power of the logits viewed as a function of (a, b)."""
    coeffs = fourier_2d(logits.to(F.dtype), F)                  # (p, p, p)
    total = coeffs.pow(2).sum(dim=(0, 1)) - coeffs[0, 0].pow(2)
    n_freq = p // 2
    per_freq = torch.stack([freq_block_power(coeffs, k, p) for k in range(1, n_freq + 1)])
    return {"coeffs": coeffs, "power_per_freq": per_freq, "total": total}


# ------------------------------------------- key frequencies, three ways
#
# Which frequencies is the model using?  The answer drives every progress
# measure and every ablation, so it must not depend on a threshold chosen
# after seeing the result.  Three rules with independent logic are computed;
# agreement between them is itself reported, and disagreement is reported
# rather than resolved by picking a favourite.

def key_freqs_rule_a(W_E: torch.Tensor, F: torch.Tensor, p: int,
                     ratio: float = 4.0) -> Dict[str, object]:
    """Embedding-norm threshold: keep basis indices whose norm exceeds max/ratio.

    Operates on per-*index* norms rather than per-frequency power, so a
    frequency counts if either its cosine or its sine component is large.
    """
    coeffs = fourier_1d(W_E[:p].to(F.dtype), F, dim=0)
    norms = coeffs.norm(dim=1)
    m = float(norms[1:].max())
    idx = [i for i in range(1, p) if float(norms[i]) > m / ratio]
    freqs = sorted({(i + 1) // 2 for i in idx})
    return {"freqs": freqs, "indices": idx, "max_norm": m, "norms": norms}


def key_freqs_rule_b(neuron_acts: torch.Tensor, F: torch.Tensor, p: int,
                     threshold: float = 0.85) -> Dict[str, object]:
    """Neuron clustering: frequencies that at least one neuron is dedicated to.

    A neuron counts as dedicated to frequency k if the 8 two-dimensional basis
    cells belonging to k explain more than `threshold` of its variance.  The
    activations are centred first; without that the constant term dominates
    every frequency's share and every neuron looks like frequency 1.
    """
    nf = neuron_frequencies(neuron_acts, F, p)
    dom, frac = nf["dominant_freq"], nf["dominant_frac"]
    above = frac > threshold
    return {
        "freqs": sorted({int(k) for k in dom[above].tolist()}),
        "freqs_unrestricted": sorted({int(k) for k in dom.tolist()}),
        "n_above": int(above.sum()),
        "n_total": int(dom.numel()),
        "frac_above": float(above.float().mean()),
        "mean_domfrac": float(frac.mean()),
        "dominant_freq": dom,
        "dominant_frac": frac,
    }


def key_freqs_consensus(W_E: torch.Tensor, neuron_acts: torch.Tensor,
                        F: torch.Tensor, p: int) -> Dict[str, object]:
    """Run all three rules and report whether they agree."""
    a = key_freqs_rule_a(W_E, F, p)
    b = key_freqs_rule_b(neuron_acts, F, p)
    spec = embedding_spectrum(W_E, F, p, method="gap")
    c = spec.key_freqs

    sets = {"rule_a_embedding_norm": set(a["freqs"]),
            "rule_b_neuron_cluster": set(b["freqs"]),
            "rule_c_power_gap": set(c)}
    union = set().union(*sets.values())
    inter = set.intersection(*sets.values()) if union else set()
    jaccard = len(inter) / len(union) if union else 0.0
    return {
        "by_rule": {k: sorted(v) for k, v in sets.items()},
        "consensus": sorted(inter),
        "union": sorted(union),
        "jaccard": jaccard,
        "agree": jaccard == 1.0,
        "rule_a": {k: v for k, v in a.items() if k != "norms"},
        "rule_b": {k: v for k, v in b.items()
                   if k not in ("dominant_freq", "dominant_frac")},
        "spectrum": spec,
    }
