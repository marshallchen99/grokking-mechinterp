"""Tests of the *shape* of what the network computes.

Two questions, both sharp enough to fail:

1.  Do the logits depend on the inputs only through (a + b) mod p?  A network
    that memorised its training set has no reason to; a network that has found
    the rule has no choice.  `additive_structure` measures this against
    controls -- (a - b), a alone, b alone -- so a high number means something.

2.  Within a single frequency, is the dependence on (a, b) really of the form
    cos(w(a+b)) and sin(w(a+b)), rather than cos(w(a-b)) and sin(w(a-b))?
    This is where the trigonometric identity lives, and it is testable exactly:

        cos(wa)cos(wb) = [cos(w(a+b)) + cos(w(a-b))] / 2
        sin(wa)sin(wb) = [cos(w(a-b)) - cos(w(a+b))] / 2
        sin(wa)cos(wb) = [sin(w(a+b)) + sin(w(a-b))] / 2
        cos(wa)sin(wb) = [sin(w(a+b)) - sin(w(a-b))] / 2

    so a function whose frequency-k 2x2 coefficient block is (cc, cs, sc, ss)
    decomposes as

        amplitude of cos(w(a+b)) = (cc - ss) / 2
        amplitude of cos(w(a-b)) = (cc + ss) / 2
        amplitude of sin(w(a+b)) = (sc + cs) / 2
        amplitude of sin(w(a-b)) = (sc - cs) / 2

    For modular addition the two "minus" amplitudes should be essentially zero.
    `trig_identity_report` reports exactly that split.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..fourier import fourier_1d, fourier_2d
from .spectra import block_indices


# -------------------------------------------------- does it depend on a+b?

def _index_grids(p: int) -> Dict[str, torch.Tensor]:
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return {
        "a+b": ((a + b) % p).view(p, p),
        "a-b": ((a - b) % p).view(p, p),
        "a": a.view(p, p),
        "b": b.view(p, p),
        "a*b": ((a * b) % p).view(p, p),
    }


def _explained_by(grid: torch.Tensor, index_grid: torch.Tensor, p: int) -> float:
    """Fraction of variance of `grid` explained by the best function of `index_grid`.

    The best predictor is the group mean, so this is a one-way ANOVA: the
    between-group sum of squares over the total sum of squares.
    """
    flat = grid.reshape(p * p, -1).double()
    idx = index_grid.reshape(-1)
    n_groups = int(idx.max()) + 1

    sums = torch.zeros(n_groups, flat.shape[1], dtype=flat.dtype)
    sums.index_add_(0, idx, flat)
    counts = torch.zeros(n_groups, dtype=flat.dtype)
    counts.index_add_(0, idx, torch.ones(p * p, dtype=flat.dtype))
    means = sums / counts[:, None].clamp(min=1)

    fitted = means[idx]
    ss_tot = (flat - flat.mean(dim=0, keepdim=True)).pow(2).sum()
    ss_res = (flat - fitted).pow(2).sum()
    return float(1.0 - ss_res / ss_tot) if float(ss_tot) > 0 else 0.0


def additive_structure(grid: torch.Tensor, p: int,
                       controls: Optional[List[str]] = None) -> Dict[str, float]:
    """How much of `grid` is a function of (a+b), versus of the controls?

    `grid` is (p, p) or (p, p, k) -- e.g. the logits for every answer.
    """
    names = controls if controls is not None else ["a+b", "a-b", "a", "b", "a*b"]
    grids = _index_grids(p)
    return {n: _explained_by(grid, grids[n], p) for n in names}


# ------------------------------------------------ is it the trig identity?

def trig_identity_report(logits: torch.Tensor, F: torch.Tensor, p: int,
                         freqs: List[int]) -> Dict[str, object]:
    """Split each frequency's energy into (a+b) and (a-b) parts.

    Returns, per frequency:
        sum_frac        fraction of that frequency's energy in cos/sin(w(a+b))
        amp_cos, amp_sin  (p,) amplitudes as a function of the answer c
        readout_frac    how much of amp_cos(c), amp_sin(c) is itself a wave at
                        frequency k -- i.e. whether the unembedding is really
                        computing cos(w(a + b - c))
    """
    coeffs = fourier_2d(logits.to(F.dtype), F)          # (p, p, p) over (a, b, c)
    out: Dict[str, object] = {"per_freq": {}}
    for k in freqs:
        ck, sk = block_indices(k, p)
        if ck == sk:          # Nyquist has no sine partner: skip it here
            continue
        cc, cs = coeffs[ck, ck], coeffs[ck, sk]         # each (p,) over c
        sc, ss = coeffs[sk, ck], coeffs[sk, sk]

        amp_cos_sum = (cc - ss) / 2                     # cos(w(a+b))
        amp_sin_sum = (sc + cs) / 2                     # sin(w(a+b))
        amp_cos_dif = (cc + ss) / 2                     # cos(w(a-b))
        amp_sin_dif = (sc - cs) / 2                     # sin(w(a-b))

        e_sum = float(amp_cos_sum.pow(2).sum() + amp_sin_sum.pow(2).sum())
        e_dif = float(amp_cos_dif.pow(2).sum() + amp_sin_dif.pow(2).sum())

        # Is the dependence on the answer c also a wave at frequency k?
        stacked = torch.stack([amp_cos_sum, amp_sin_sum])          # (2, p)
        cf = fourier_1d(stacked, F, dim=1)                         # (2, p)
        e_tot = float(cf.pow(2).sum())
        e_at_k = float(cf[:, ck].pow(2).sum() + cf[:, sk].pow(2).sum())

        out["per_freq"][k] = {
            "sum_frac": e_sum / (e_sum + e_dif) if (e_sum + e_dif) > 0 else 0.0,
            "energy_sum": e_sum,
            "energy_diff": e_dif,
            "readout_frac": e_at_k / e_tot if e_tot > 0 else 0.0,
            "amp_cos": amp_cos_sum,
            "amp_sin": amp_sin_sum,
        }

    fr = [v["sum_frac"] for v in out["per_freq"].values()]
    rd = [v["readout_frac"] for v in out["per_freq"].values()]
    out["mean_sum_frac"] = sum(fr) / len(fr) if fr else 0.0
    out["mean_readout_frac"] = sum(rd) / len(rd) if rd else 0.0
    return out


def readout_budget(logits: torch.Tensor, F: torch.Tensor, p: int, freqs: List[int],
                   sign: int = +1) -> Dict[str, float]:
    """Where the third step of the algorithm's energy actually goes.

    `trig_identity_report` checks the first two steps -- that each frequency's
    dependence on (a, b) is a function of (a+b) rather than (a-b).  The third
    step is the readout: the claim that the amplitudes A(c) and B(c) of
    cos(w(a+b)) and sin(w(a+b)) are themselves waves at the same frequency in
    the answer c, which is what makes the sum a matched filter peaked at
    c = a + b.

    Reporting a single "fraction at frequency k" for that hides what the rest
    is.  It splits cleanly into three named things:

        own        the wave the algorithm predicts
        dc         a constant offset in A(c) or B(c), which is a per-answer
                   logit bias rather than a failure of the wave structure
        cross      energy at a DIFFERENT key frequency -- the circuits for two
                   frequencies interfering with each other

    and whatever is left over. `sign=-1` reads the (a-b) amplitudes instead,
    which is the direction a subtraction model uses; reading a model in the
    wrong direction returns noise, and that is itself diagnostic.
    """
    from .spectra import block_indices
    from ..fourier import frequency_of_index

    C = fourier_2d(logits.to(F.dtype), F)
    keys = set(freqs)
    tot = own = dc = cross = 0.0
    for k in freqs:
        ck, sk = block_indices(k, p)
        if ck == sk:                      # Nyquist has no sine partner
            continue
        cc, cs, sc, ss = C[ck, ck], C[ck, sk], C[sk, ck], C[sk, sk]
        if sign > 0:
            A, B = (cc - ss) / 2, (sc + cs) / 2
        else:
            A, B = (cc + ss) / 2, (sc - cs) / 2
        e = fourier_1d(torch.stack([A, B]), F, dim=1).pow(2).sum(0)
        tot += float(e.sum())
        own += float(e[ck] + e[sk])
        dc += float(e[0])
        for i in range(1, p):
            f = frequency_of_index(i)
            if f in keys and f != k:
                cross += float(e[i])
    if tot <= 0:
        return {"own": 0.0, "dc": 0.0, "cross": 0.0, "unexplained": 1.0}
    return {"own": own / tot, "dc": dc / tot, "cross": cross / tot,
            "unexplained": max(0.0, 1.0 - (own + dc + cross) / tot)}
