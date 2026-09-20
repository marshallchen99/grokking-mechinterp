"""Causal tests: edit the model, then see whether it still works.

Everything in `spectra.py` and `structure.py` is correlational -- it shows that
the trained network's weights and activations *look like* the claimed
algorithm.  That is not yet evidence that the network is *using* it.  The
difference matters: a component can be highly structured and still be
irrelevant to the output.

These functions close that gap by intervening.  Each one produces an edited
copy of the model, runs it on the whole input table, and reports the damage.
Two predictions are on the line:

    keeping only the key frequencies      -> performance should survive
    deleting only the key frequencies     -> performance should collapse

If both hold, the key frequencies are not merely present, they are load-bearing.
Note these edit the *weights*, not the logits, so the intervention propagates
through the rest of the network exactly as a real change would.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as tF

from ..fourier import fourier_1d
from ..model import OneLayerTransformer, final_logits
from .spectra import block_indices


def _clone(model: OneLayerTransformer) -> OneLayerTransformer:
    m = copy.deepcopy(model)
    m.eval()
    return m


@torch.no_grad()
def evaluate_model(model: OneLayerTransformer, data) -> Dict[str, float]:
    """Loss and accuracy on train and test for a (possibly edited) model."""
    p = data.p
    logits = final_logits(model(data.inputs, last_only=True), p)
    out = {}
    for name, idx in (("train", data.train_idx), ("test", data.test_idx)):
        sub, lab = logits[idx], data.labels[idx]
        out[f"{name}_loss"] = float(tF.cross_entropy(sub, lab))
        out[f"{name}_acc"] = float((sub.argmax(-1) == lab).float().mean())
    return out


# ------------------------------------------------------- embedding surgery

@torch.no_grad()
def ablate_embedding_frequencies(model: OneLayerTransformer, F: torch.Tensor,
                                 p: int, freqs: Sequence[int], keep: bool = True
                                 ) -> OneLayerTransformer:
    """Filter the number embeddings in the Fourier basis.

    keep=True   zero every frequency EXCEPT those listed (plus the mean)
    keep=False  zero exactly those frequencies
    """
    m = _clone(model)
    W = m.W_E.data[:p].to(F.dtype)
    coeffs = fourier_1d(W, F, dim=0)                       # (p, d_model)

    mask = torch.zeros(p, dtype=torch.bool)
    mask[0] = True                                          # constant term
    for k in freqs:
        ck, sk = block_indices(k, p)
        mask[ck] = mask[sk] = True

    if keep:
        coeffs = coeffs * mask[:, None]
    else:
        coeffs = coeffs * (~mask | (torch.arange(p) == 0))[:, None]

    m.W_E.data[:p] = (F.T @ coeffs).to(m.W_E.dtype)
    return m


# ---------------------------------------------------------- neuron surgery

@torch.no_grad()
def ablate_neurons(model: OneLayerTransformer, keep_mask: torch.Tensor,
                   mean_act: Optional[torch.Tensor] = None) -> OneLayerTransformer:
    """Remove MLP neurons by zeroing their output weights.

    If `mean_act` is given the neurons are *mean*-ablated instead of
    zero-ablated: their contribution is replaced by its average over the input
    distribution, which is the fairer intervention because it removes the
    neuron's information without also removing its constant offset.
    """
    m = _clone(model)
    drop = ~keep_mask
    if mean_act is not None:
        # fold the average contribution of the dropped neurons into W_pos, so
        # the residual stream keeps the same mean it had before
        bias = mean_act[drop] @ m.W_out.data[drop]          # (d_model,)
        m.W_pos.data[-1] += bias
    m.W_out.data[drop] = 0.0
    return m


@torch.no_grad()
def ablate_head(model: OneLayerTransformer, heads: Sequence[int]) -> OneLayerTransformer:
    """Delete attention heads by zeroing their output projections."""
    m = _clone(model)
    for h in heads:
        m.W_O.data[h] = 0.0
    return m


@torch.no_grad()
def ablate_mlp(model: OneLayerTransformer) -> OneLayerTransformer:
    """Delete the whole MLP, leaving only the attention and the direct path."""
    m = _clone(model)
    m.W_out.data.zero_()
    return m


# ------------------------------------------------------------- experiments

def frequency_ablation_study(model: OneLayerTransformer, data, F: torch.Tensor,
                             key_freqs: List[int]) -> Dict[str, Dict[str, float]]:
    """The headline causal experiment, plus the controls that make it mean something.

    A random-frequency control matters here: deleting ANY five of fifty-six
    frequencies removes some of the embedding's norm, so "performance dropped"
    is only interesting if it drops much further for the key five than for five
    arbitrary ones.
    """
    p = data.p
    all_freqs = list(range(1, p // 2 + 1))
    others = [k for k in all_freqs if k not in set(key_freqs)]

    # deterministic "random" control: evenly spaced non-key frequencies, so the
    # result does not depend on an RNG draw
    n = len(key_freqs)
    ctrl = [others[round(i * (len(others) - 1) / max(n - 1, 1))] for i in range(n)] if others else []
    ctrl = sorted(set(ctrl))

    out = {"baseline": evaluate_model(model, data)}
    out["keep_key_freqs"] = evaluate_model(
        ablate_embedding_frequencies(model, F, p, key_freqs, keep=True), data)
    out["drop_key_freqs"] = evaluate_model(
        ablate_embedding_frequencies(model, F, p, key_freqs, keep=False), data)
    out["drop_control_freqs"] = evaluate_model(
        ablate_embedding_frequencies(model, F, p, ctrl, keep=False), data)
    out["keep_control_freqs"] = evaluate_model(
        ablate_embedding_frequencies(model, F, p, ctrl, keep=True), data)
    out["_control_freqs"] = {"freqs": ctrl}
    return out


def neuron_ablation_study(model: OneLayerTransformer, data, dominant_freq: torch.Tensor,
                          dominant_frac: torch.Tensor, key_freqs: List[int],
                          neuron_acts: Optional[torch.Tensor] = None,
                          frac_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """Keep only the neurons that are well explained by a key frequency."""
    key = torch.tensor(sorted(key_freqs))
    on_key = torch.isin(dominant_freq, key) & (dominant_frac > frac_threshold)

    mean_act = None
    if neuron_acts is not None:
        mean_act = neuron_acts.reshape(-1, neuron_acts.shape[-1]).mean(dim=0)

    n_total = int(on_key.numel())
    n_keep = int(on_key.sum())
    out = {
        "baseline": evaluate_model(model, data),
        "keep_key_freq_neurons": evaluate_model(ablate_neurons(model, on_key, mean_act), data),
        "drop_key_freq_neurons": evaluate_model(ablate_neurons(model, ~on_key, mean_act), data),
        "_counts": {"kept": n_keep, "total": n_total, "frac": n_keep / n_total},
    }

    # control: keep the same NUMBER of neurons, chosen by activation norm rather
    # than by frequency, so the comparison is not just "fewer neurons is worse"
    if neuron_acts is not None and 0 < n_keep < n_total:
        norms = neuron_acts.reshape(-1, n_total).std(dim=0)
        top = torch.zeros(n_total, dtype=torch.bool)
        top[norms.argsort(descending=True)[:n_keep]] = True
        out["keep_same_count_by_norm"] = evaluate_model(ablate_neurons(model, top, mean_act), data)
    return out


def component_ablation_study(model: OneLayerTransformer, data) -> Dict[str, Dict[str, float]]:
    """How much does each coarse component matter?"""
    n_heads = model.cfg.n_heads
    out = {"baseline": evaluate_model(model, data),
           "no_mlp": evaluate_model(ablate_mlp(model), data)}
    for h in range(n_heads):
        out[f"no_head_{h}"] = evaluate_model(ablate_head(model, [h]), data)
    out["no_attention"] = evaluate_model(ablate_head(model, list(range(n_heads))), data)
    return out
