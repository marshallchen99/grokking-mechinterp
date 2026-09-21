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



# ----------------------------------------------------- weight-level surgery

@torch.no_grad()
def ablate_dlog_frequencies(model, p: int, freqs, keep: bool = True):
    """Edit the embedding in the multiplicative-character basis.

    The nonzero residues' embedding rows are re-indexed by discrete logarithm,
    transformed in the Fourier basis over Z_{p-1}, filtered, transformed back
    and written into the same rows.  The rows for 0 (not in the group) and for
    "=" are left untouched.  This is a genuine intervention: the edited network
    is re-run end to end, unlike a projection of its output logits.
    """
    import copy

    from ..fourier import fourier_1d, make_fourier_basis
    from .spectra import block_indices

    m = copy.deepcopy(model)
    m.eval()
    _, exp_table, _ = discrete_log_table(p)
    n = p - 1
    Fn, _ = make_fourier_basis(n, dtype=torch.float64)
    rows = m.W_E.data[exp_table].to(torch.float64)        # (n, d_model), by exponent
    coeffs = fourier_1d(rows, Fn, dim=0)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[0] = True                                          # the mean
    for k in freqs:
        ck, sk = block_indices(k, n)
        mask[ck] = mask[sk] = True
    if not keep:
        mask = ~mask
        mask[0] = True
    coeffs = coeffs * mask[:, None]
    m.W_E.data[exp_table] = (Fn.T @ coeffs).to(m.W_E.dtype)
    return m


def zero_element_interventions(model, data, n_random: int = 20,
                               n_subset: int = 8, scale: float = 10.0) -> Dict[str, object]:
    """Is the small norm of 0's embedding what makes the model output 0?

    Accuracy on the pairs containing a zero, after each edit to row 0 alone.
    If shrinking the row were the mechanism, restoring its norm should break it.
    """
    import copy

    from ..model import final_logits

    p = data.p
    W = model.W_E.detach()
    norms = W[1:p].norm(dim=1)
    mean_norm = float(norms.mean())
    lab = data.labels.view(p, p)
    zero = torch.zeros(p, p, dtype=torch.bool)
    zero[0, :] = True
    zero[:, 0] = True

    def acc_after(edit):
        m = copy.deepcopy(model)
        m.eval()
        with torch.no_grad():
            edit(m.W_E.data)
            pred = final_logits(m(data.inputs, last_only=True), p).argmax(-1).view(p, p)
        return float((pred == lab)[zero].float().mean())

    # One random direction is one draw from a wide spread (a single draw has
    # given 0.20 and 0.99 in two different hands), so report the distribution.
    spread = []
    for sd in range(n_random):
        g = torch.Generator().manual_seed(sd)
        v = torch.randn(W.shape[1], generator=g, dtype=W.dtype)
        v = v / v.norm() * mean_norm
        spread.append(acc_after(lambda w, v=v: w[0].copy_(v)))
    spread.sort()
    r0 = W[0].clone()
    out = {
        "unedited": acc_after(lambda w: None),
        "row0_zeroed": acc_after(lambda w: w[0].zero_()),
        "row0_rescaled_to_mean_norm": acc_after(lambda w: w[0].mul_(mean_norm / float(r0.norm()))),
        "row0_doubled": acc_after(lambda w: w[0].mul_(2.0)),
        "row0_replaced_by_mean_of_others": acc_after(lambda w: w[0].copy_(W[1:p].mean(0))),
        "row0_scaled": acc_after(lambda w: w[0].mul_(scale)),
    }
    # What does the model answer when an input carries no multiplicative signal?
    # Measured on inputs that are NOT 0, so the answer cannot come from 0's own
    # embedding.  Two design points matter, both learned from a review:
    #   * a single-input edit must be scored only on pairs where the OTHER input
    #     is untouched -- (a, a) has both inputs edited and belongs to the
    #     two-input case;
    #   * the two-input case must edit a SUBSET of rows.  Blanking every row makes
    #     every input identical, so "every pair gets the same answer" would hold
    #     by construction and measure nothing.
    def predictions_after(edit):
        m = copy.deepcopy(model)
        m.eval()
        with torch.no_grad():
            edit(m.W_E.data)
            return final_logits(m(data.inputs, last_only=True), p).argmax(-1).view(p, p)

    mean_row = W[1:p].mean(0)
    one = {"zeroed": [0, 0], "mean": [0, 0]}          # [answered 0, pairs scored]
    for a in range(1, p):
        for how, edit in (("zeroed", lambda w, a=a: w[a].zero_()),
                          ("mean", lambda w, a=a: w[a].copy_(mean_row))):
            pr = predictions_after(edit)
            others = [b for b in range(1, p) if b != a]
            idx = torch.tensor(others)
            hits = int((pr[a, idx] == 0).sum()) + int((pr[idx, a] == 0).sum())
            one[how][0] += hits
            one[how][1] += 2 * len(others)
    g2 = torch.Generator().manual_seed(1)
    subset = (torch.randperm(p - 1, generator=g2)[:n_subset] + 1).tolist()
    sub_t = torch.tensor(subset)
    # NB: w[sub_t].zero_() would zero a COPY (advanced indexing does not return a
    # view) and leave the weights untouched; index_fill_ writes in place.
    both_pr = predictions_after(lambda w: w.index_fill_(0, sub_t, 0.0))
    block = both_pr[sub_t][:, sub_t]                 # pairs where BOTH inputs were blanked
    vals, counts = torch.unique(block, return_counts=True)
    out["one_input_without_signal"] = {
        "residues_probed": p - 1, "positions": "both",
        "pairs_scored": one["zeroed"][1],
        "frac_predicted_0_zeroed": one["zeroed"][0] / one["zeroed"][1],
        "frac_predicted_0_mean": one["mean"][0] / one["mean"][1]}
    out["both_inputs_without_signal"] = {
        "rows_blanked": subset, "pairs_scored": int(block.numel()),
        "most_common_prediction": int(vals[counts.argmax()]),
        "its_share": float(counts.max()) / int(block.numel()),
        "frac_predicted_0": float((block == 0).float().mean())}
    # (0, 0) itself, once 0's own row is blanked: is the answer still 0?
    pr00 = predictions_after(lambda w: w[0].zero_())
    out["zero_zero_with_row0_blanked"] = int(pr00[0, 0])

    out["random_direction_mean_norm"] = {
        "n": n_random, "min": spread[0],
        "median": (spread[(n_random - 1) // 2] + spread[n_random // 2]) / 2,
        "max": spread[-1], "all": spread}
    out["scale_factor"] = scale
    out["row0_norm"] = float(r0.norm())
    out["mean_other_norm"] = mean_norm
    out["row0_is_smallest"] = bool(float(r0.norm()) < float(norms.min()))
    return out
