"""Calibration tests for the mechanistic-analysis tools.

Every metric reported in the write-up is run here against inputs whose answer
is known in advance, so that a number measured on a real model can be read
against its null level rather than judged by whether it "looks high".

The pattern throughout: build logits that ARE the claimed algorithm, check the
metric saturates; build a plausible wrong algorithm, check it does not; build
noise, check the metric returns chance.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.analysis.progress import (
    excluded_loss, filter_logits, restricted_loss,
)
from grokking.analysis.spectra import (
    embedding_spectrum, freq_block_power, key_frequencies, neuron_frequencies,
)
from grokking.analysis.structure import additive_structure, trig_identity_report
from grokking.data import make_dataset
from grokking.fourier import discrete_log_table, make_fourier_basis

P = 113
KEYS = [14, 35, 41, 52]
CHANCE_VAR = 1.0 / P


@pytest.fixture(scope="module")
def basis():
    return make_fourier_basis(P, dtype=torch.float64)[0]


def _algorithm_logits(keys, sign=+1, p=P, amp=None):
    """logits(a, b, c) = sum_k amp_k * cos(w_k * (a + sign*b - c)) -- the claimed algorithm."""
    x = torch.arange(p, dtype=torch.float64)
    out = torch.zeros(p, p, p, dtype=torch.float64)
    for i, k in enumerate(keys):
        w = 2 * torch.pi * k / p
        a = 1.0 if amp is None else amp[i]
        out += a * torch.cos(w * (x[:, None, None] + sign * x[None, :, None] - x[None, None, :]))
    return out


# ------------------------------------------------------ structure metrics

def test_exact_algorithm_saturates_every_metric(basis):
    L = _algorithm_logits(KEYS)
    st = additive_structure(L, P)
    assert st["a+b"] == pytest.approx(1.0, abs=1e-9)
    assert st["a-b"] == pytest.approx(0.0, abs=1e-9)
    r = trig_identity_report(L, basis, P, KEYS)
    assert r["mean_sum_frac"] == pytest.approx(1.0, abs=1e-9)
    assert r["mean_readout_frac"] == pytest.approx(1.0, abs=1e-9)


def test_wrong_algorithm_is_rejected(basis):
    """An (a-b) circuit must score ~0 on the (a+b) tests, not merely lower."""
    L = _algorithm_logits(KEYS, sign=-1)
    st = additive_structure(L, P)
    assert st["a-b"] == pytest.approx(1.0, abs=1e-9)
    assert st["a+b"] < 1e-9
    r = trig_identity_report(L, basis, P, KEYS)
    assert r["mean_sum_frac"] < 1e-9


def test_random_logits_sit_at_chance(basis):
    torch.manual_seed(0)
    L = torch.randn(P, P, P, dtype=torch.float64)
    st = additive_structure(L, P)
    for k, v in st.items():
        assert v == pytest.approx(CHANCE_VAR, abs=0.004), (k, v)
    r = trig_identity_report(L, basis, P, KEYS)
    # with no structure the energy splits evenly between the (a+b) and (a-b)
    # directions, so the null level of this metric is one half, not zero
    assert r["mean_sum_frac"] == pytest.approx(0.5, abs=0.15)
    assert r["mean_readout_frac"] < 0.1


def test_memorisation_looks_like_nothing(basis):
    """A lookup table has no (a+b) structure -- the pre-grokking picture.

    The fixture has to be right for this to mean anything: a memorising model
    is confidently correct on the cells it was shown and arbitrary on the rest.
    Writing the correct answer everywhere would make it a *perfect* model, which
    of course does depend on (a+b) -- and indeed scores 0.999 here, which is a
    useful check on the metric in its own right.
    """
    gen = torch.Generator().manual_seed(1)
    d = make_dataset(p=P, train_frac=0.3, seed=0)
    L = torch.randn(P, P, P, dtype=torch.float64) * 0.1
    answers = d.label_grid().clone()
    test_mask = ~d.train_mask()
    noise = torch.randint(0, P, (int(test_mask.sum()),), generator=gen)
    answers[test_mask] = noise                         # arbitrary off the training set
    L.scatter_(2, answers[:, :, None], 30.0)
    st = additive_structure(L, P)
    assert st["a+b"] < 0.35, st                        # far below the 0.96 a real
    #                                                    grokked model reaches
    perfect = torch.randn(P, P, P, dtype=torch.float64) * 0.1
    perfect.scatter_(2, d.label_grid()[:, :, None], 30.0)
    assert additive_structure(perfect, P)["a+b"] > 0.99


# ------------------------------------------------------- progress measures

def test_restriction_is_lossless_for_the_exact_algorithm(basis):
    """Keeping only the key frequencies' (a+b) directions must not change an
    algorithm that consists of nothing else."""
    L = _algorithm_logits(KEYS)
    kept = filter_logits(L, basis, P, KEYS, keep=True, mode="sum")
    assert (kept - L).abs().max() < 1e-9


def test_exclusion_destroys_the_exact_algorithm(basis):
    L = _algorithm_logits(KEYS)
    dropped = filter_logits(L, basis, P, KEYS, keep=False, mode="sum")
    assert dropped.abs().max() < 1e-9


def test_restricted_and_excluded_are_complementary(basis):
    """restricted + excluded must reconstruct the original exactly."""
    torch.manual_seed(2)
    L = torch.randn(P, P, P, dtype=torch.float64)
    kept = filter_logits(L, basis, P, KEYS, keep=True, mode="sum")
    dropped = filter_logits(L, basis, P, KEYS, keep=False, mode="sum")
    # the constant term is deliberately kept by BOTH, so add it back once
    const = L.mean(dim=(0, 1), keepdim=True).expand_as(L)
    assert (kept + dropped - const - L).abs().max() < 1e-8


class _FakeSnapshot:
    def __init__(self, logits, data):
        self.logits, self.data = logits, data

    @property
    def p(self):
        return self.data.p


def test_progress_measures_on_a_known_model(basis):
    d = make_dataset(p=P, train_frac=0.3, seed=0)
    L = _algorithm_logits(KEYS) * 8.0
    snap = _FakeSnapshot(L, d)
    import torch.nn.functional as tF
    baseline = float(tF.cross_entropy(L.reshape(P * P, P), d.labels))
    r = restricted_loss(snap, basis, KEYS, mode="sum", split="all")
    e = excluded_loss(snap, basis, KEYS, mode="sum", split="train")
    # The model IS the restricted subspace, so restricting must be a no-op --
    # that is the meaningful assertion, not any particular small number.
    assert r["loss"] == pytest.approx(baseline, rel=1e-6)
    assert r["acc"] == 1.0
    assert e["acc"] < 0.02                      # at or below chance
    assert e["loss"] > math.log(P) - 0.5        # no better than guessing


# ------------------------------------------------------- key-frequency rules

def test_key_frequency_rules_find_a_planted_set(basis):
    """Plant a known sparse spectrum in an embedding and check it is recovered."""
    torch.manual_seed(3)
    d_model = 128
    coeffs = torch.randn(P, d_model, dtype=torch.float64) * 0.01
    for k in KEYS:
        coeffs[2 * k - 1] *= 60.0
        coeffs[2 * k] *= 60.0
    W = basis.T @ coeffs
    spec = embedding_spectrum(W, basis, P, method="gap")
    assert spec.key_freqs == sorted(KEYS)
    assert spec.sparsity_report()["frac_power_in_key_freqs"] > 0.95
    assert spec.gini > 0.8
    assert key_frequencies(spec.power_per_freq, method="topk", k=4) == sorted(KEYS)


def test_key_frequency_rules_find_nothing_in_noise(basis):
    torch.manual_seed(4)
    W = torch.randn(P, 128, dtype=torch.float64)
    spec = embedding_spectrum(W, basis, P, method="gap")
    assert spec.gini < 0.3
    assert spec.sparsity_report()["frac_power_in_key_freqs"] < 0.35


def test_neuron_assignment_recovers_planted_frequencies(basis):
    """Build neurons that ARE single-frequency waves and check the assignment."""
    x = torch.arange(P, dtype=torch.float64)
    acts = []
    truth = []
    for k in KEYS:
        w = 2 * torch.pi * k / P
        for phase in (0.0, 1.1):
            acts.append(torch.cos(w * (x[:, None] + x[None, :]) + phase))
            truth.append(k)
    A = torch.stack(acts, dim=-1)
    nf = neuron_frequencies(A, basis, P)
    assert nf["dominant_freq"].tolist() == truth
    assert float(nf["dominant_frac"].min()) > 0.99


# -------------------------------------------------------------- discrete log

def test_dlog_turns_multiplication_into_addition():
    p = 113
    dlog, exp_table, g = discrete_log_table(p)
    n = p - 1
    e = torch.arange(n)
    prod = (exp_table[e][:, None] * exp_table[e][None, :]) % p
    assert torch.equal(dlog[prod], (e[:, None] + e[None, :]) % n)


# ------------------------------------------- no hand-typed results in prose

def test_report_prose_contains_no_hand_typed_results():
    """Every number in a README sentence must come from a results file.

    An earlier version of report_blocks.py printed string literals such as
    "seven orders of magnitude" and "0.998 and 0.999" whatever the data said,
    and one of them was wrong by four orders of magnitude.  This scans every
    literal text fragment in the module for things that look like measured
    values -- decimals, thousands-separated counts, "N orders", "Nx" -- and
    fails if any appear outside a computed expression.
    """
    import ast
    import re

    src_path = Path(__file__).resolve().parents[1] / "src" / "grokking" / "report_blocks.py"
    tree = ast.parse(src_path.read_text())
    # decimals, thousands-separated counts, "N orders", "Nx", "N%", and any bare
    # integer of two or more digits.  Number *words* are not caught (the prose
    # legitimately says "one seed", "two directions"), which is a known limit.
    looks_measured = re.compile(
        r"(?<![\w.])(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*orders?\b|\d+x\b"
        r"|\d+%|\d{2,})(?![\w])")
    # Only these spans are exempt, and only the matched span, not the fragment:
    # arXiv ids, years, format specs, and the names of the moduli/operations.
    exempt_span = re.compile(r"arXiv:\S+|\b(19|20)\d{2}\b|^\d+$")
    # Docstrings never reach the README; they are allowed to quote old mistakes.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    offenders = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.JoinedStr):
            parts = [v.value for v in node.values if isinstance(v, ast.Constant)
                     and isinstance(v.value, str)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = [node.value]
        else:
            continue
        for text in parts:
            if re.fullmatch(r"\s*\d+\s*", text):
                continue                        # a code constant, not prose
            cleaned = re.sub(r"arXiv:\S+", " ", text)
            cleaned = re.sub(r"Hypothesis \d+(\.\d+)?", " ", cleaned)   # a citation locator
            cleaned = re.sub(r"\b(19|20)\d{2}\b", " ", cleaned)   # years in citations
            for m in looks_measured.finditer(cleaned):
                offenders.append((getattr(node, "lineno", "?"), m.group(0), text.strip()[:70]))
    assert not offenders, "hand-typed numbers in report prose:\n" + "\n".join(
        f"  line {ln}: {num!r} in {txt!r}" for ln, num, txt in offenders)



def test_readme_matches_a_fresh_render():
    """The README claims every section is regenerated from results/; check it.

    Re-renders every block with the shipped defaults and compares it with the
    text between the README's markers.  A drift here means a table in the
    README no longer comes from the data it claims to.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    from grokking.report_blocks import OP_TAGS, PENDING, build

    blocks = build(root, "main_add_s0", op_tags=OP_TAGS)
    mismatched = []
    for name, body in blocks.items():
        m = re.search(r"<!-- BEGIN:%s -->\n(.*?)\n<!-- END:%s -->" % (name, name), readme, re.S)
        if m is None:
            continue
        if body == PENDING:
            mismatched.append((name, "pending"))
        elif m.group(1) != body:
            mismatched.append((name, "differs"))
    assert not mismatched, mismatched
