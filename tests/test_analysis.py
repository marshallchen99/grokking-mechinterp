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
    markers = set(re.findall(r"<!-- BEGIN:([a-z_]+) -->", readme))
    assert markers == set(blocks), ("README markers and generators differ",
                                    sorted(markers ^ set(blocks)))
    mismatched = []
    for name, body in blocks.items():
        assert "Block failed to render" not in body, (name, body[:200])
        assert body != PENDING, f"{name} is pending"
        m = re.search(r"<!-- BEGIN:%s -->\n(.*?)\n<!-- END:%s -->" % (name, name), readme, re.S)
        if m.group(1) != body:
            mismatched.append(name)
    assert not mismatched, mismatched


def test_controls_are_distinguishable():
    """Every control row must describe a different configuration.

    A reference run once rendered with the same description as its own float32
    control, because a missing record field fell back to a default; the README
    and a fresh render agreed, so the consistency test above could not see it.
    """
    from grokking.report_blocks import CONTROL_ORDER, READOUT_RUNS, describe_run
    root = Path(__file__).resolve().parents[1]
    for group in (CONTROL_ORDER, READOUT_RUNS):
        descs = {t: describe_run(root, t) for t in group
                 if (root / "results" / f"{t}_history.json").exists()}
        assert len(set(descs.values())) == len(descs), descs
        assert not any("not recorded" in d for d in descs.values()), descs


def test_every_reported_run_has_a_known_loss_precision():
    from grokking.runinfo import training_choice
    root = Path(__file__).resolve().parents[1]
    for f in sorted((root / "results").glob("*_history.json")):
        tag = f.name[: -len("_history.json")]
        assert training_choice(root, tag, "loss_dtype") in ("float32", "float64"), tag


def test_report_prose_has_no_hand_picked_numbers_in_expressions():
    """Numeric constants inside formatted expressions are method choices too.

    Allowed: 0, 1, 2 (counting, halving, pairs), 60 and 1000 (unit conversions)
    and 100 (percent). Anything
    else -- a 0.8 or a *5 used to pick where to sample a curve -- must be a
    named constant, so that it is visible as a choice.
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "src" / "grokking" / "report_blocks.py")
    tree = ast.parse(src.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue):
            for sub in ast.walk(node.value):
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float))
                        and not isinstance(sub.value, bool)
                        and sub.value not in (0, 1, 2, 60, 100, 1000)):   # counts; s->min, s->ms, %
                    bad.append((sub.lineno, sub.value))
    assert not bad, f"hand-picked numbers inside formatted expressions: {bad}"


# ------------------------------------------------------- interventions

def _small_model_and_data(p=31, seed=3):
    from grokking.analysis.core import load_snapshot  # noqa: F401  (import check)
    from grokking.model import ModelConfig, OneLayerTransformer
    d = make_dataset(p=p, seed=0)
    m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size, d_model=32, n_heads=4,
                                        d_head=8, d_mlp=64, seed=seed))
    m.eval()
    return m, d


def test_mean_ablation_replaces_only_the_dropped_neurons():
    """Mean-ablation must leave every kept neuron's activation untouched.

    The first version folded the dropped neurons' mean output into the position
    embedding, which sent it through attention and the MLP again; every
    neuron-ablation number it produced was wrong and no test noticed.
    """
    from grokking.analysis.ablation import ablate_neurons
    m, d = _small_model_and_data()
    _, cache = m.run_with_cache(d.inputs)
    post = cache["mlp_post"][:, -1]                       # (p*p, d_mlp)
    mean = post.mean(0)
    keep = torch.ones(post.shape[1], dtype=torch.bool)
    keep[::3] = False
    m2 = ablate_neurons(m, keep, mean)
    logits2, cache2 = m2.run_with_cache(d.inputs)
    assert torch.equal(cache2["mlp_post"][:, -1][:, keep], post[:, keep])
    # and the logits are exactly "dropped neurons pinned at their mean"
    mixed = torch.where(keep, post, mean)
    want = (cache["resid_mid"][:, -1] + mixed @ m.W_out) @ m.W_U
    assert torch.allclose(logits2[:, -1], want, atol=1e-5)


def test_mean_ablating_nothing_is_the_identity():
    from grokking.analysis.ablation import ablate_neurons
    m, d = _small_model_and_data()
    _, cache = m.run_with_cache(d.inputs)
    mean = cache["mlp_post"][:, -1].mean(0)
    keep = torch.ones(mean.numel(), dtype=torch.bool)
    assert torch.equal(ablate_neurons(m, keep, mean)(d.inputs), m(d.inputs))


def test_embedding_surgery_keeping_everything_is_the_identity():
    from grokking.analysis.ablation import ablate_embedding_frequencies
    m, d = _small_model_and_data()
    F = make_fourier_basis(d.p, dtype=torch.float64)[0]
    every = list(range(1, d.p // 2 + 1))
    kept = ablate_embedding_frequencies(m, F, d.p, every, keep=True)
    assert torch.allclose(kept.W_E[: d.p], m.W_E[: d.p], atol=1e-6)
    none_dropped = ablate_embedding_frequencies(m, F, d.p, [], keep=False)
    assert torch.allclose(none_dropped.W_E[: d.p], m.W_E[: d.p], atol=1e-6)
    # deleting a frequency removes exactly its two basis directions
    gone = ablate_embedding_frequencies(m, F, d.p, [5], keep=False)
    from grokking.analysis.spectra import block_indices
    from grokking.fourier import fourier_1d
    c = fourier_1d(gone.W_E.detach()[: d.p].double(), F, dim=0)
    ck, sk = block_indices(5, d.p)
    assert float(c[[ck, sk]].abs().max()) < 1e-6


def test_readout_budget_on_known_logits(basis):
    """Exact algorithm: all readout energy at the predicted frequency.

    Plus an (a+b)-shaped offset shared by every class, which softmax ignores:
    recentring must remove it completely.
    """
    from grokking.analysis.structure import readout_budget
    L = _algorithm_logits(KEYS)
    rb = readout_budget(L, basis, P, KEYS)
    assert rb["own"] == pytest.approx(1.0, abs=1e-9)
    assert rb["cross"] == pytest.approx(0.0, abs=1e-9)
    x = torch.arange(P, dtype=torch.float64)
    offset = 5 * torch.cos(2 * torch.pi * 7 * (x[:, None] + x[None, :]) / P)[:, :, None]
    assert readout_budget(L + offset, basis, P, KEYS)["own"] == pytest.approx(1.0, abs=1e-9)
    # a subtraction algorithm has all its readout energy in the (a-b) direction
    # and none in (a+b); the fractions of the empty direction mean nothing
    Lsub = _algorithm_logits(KEYS, sign=-1)
    right = readout_budget(Lsub, basis, P, KEYS, sign=-1)
    wrong = readout_budget(Lsub, basis, P, KEYS, sign=+1)
    assert right["own"] == pytest.approx(1.0, abs=1e-9)
    assert wrong["total"] < 1e-12 * right["total"]


def test_crossing_step_interpolates_between_logged_steps():
    from grokking.analysis.timing import crossing_step
    rows = [{"step": 0, "a": 0.0}, {"step": 10, "a": 0.2}, {"step": 20, "a": 0.6}]
    assert crossing_step(rows, "a", 0.4) == pytest.approx(15.0)
    assert crossing_step(rows, "a", 0.7) is None
    assert crossing_step(rows, "a", 0.0) == 0.0


def test_a_changed_split_is_refused(tmp_path):
    """If torch ever rebuilt a run's split differently, analysis must stop."""
    import json
    from grokking.runinfo import run_dataset
    (tmp_path / "results").mkdir()
    rec = {"data": {"p": 31, "op": "add", "train_frac": 0.3, "seed": 2}}
    good = make_dataset(p=31, seed=2, train_frac=0.3).split_hash()
    (tmp_path / "results" / "x_history.json").write_text(
        json.dumps({"data": {**rec["data"], "split_hash": good}}))
    assert run_dataset(tmp_path, "x").split_hash() == good
    (tmp_path / "results" / "x_history.json").write_text(
        json.dumps({"data": {**rec["data"], "split_hash": "0" * 16}}))
    with pytest.raises(RuntimeError):
        run_dataset(tmp_path, "x")


def test_every_shipped_split_is_the_one_trained_on():
    import json
    from grokking.runinfo import run_dataset
    root = Path(__file__).resolve().parents[1]
    table = json.loads((root / "results" / "split_hashes.json").read_text())
    hists = sorted((root / "results").glob("*_history.json"))
    assert {h.name[: -len("_history.json")] for h in hists} == set(table)
    for tag in table:
        run_dataset(root, tag)          # raises on a mismatch


def test_page_is_the_template_with_the_data():
    root = Path(__file__).resolve().parents[1]
    tpl = (root / "web" / "page.template.html").read_text()
    data = (root / "web" / "data.json").read_text()
    assert (root / "web" / "index.html").read_text() == tpl.replace(
        "__DATA__", data.replace("</", "<\\/"))
