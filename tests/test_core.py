"""Correctness tests for the pieces everything else is built on.

These are deliberately paranoid.  A silent bug in the data split or in the
Fourier basis would not crash anything -- it would just quietly produce a
plausible-looking wrong answer, which is the worst possible failure mode for a
project whose entire output is numbers.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.data import OPERATIONS, make_dataset
from grokking.fourier import (
    fourier_1d, fourier_2d, frequency_of_index, gini, inverse_fourier_2d,
    make_fourier_basis, power_by_frequency,
)
from grokking.model import ModelConfig, OneLayerTransformer, final_logits
from grokking.train import checkpoint_steps

P = 113


# ----------------------------------------------------------------- data

def test_split_is_a_partition():
    d = make_dataset(p=P, train_frac=0.3, seed=0)
    tr, te = set(d.train_idx.tolist()), set(d.test_idx.tolist())
    assert tr & te == set()
    assert len(tr | te) == P * P
    assert d.n_train == int(0.3 * P * P)


def test_split_is_deterministic_in_seed():
    a = make_dataset(seed=0).train_idx
    b = make_dataset(seed=0).train_idx
    c = make_dataset(seed=1).train_idx
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_split_does_not_consume_global_rng():
    torch.manual_seed(0)
    x = torch.randn(3)
    torch.manual_seed(0)
    make_dataset(seed=7)
    y = torch.randn(3)
    assert torch.equal(x, y)


@pytest.mark.parametrize("op", sorted(OPERATIONS))
def test_labels_match_python_arithmetic(op):
    d = make_dataset(op=op)
    grid = d.label_grid()
    py = {
        "add": lambda a, b: (a + b) % P,
        "sub": lambda a, b: (a - b) % P,
        "mul": lambda a, b: (a * b) % P,
        "sq_sum": lambda a, b: (a * a + b * b) % P,
        "sq_sum_cross": lambda a, b: (a * a + a * b + b * b) % P,
        "cube_add": lambda a, b: (a ** 3 + b) % P,
        "cube_cross": lambda a, b: (a ** 3 + a * b) % P,
    }[op]
    for a in (0, 1, 7, 56, 112):
        for b in (0, 3, 44, 112):
            assert int(grid[a, b]) == py(a, b), (op, a, b)


def test_tokenisation():
    d = make_dataset()
    assert d.vocab_size == P + 1
    assert d.eq_token == P
    # row-major: row index i is (a = i // p, b = i % p)
    for i in (0, 1, P, 5 * P + 7, P * P - 1):
        a, b, eq = d.inputs[i].tolist()
        assert (a, b, eq) == (i // P, i % P, P)


# -------------------------------------------------------------- fourier

def test_basis_is_orthonormal():
    F, names = make_fourier_basis(P, dtype=torch.float64)
    assert F.shape == (P, P)
    assert len(names) == P
    err = (F @ F.T - torch.eye(P, dtype=torch.float64)).abs().max()
    assert err < 1e-12


def test_fourier_round_trip():
    F, _ = make_fourier_basis(P, dtype=torch.float64)
    x = torch.randn(P, P, 3, dtype=torch.float64)
    assert (inverse_fourier_2d(fourier_2d(x, F), F) - x).abs().max() < 1e-10


def test_parseval():
    F, _ = make_fourier_basis(P, dtype=torch.float64)
    x = torch.randn(P, P, dtype=torch.float64)
    c = fourier_2d(x, F)
    assert math.isclose(float(x.pow(2).sum()), float(c.pow(2).sum()), rel_tol=1e-10)


@pytest.mark.parametrize("k", [1, 2, 17, 56])
def test_pure_wave_is_one_coefficient(k):
    F, names = make_fourier_basis(P, dtype=torch.float64)
    x = torch.arange(P, dtype=torch.float64)
    v = torch.cos(2 * torch.pi * k * x / P)
    c = fourier_1d(v, F)
    nz = (c.abs() > 1e-9).nonzero().flatten().tolist()
    assert nz == [2 * k - 1]
    assert names[nz[0]] == f"cos {k}"
    assert frequency_of_index(nz[0]) == k


@pytest.mark.parametrize("k", [3, 23, 41])
def test_trig_identity_decomposition(k):
    """cos(w(a+b)) must be exactly cos(wa)cos(wb) - sin(wa)sin(wb)."""
    F, names = make_fourier_basis(P, dtype=torch.float64)
    x = torch.arange(P, dtype=torch.float64)
    target = torch.cos(2 * torch.pi * k * (x[:, None] + x[None, :]) / P)
    c = fourier_2d(target, F)
    nz = (c.abs() > 1e-9).nonzero().tolist()
    assert len(nz) == 2
    got = {(names[i], names[j]): round(float(c[i, j]), 6) for i, j in nz}
    assert got[(f"cos {k}", f"cos {k}")] > 0
    assert got[(f"sin {k}", f"sin {k}")] < 0
    assert math.isclose(got[(f"cos {k}", f"cos {k}")], -got[(f"sin {k}", f"sin {k}")], rel_tol=1e-9)
    pw = power_by_frequency(c, P, dims=(0, 1)).sum(dim=1)
    assert math.isclose(float(pw[k] / pw.sum()), 1.0, rel_tol=1e-9)


def test_gini_bounds():
    assert float(gini(torch.ones(10))) == pytest.approx(0.0, abs=1e-9)
    spike = torch.zeros(10); spike[0] = 1.0
    assert float(gini(spike)) > 0.85


# ---------------------------------------------------------------- model

def test_param_count_matches_formula():
    cfg = ModelConfig(d_vocab=P + 1)
    m = OneLayerTransformer(cfg)
    expected = (
        cfg.d_vocab * cfg.d_model                       # W_E
        + cfg.n_ctx * cfg.d_model                       # W_pos
        + 3 * cfg.n_heads * cfg.d_model * cfg.d_head    # W_Q, W_K, W_V
        + cfg.n_heads * cfg.d_head * cfg.d_model        # W_O
        + cfg.d_model * cfg.d_mlp * 2                   # W_in, W_out
        + cfg.d_model * cfg.d_vocab_out                 # W_U: output space is the
        #                                                 residues only, no "=" column
    )
    assert m.n_params() == expected
    # this repository's configuration at p = 113 (Nanda et al.'s model adds MLP
    # biases and a live output class for '=', so it is not their count)
    assert OneLayerTransformer(ModelConfig()).n_params() == 226_048


def test_untrained_loss_is_uniform():
    d = make_dataset()
    m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size, seed=3))
    x, y = d.train()
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(final_logits(m(x, last_only=True), P), y)
    assert abs(float(loss) - math.log(P)) < 0.2


def test_attention_is_causal():
    d = make_dataset()
    m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size))
    _, cache = m.run_with_cache(d.inputs[:4])
    pat = cache["attn_pattern"]
    assert pat.shape[-2:] == (3, 3)
    assert float(pat[..., 0, 1:].abs().max()) == 0.0
    assert float(pat[..., 1, 2:].abs().max()) == 0.0
    assert torch.allclose(pat.sum(-1), torch.ones(1), atol=1e-6)


def test_fast_path_matches_full_path():
    """last_only must be numerically identical at the final position."""
    d = make_dataset()
    m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size, seed=11))
    x = d.inputs[:512]
    with torch.no_grad():
        full = final_logits(m(x), P)
        fast = final_logits(m(x, last_only=True), P)
    assert (full - fast).abs().max() < 1e-5


def test_fast_path_gradients_match():
    d = make_dataset()
    x, y = d.train()
    x, y = x[:256], y[:256]

    def grad_vec(last_only):
        m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size, seed=5))
        loss = torch.nn.functional.cross_entropy(
            final_logits(m(x, last_only=last_only), P), y)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])

    assert (grad_vec(False) - grad_vec(True)).abs().max() < 1e-6


def test_no_layernorm_no_biases():
    m = OneLayerTransformer(ModelConfig(d_vocab=P + 1))
    assert not any(isinstance(mod, torch.nn.LayerNorm) for mod in m.modules())
    assert all("bias" not in n for n, _ in m.named_parameters())


# ---------------------------------------------------------------- train

def test_checkpoint_schedule():
    cs = checkpoint_steps(40_000, 120, (5_000, 20_000, 250))
    assert cs[0] == 0 and cs[-1] == 40_000
    assert cs == sorted(set(cs))
    # the log band resolves the memorisation phase, which is over by step ~200
    assert sum(1 for s in cs if s <= 1_000) > 30
    # the dense band resolves the transition
    dense = [s for s in cs if 5_000 <= s <= 20_000]
    assert len(dense) >= 61
    gaps = [b - a for a, b in zip(dense, dense[1:])]
    assert max(gaps) <= 250
    # a run shorter than the dense band must not produce steps past its end
    short = checkpoint_steps(3_000, 120, (5_000, 20_000, 250))
    assert short[-1] == 3_000 and max(short) <= 3_000


# ------------------------------------------------- even-order basis & dlog

from grokking.fourier import discrete_log_table, n_frequencies  # noqa: E402


@pytest.mark.parametrize("n", [7, 8, 112, 113])
def test_basis_orthonormal_for_even_and_odd(n):
    F, names = make_fourier_basis(n, dtype=torch.float64)
    assert F.shape == (n, n)
    assert len(names) == n
    assert (F @ F.T - torch.eye(n, dtype=torch.float64)).abs().max() < 1e-12


def test_even_basis_has_nyquist_term():
    n = 112
    F, names = make_fourier_basis(n, dtype=torch.float64)
    assert names[-1] == f"cos {n // 2}"
    x = torch.arange(n, dtype=torch.float64)
    nyq = torch.cos(torch.pi * x)
    nyq = nyq / nyq.norm()
    assert torch.allclose(F[-1].abs(), nyq.abs(), atol=1e-12)


def test_n_frequencies():
    assert n_frequencies(113) == 57     # const + 1..56
    assert n_frequencies(112) == 57     # const + 1..56 (56 is Nyquist)


def test_discrete_log_is_an_isomorphism():
    """a * b mod p must equal g^(dlog[a] + dlog[b] mod p-1)."""
    p = 113
    dlog, exp_table, g = discrete_log_table(p)
    assert g == 3
    assert int(dlog[0]) == -1
    for a in range(1, p):
        assert pow(g, int(dlog[a]), p) == a
    a = torch.arange(1, p)
    prod = (a[:, None] * a[None, :]) % p
    lhs = dlog[prod]
    rhs = (dlog[a][:, None] + dlog[a][None, :]) % (p - 1)
    assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("p", [53, 113])
def test_primitive_root_generates_whole_group(p):
    _, exp_table, g = discrete_log_table(p)
    assert sorted(exp_table.tolist()) == list(range(1, p))


# ------------------------------------------------------------ determinism

def test_single_threaded_training_is_bit_reproducible(tmp_path):
    """Same seeds, one thread -> identical weights, through the real Trainer.

    Worth pinning down, because it is NOT true with more than one thread:
    PyTorch's multi-threaded CPU reductions do not fix their summation order,
    so two runs of this same code at 6 threads diverge within a hundred steps.
    Runs in this repository are therefore reproducible in distribution but not
    bit-exact unless single-threaded, and the write-up says so.  Fifteen steps
    cover the warmup and the float64 loss path the shipped runs used.
    """
    import hashlib

    from grokking.train import TrainConfig, Trainer

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        def run(out):
            d = make_dataset(p=P, seed=0)
            m = OneLayerTransformer(ModelConfig(d_vocab=d.vocab_size, seed=0))
            cfg = TrainConfig(steps=15, log_every=5, n_log_checkpoints=2,
                              dense_from=0, dense_to=0, dense_every=1)
            Trainer(m, d, cfg, out, tag="det").run(verbose=False)
            with torch.no_grad():
                return hashlib.sha256(
                    b"".join(p.numpy().tobytes() for p in m.parameters())).hexdigest()

        assert run(tmp_path / "a") == run(tmp_path / "b")
    finally:
        torch.set_num_threads(old)


# ------------------------------------------------ the data-seed regression

def test_analysis_uses_the_runs_own_split(tmp_path):
    """A run trained on seed 7 must be analysed on seed 7's split.

    This was once wrong: the analysis scripts took the seed as a flag that
    defaulted to 0, so a non-zero-seed run was silently evaluated on another
    split, where about half its training pairs are really held-out pairs.
    """
    import json

    from grokking.runinfo import run_config, run_dataset

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r7_history.json").write_text(json.dumps(
        {"data": {"p": P, "op": "add", "train_frac": 0.3, "seed": 7}, "history": []}))
    assert run_config(tmp_path, "r7")["seed"] == 7
    got = run_dataset(tmp_path, "r7").train_idx
    assert torch.equal(got, make_dataset(p=P, seed=7).train_idx)
    assert not torch.equal(got, make_dataset(p=P, seed=0).train_idx)


def test_missing_history_is_an_error_not_a_default(tmp_path):
    from grokking.runinfo import run_config

    (tmp_path / "results").mkdir()
    with pytest.raises(FileNotFoundError):
        run_config(tmp_path, "nope")


def test_no_analysis_script_builds_its_own_split():
    """Every analysis script must get its dataset from runinfo, not from flags."""
    import re

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("analyze_run.py", "analyze_mechanism.py",
                 "analyze_redundancy.py", "analyze_quadratic.py"):
        src = (scripts / name).read_text()
        assert "run_dataset(" in src, name
        assert not re.search(r"make_dataset\([^)]*seed=(args\.data_seed|0)\b", src), name


def test_training_sets_the_output_width_to_p():
    """No dead output columns at any modulus, not just at p = 113."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_train.py").read_text()
    assert "d_vocab_out=args.d_vocab_out or data.p" in src
    from grokking.model import ModelConfig
    assert ModelConfig(d_vocab=60).d_vocab_out == 59
