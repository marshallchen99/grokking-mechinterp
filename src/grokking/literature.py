"""Published numbers, quarantined.

Every reference string below was checked against the arXiv metadata API
(export.arxiv.org) on 2026-09-21: authors, title and identifier.  Venues are
given only where the arXiv record or the proceedings confirm them.  An earlier
version carried an author list for Notsawo et al. that was reconstructed from
memory and wrong in three of six names; that is why this is checked rather
than recalled.

Every value in this module comes from a paper, carries its reference, and may
appear in the write-up only in a column headed "published".  Nothing here is
ever used as a target, a calibration, or a default.  Numbers produced by this
repository live in results/ and reach the README through `grokking.report`;
the two are never mixed in the same column.

`DISPUTED` records values that a source states but that should not be quoted as
fact, with the reason.  They are kept here so that the reason is written down
rather than rediscovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

Number = Union[int, float, str, List, None]


@dataclass(frozen=True)
class Fact:
    value: Number
    what: str
    source: str
    note: str = ""


NANDA_2023 = "Nanda, Chan, Lieberum, Smith, Steinhardt, 'Progress measures for grokking via mechanistic interpretability', arXiv:2301.05217"
POWER_2022 = "Power, Burda, Edwards, Babuschkin, Misra, 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets', arXiv:2201.02177"
LIU_2022 = "Liu, Michaud, Tegmark, 'Omnigrok: Grokking Beyond Algorithmic Data', arXiv:2210.01117"
VARMA_2023 = "Varma, Shah, Kenton, Kramar, Kumar, 'Explaining grokking through circuit efficiency', arXiv:2309.02390"
CHUGHTAI_2023 = "Chughtai, Chan, Nanda, 'A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations', ICML 2023 (PMLR 202), arXiv:2302.03025"
DOSHI_2024 = "Doshi, He, Das, Gromov, 'Grokking Modular Polynomials', arXiv:2406.03495"
FURUTA_2024 = "Furuta, Minegishi, Iwasawa, Matsuo, 'Towards Empirical Interpretation of Internal Circuits and Properties in Grokked Transformers on Modular Polynomials', TMLR 2024, arXiv:2402.16726"
NOTSAWO_2023 = "Notsawo, Zhou, Pezeshki, Rish, Dumas, 'Predicting Grokking Long Before it Happens: A look into the loss landscape of models which grok', arXiv:2306.13253"
LYU_2023 = "Lyu, Jin, Li, Du, Lee, Hu, 'Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking', ICLR 2024, arXiv:2311.18817"
NGUYEN_2026 = "Nguyen, 'The Discrete-Log Clock: How a Transformer Learns Modular Multiplication', Mechanistic Interpretability Workshop at ICML 2026, arXiv:2606.17399"
CHEN_2026 = "Chen, Hasan, Srinivasan, Bandi, Alper, 'Multiplication Beyond Groups: Stratified Fourier Mechanisms in Transformer Circuits', Mechanistic Interpretability Workshop at ICML 2026, arXiv:2607.07066"
KHANH_2026 = "Khanh, Hoa, Trung, Duc, 'First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation', arXiv:2605.18845"

PUBLISHED: Dict[str, Fact] = {
    "uniform_loss_p113": Fact(
        4.727387818712341, "loss of a uniform guess over 113 classes",
        "arithmetic: ln(113)", "not a citation; stated for comparison"),
    "n_train_steps": Fact(
        40_000, "training steps in the mainline run", NANDA_2023),
    "neurons_above_85pct": Fact(
        433, "MLP neurons (of 512) whose variance is >85% explained by one frequency",
        NANDA_2023, "84.6% of 512"),
    "phase_memorisation_end": Fact(
        1_400, "end of the memorisation phase, in steps", NANDA_2023),
    "phase_circuit_formation_end": Fact(
        9_400, "end of the circuit-formation phase, in steps", NANDA_2023),
    "phase_cleanup_end": Fact(
        14_000, "end of the cleanup phase, in steps", NANDA_2023),
    "gini_W_E_range": Fact(
        [0.55, 0.80], "range of Gini(W_E) across the paper's weight-decay settings",
        NANDA_2023, "Table 5; the vector it is computed over is not specified"),
    "gini_W_L_range": Fact(
        [0.68, 0.91], "range of Gini(W_L) across the paper's weight-decay settings",
        NANDA_2023, "Table 5"),
    "n_key_freqs_range": Fact(
        [2, 9], "number of key frequencies observed across settings", NANDA_2023),
    "logit_var_from_five_coeffs": Fact(
        0.95, "fraction of logit variance from five cube coefficients", NANDA_2023),
    "attention_zero_ablation_loss": Fact(
        24.3, "loss after zero-ablating attention", NANDA_2023),
    "grokking_train_frac_window": Fact(
        [0.30, 0.50], "training fractions at p=113 where grokking is observed",
        NANDA_2023, "at >=60% generalisation is immediate; at 10-20% it does not occur"),
    "power_sensitivity": Fact(
        [0.40, 0.50], "relative increase in median time-to-generalisation per 1% less data",
        POWER_2022, "near 25-30% training data"),
    "power_headline_optimizer": Fact(
        "Adam, no weight decay, 1e6 step budget",
        "the setup behind the headline division-mod-97 curve", POWER_2022),
}

DISPUTED: Dict[str, Fact] = {
    "key_frequencies": Fact(
        [14, 35, 41, 42, 52], "the key frequency set", NANDA_2023,
        "one particular run; the set is seed-dependent and not reproducible. "
        "Quote only as 'one published run', never as the answer."),
    "weight_decay_speed": Fact(
        None, "whether larger weight decay makes grokking faster or slower",
        NANDA_2023 + " Appendix D.1",
        "the source contradicts itself: three prose statements say faster, the "
        "numeric list says otherwise. Not quotable in either direction."),
    "sixth_frequency": Fact(
        None, "the identity of the sixth, non-key embedding frequency", NANDA_2023,
        "not determined by the source"),
}

RELATED_WORK = {
    "discrete_log_for_multiplication": (
        DOSHI_2024,
        "Gives an analytic construction for modular multiplication that explicitly "
        "uses the discrete logarithm, and a learnability hypothesis (Hyp. 5.1) for forms "
        "h(g1(a) + g2(b)) mod p. The reduction is therefore published prior art: "
        "this repository tests it causally, it does not claim it."),
    "group_representations": (
        CHUGHTAI_2023,
        "The learned features are irreducible representations of the underlying "
        "group; for (Z/pZ)* under multiplication those are the multiplicative "
        "characters."),
    "operation_survey": (
        FURUTA_2024,
        "Maps which modular operations grok and their spectral signatures: "
        "subtraction is addition-like but asymmetric; multiplication appears "
        "dense in the additive basis; non-factorisable quadratics fail to find "
        "sparse embeddings."),
    "circuit_efficiency": (
        VARMA_2023,
        "Explains grokking as competition between a memorising and a generalising "
        "circuit with different efficiency, and predicts ungrokking and "
        "semi-grokking."),
    "init_scale": (
        LIU_2022,
        "Initialisation scale controls the grokking delay; large init lengthens it."),
    "causal_irrep_ablation": (
        CHUGHTAI_2023,
        "Runs restricted loss, excluded loss and ablations of irreducible-representation "
        "subspaces for group composition, including cyclic groups. (Z/pZ)* under "
        "multiplication is the cyclic group of order p-1, whose irreps are the discrete-log "
        "Fourier modes -- so the causal test in that basis is published in principle; this "
        "repository applies it to a model trained on the full multiplication table."),
    "dlog_transformer": (
        NGUYEN_2026,
        "Finds the discrete-log-basis sparsity in a transformer trained on modular "
        "multiplication. Its abstract reports no ablations."),
    "zero_divisors": (
        CHEN_2026,
        "Treats non-invertible elements as a separate algebraic region, on composite "
        "moduli, and describes its evidence as correlational. Doshi et al. also single out "
        "0 as warranting separate treatment."),
    "forecasting": (
        NOTSAWO_2023,
        "Forecasts grokking from early training-loss curves. Forecasting from early "
        "signals is therefore prior art; this repository's contribution is a comparison of "
        "which signals do it, within one configuration."),
    "weight_decay_scaling": (
        LYU_2023,
        "Proves grokking time scales like 1/lambda in weight decay; Khanh et al. 2026 "
        "(arXiv:2605.18845) give a calibrated delay law with the same dependence. The phase "
        "diagram here reproduces it rather than settling an open question."),
}


def cite(key: str) -> str:
    f = PUBLISHED.get(key) or DISPUTED.get(key)
    if f is None:
        raise KeyError(key)
    return f.source


def quotable(key: str) -> bool:
    return key in PUBLISHED
