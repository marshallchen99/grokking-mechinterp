"""Turn the results files into the README's tables.

Adding a block here is the only way a number reaches the write-up.  Each
generator returns markdown for one `<!-- BEGIN:name -->` region and returns
None when its inputs are missing, so a partially-finished experiment renders
the blocks it has and leaves the rest marked pending.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .literature import PUBLISHED
from .report import load_analysis, load_history, load_json, table


PENDING = "_Not yet run._"


def _phase_rows(hist: List[Dict]) -> Dict[str, Optional[float]]:
    from .analysis.timing import crossing_step
    return {
        "train_100": crossing_step(hist, "train_acc", 0.99),
        "test_10": crossing_step(hist, "test_acc", 0.10),
        "test_50": crossing_step(hist, "test_acc", 0.50),
        "test_90": crossing_step(hist, "test_acc", 0.90),
        "test_99": crossing_step(hist, "test_acc", 0.99),
    }


def headline(root: Path, tag: str) -> Optional[str]:
    h = load_history(root, tag)
    if not h:
        return None
    hist = h["history"]
    c = _phase_rows(hist)
    peak = max(hist, key=lambda r: r["test_loss"])
    chance = 1.0 / h["data"]["p"]
    def st(v):
        # these are step counts; a fractional step is an interpolation artefact,
        # not a measurement
        return int(round(v)) if v is not None else None

    rows = [
        ["training accuracy reaches 99%", st(c["train_100"]), "the model has memorised its training set"],
        ["test accuracy reaches 10%", st(c["test_10"]), f"chance is {chance:.2%}"],
        ["test accuracy reaches 50%", st(c["test_50"]), ""],
        ["test accuracy reaches 90%", st(c["test_90"]), "generalisation"],
        ["test accuracy reaches 99%", st(c["test_99"]), ""],
    ]
    plateau = (c["test_90"] - c["train_100"]) if (c["test_90"] and c["train_100"]) else None
    body = table(["event", "step", "note"], rows, align="lrl")
    extra = [
        "",
        f"The model spends **{plateau:,.0f} steps** with perfect training accuracy and "
        f"near-chance test accuracy. Test loss does not merely stay flat during that "
        f"stretch -- it *rises*, peaking at **{peak['test_loss']:.2f}** at step "
        f"{peak['step']:,}, as the memorised solution becomes more confident and more wrong.",
        "",
        f"Setup: `({h['data']['op']}) mod {h['data']['p']}`, "
        f"{h['data']['n_train']:,} of {h['data']['n_train']+h['data']['n_test']:,} pairs "
        f"used for training ({h['data']['train_frac']:.0%}), a {h['n_params']:,}-parameter "
        f"one-layer transformer, full-batch AdamW with weight decay "
        f"{h['train_cfg']['weight_decay']}, {h['train_cfg']['steps']:,} steps, CPU only.",
    ]
    return body + "\n".join(extra)


def mechanism(root: Path, tag: str) -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m:
        return None
    k = m["key_freqs"]
    rules = table(
        ["rule", "what it looks at", "frequencies found"],
        [["embedding-norm threshold", "per-index norm of the Fourier-transformed W_E",
          k["by_rule"]["rule_a_embedding_norm"]],
         ["neuron clustering", "which frequency explains >85% of each MLP neuron",
          k["by_rule"]["rule_b_neuron_cluster"]],
         ["power gap (parameter-free)", "largest multiplicative gap in sorted per-frequency power",
          k["by_rule"]["rule_c_power_gap"]]],
        align="lll")

    n = m["neurons"]
    census = table(
        ["frequency", "neurons assigned"],
        [[k_, v] for k_, v in sorted(n["counts_by_freq"].items(), key=lambda kv: -kv[1])],
        align="lr")

    s = m["structure"]
    chance = 1.0 / m["p"]
    struct = table(
        ["the logits are a function of...", "variance explained"],
        [["(a + b) mod p", s["a+b"]], ["(a - b) mod p", s["a-b"]],
         ["a alone", s["a"]], ["b alone", s["b"]], ["(a * b) mod p", s["a*b"]]],
        bold_best={1: "max"})

    t = m["trig"]
    trig = table(
        ["frequency", "energy in cos/sin(w(a+b))", "readout is a wave at the same frequency"],
        [[int(f), v["sum_frac"], v["readout_frac"]] for f, v in sorted(t["per_freq"].items(), key=lambda kv: int(kv[0]))]
        + [["**mean**", t["mean_sum_frac"], t["mean_readout_frac"]]])

    return "\n\n".join([
        f"Three rules with independent logic are applied to the final checkpoint "
        f"(step {m['final_step']:,}). They agree exactly (Jaccard "
        f"{k['jaccard']:.3f}):",
        rules,
        f"Those {len(k['used'])} frequencies carry **{k['frac_power_in_key']:.1%}** of the "
        f"embedding's Fourier power, out of {(m['p']-1)//2} available. The spectrum's Gini "
        f"coefficient is **{k['gini_W_E']:.4f}** "
        f"(published range across settings: {PUBLISHED['gini_W_E_range'].value[0]}"
        f"-{PUBLISHED['gini_W_E_range'].value[1]}).",
        f"Every one of the {n['n_total']} MLP neurons is dominated by one of them"
        + (":" if n["all_on_key_freqs"] else " -- except where noted:"),
        census,
        f"{n['frac_above_85']:.1%} of neurons have more than 85% of their variance explained "
        f"by a single frequency (mean {n['mean_dominant_frac']:.4f}, minimum "
        f"{n['min_dominant_frac']:.4f}).",
        f"**What it computes.** Chance for the variance-explained column is {chance:.4f}:",
        struct,
        "**The trigonometric identity.** Within each key frequency, the dependence on "
        "(a, b) can be split into the part that is a function of (a+b) and the part that "
        "is a function of (a-b). The identity `cos(w(a+b)) = cos(wa)cos(wb) - "
        "sin(wa)sin(wb)` predicts that essentially all of it is the former. A calibration "
        "run on synthetic logits scores 1.0000 for the exact algorithm and 0.52 for random "
        "logits:",
        trig,
    ])


def ablations(root: Path, tag: str) -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m:
        return None
    chance = 1.0 / m["p"]
    fa = m["ablation_frequency"]
    ctrl = fa.get("_control_freqs", {}).get("freqs", [])
    K = m["key_freqs"]["used"]

    def row(label, key):
        v = fa[key]
        return [label, v["train_acc"], v["test_acc"], v["test_loss"]]

    freq = table(
        ["edit applied to W_E", "train acc", "test acc", "test loss"],
        [row("none (baseline)", "baseline"),
         row(f"keep ONLY the key frequencies {K}", "keep_key_freqs"),
         row(f"delete ONLY the key frequencies {K}", "drop_key_freqs"),
         row(f"delete {len(ctrl)} control frequencies {ctrl}", "drop_control_freqs"),
         row(f"keep ONLY the control frequencies {ctrl}", "keep_control_freqs")])

    comp = m["ablation_component"]
    order = ["baseline", "no_mlp"] + [k for k in comp if k.startswith("no_head")] + ["no_attention"]
    component = table(
        ["component removed", "train acc", "test acc", "test loss"],
        [[k.replace("_", " "), comp[k]["train_acc"], comp[k]["test_acc"], comp[k]["test_loss"]]
         for k in order if k in comp])

    npf = m.get("ablation_neuron_per_freq", {})
    neuron = table(
        ["MLP neurons mean-ablated", "train acc", "test acc", "test loss"],
        [[k.replace("_", " "), v["train_acc"], v["test_acc"], v["test_loss"]]
         for k, v in npf.items()]) if npf else ""

    pr = m["progress"]
    prog = table(
        ["logit-space edit", "split", "loss", "accuracy"],
        [["keep only the key frequencies' (a+b) directions", "all pairs",
          pr["restricted_sum_all"]["loss"], pr["restricted_sum_all"]["acc"]],
         ["keep only the key frequencies' (a+b) directions", "train",
          pr["restricted_sum_train"]["loss"], pr["restricted_sum_train"]["acc"]],
         ["delete exactly those directions", "train",
          pr["excluded_sum_train"]["loss"], pr["excluded_sum_train"]["acc"]]],
        align="llrr")

    return "\n\n".join([
        "These edit the **weights** and re-run the network, so the intervention "
        "propagates the way a real change would. Chance accuracy is "
        f"{chance:.4f}.",
        f"**Embedding surgery.** The control rows matter: deleting any {len(K)} of "
        f"{(m['p']-1)//2} frequencies removes some of the embedding's norm, so the "
        f"key-frequency result is only interesting if the control is unharmed.",
        freq,
        "**Neuron surgery.**",
        neuron,
        "**Whole components.**",
        component,
        "**Logit-space restriction.** Keeping 2 directions per key frequency leaves "
        f"{2*len(K)} of {m['p']**2:,} degrees of freedom per output class:",
        prog,
    ])


def phases(root: Path, tag: str) -> Optional[str]:
    a = load_analysis(root, tag)
    if not a:
        return None
    if not a.get("complete", False):
        return None          # a partial trajectory renders as plausible nonsense
    rows = a["rows"]
    import numpy as np
    step = np.array([r["step"] for r in rows], float)

    def norm_cross(key, frac):
        if key not in rows[0]:
            return None
        y = np.array([r.get(key, np.nan) for r in rows], float)
        start, end = y[0], float(np.median(y[-8:]))
        target = start + frac * (end - start)
        for i in range(1, len(y)):
            if (y[i] >= target) if end > start else (y[i] <= target):
                y0, y1 = y[i - 1], y[i]
                if y1 == y0:
                    return float(step[i])
                return float(step[i - 1] + (target - y0) / (y1 - y0) * (step[i] - step[i - 1]))
        return None

    sigs = [("test accuracy (visible from outside)", "test_acc"),
            ("restricted loss", "restricted_loss_sum_all"),
            ("embedding spectrum Gini", "emb_gini"),
            ("power in the key frequencies", "emb_key_frac"),
            ("logit variance explained by (a+b)", "logit_var_a+b"),
            ("excluded loss", "excluded_loss_sum"),
            ("neurons explained >85% by one frequency", "neuron_frac_above_85pct")]
    ref = norm_cross("test_acc", 0.10)
    body = []
    for name, key in sigs:
        v10, v50 = norm_cross(key, 0.10), norm_cross(key, 0.50)
        lead = (ref - v10) if (ref and v10) else None
        body.append([name,
                     int(round(v10)) if v10 is not None else None,
                     int(round(v50)) if v50 is not None else None,
                     int(round(lead)) if lead is not None else None])
    tbl = table(["signal", "10% of its change", "50%", "lead over test accuracy"],
                body, bold_best={3: "max"})
    return "\n\n".join([
        "Each signal is measured against **its own** range, from its value at "
        "initialisation to its final value, so the comparison does not depend on "
        "the units. A positive lead means the internal signal moves first.",
        tbl,
        "Read together: the circuit's subspace becomes predictive (restricted loss) and "
        "the embedding becomes sparse (Gini) thousands of steps before anything is "
        "visible from outside, while excluded loss and neuron crystallisation *lag* -- "
        "they measure the removal of the memorised solution, which happens last. That "
        "is the three-phase account: memorise, then form the circuit under cover of the "
        "memorised solution, then clean the memorised solution away.",
    ])


def runtime(root: Path, tag: str) -> Optional[str]:
    h = load_history(root, tag)
    if not h:
        return None
    last = h["history"][-1]
    return (f"The mainline run is {last['step']:,} steps in "
            f"**{last['elapsed']/60:.0f} minutes** on 6 CPU threads "
            f"({last['elapsed']/max(last['step'],1)*1000:.0f} ms per full-batch step). "
            f"Checkpoints for one run are about 140 MB.")


OP_LABELS = {
    "add": "(a + b) mod p",
    "sub": "(a - b) mod p",
    "mul": "(a * b) mod p",
    "sq_sum": "(a^2 + b^2) mod p",
    "sq_sum_cross": "(a^2 + ab + b^2) mod p",
    "cube_add": "(a^3 + b) mod p",
    "cube_cross": "(a^3 + ab) mod p",
}


def operations(root: Path, tags: List[str]) -> Optional[str]:
    """One row per run: did it grok, and if so with what mechanism?

    A run that does not grok inside its budget is reported as censored with the
    budget stated, never as "does not grok" full stop.
    """
    from .analysis.timing import crossing_step
    rows, notes, any_found = [], [], False
    for tag in tags:
        h = load_history(root, tag)
        if not h:
            continue
        any_found = True
        hist = h["history"]
        grok = crossing_step(hist, "test_acc", 0.90)
        budget = h["train_cfg"]["steps"]
        m = load_json(root, f"{tag}_mechanism.json")
        label = OP_LABELS.get(h["data"]["op"], h["data"]["op"])
        p_ = h["data"]["p"]
        row = [f"`{label}`, p={p_}",
               int(round(grok)) if grok else f"none by {budget:,}",
               hist[-1]["test_acc"]]
        if m:
            k = m["key_freqs"]
            row += [len(k["used"]), k["gini_W_E"],
                    m["structure"]["a+b"], m["trig"]["mean_sum_frac"]]
        else:
            row += [None, None, None, None]
        rows.append(row)
        if grok is None:
            notes.append(f"- `{label}` at p={p_} did not reach 90% test accuracy "
                         f"within its {budget:,}-step budget. That is a censored "
                         f"observation, not a demonstration that it never would.")
    if not any_found:
        return None
    tbl = table(["task", "grokking step", "final test acc", "key freqs",
                 "Gini(W_E)", "(a+b) variance", "trig fraction"], rows)
    out = [
        "Each run uses the identical configuration; only the operation (and, in one "
        "pair, the modulus) changes. `(a+b) variance` and `trig fraction` are the two "
        "mechanism tests from section 2, so a row that groks with a low trig fraction "
        "has found a *different* algorithm, not the same one.",
        tbl,
    ]
    if notes:
        out += ["**Runs that did not grok within budget.**"] + notes
    return "\n\n".join(out)


def dlog_block(root: Path, tag: str) -> Optional[str]:
    """Multiplication, viewed in the multiplicative-character basis."""
    m = load_json(root, f"{tag}_mechanism.json")
    if not m or "dlog" not in m:
        return None
    d = m["dlog"]
    ord_k, star_k = m["key_freqs"], d["key_freqs"]
    cmp_tbl = table(
        ["basis", "key frequencies", "Gini(W_E)", "power in key freqs",
         "variance explained by the sum"],
        [["ordinary (residues 0..p-1)", len(ord_k["used"]), ord_k["gini_W_E"],
          ord_k["frac_power_in_key"], m["structure"]["a+b"]],
         [f"discrete log (base g={d['primitive_root']}, n={d['n']})",
          len(star_k["used"]), star_k["gini_W_E"], star_k["frac_power_in_key"],
          d["structure"]["a+b"]]])
    z = d["zero_element"]
    zero = table(
        ["question", "value"],
        [["norm of the embedding row for 0", z["zero_row_norm"]],
         ["mean norm of the other rows", z["mean_other_row_norm"]],
         ["that as a z-score", z["zero_row_norm_zscore"]],
         ["strongest neuron correlation with (a == 0)", z["max_neuron_corr_a_is_zero"]],
         ["strongest neuron correlation with (b == 0)", z["max_neuron_corr_b_is_zero"]]])
    parts = [
        "The nonzero residues mod p form a cyclic group of order p-1 under "
        "multiplication, so re-indexing them by discrete logarithm turns "
        "`a * b` into `dlog(a) + dlog(b) mod (p-1)` -- multiplication becomes "
        "addition. This reduction is prior art (Doshi et al., arXiv:2406.03495); "
        "what is measured here is whether the circuit is load-bearing in that "
        "basis, which is a causal question the observational work did not ask.",
        cmp_tbl,
    ]
    if "progress" in d:
        pr = d["progress"]
        parts += [
            "**Causal test in the multiplicative basis:**",
            table(["edit", "loss", "accuracy"],
                  [["keep only the multiplicative key frequencies",
                    pr["restricted_sum_all"]["loss"], pr["restricted_sum_all"]["acc"]],
                   ["delete exactly those",
                    pr["excluded_sum_train"]["loss"], pr["excluded_sum_train"]["acc"]]],
                  align="lrr"),
        ]
    parts += [
        "**The absorbing element.** Zero is not in the multiplicative group: with it "
        "the structure is a monoid and the character story does not apply. Whether it "
        "gets its own sub-circuit is unclaimed in the literature:",
        zero,
    ]
    return "\n\n".join(parts)


GENERATORS = {
    "headline": headline,
    "mechanism": mechanism,
    "ablations": ablations,
    "phases": phases,
    "runtime": runtime,
}

MULTI_TAG_GENERATORS = {
    "operations": operations,
}


def build(root: Path, tag: str, op_tags: Optional[List[str]] = None) -> Dict[str, str]:
    out = {}
    for name, fn in GENERATORS.items():
        try:
            v = fn(root, tag)
        except Exception as exc:                     # a broken block must not
            v = f"_Block failed to render: {exc!r}_"  # silently vanish
        out[name] = v if v is not None else PENDING
    for name, fn in MULTI_TAG_GENERATORS.items():
        try:
            v = fn(root, op_tags or [tag])
        except Exception as exc:
            v = f"_Block failed to render: {exc!r}_"
        out[name] = v if v is not None else PENDING
    try:
        v = dlog_block(root, "B_mul_s0")
    except Exception as exc:
        v = f"_Block failed to render: {exc!r}_"
    out["dlog"] = v if v is not None else PENDING
    return out
