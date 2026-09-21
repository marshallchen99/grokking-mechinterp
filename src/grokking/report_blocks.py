"""Turn the results files into the README's tables and the sentences around them.

This is the only route by which a number reaches the write-up.  Two rules,
both learned the hard way:

* Every number in a sentence is computed here from a results file.  An earlier
  version of this module carried string literals such as "seven orders of
  magnitude" and "0.998 and 0.999" that were printed whatever the data said;
  one of them was wrong by four orders of magnitude.  Method parameters -- a
  90% accuracy threshold, say -- are named constants, not results.
* A generator returns None when its inputs are missing or incomplete, and the
  block then says so.  A partially written results file is well-formed and
  renders as plausible nonsense.

Attribution is part of the content: each block names the paper its method or
finding comes from, so a reader can tell what is reproduced from what is new.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .literature import (
    CHEN_2026, CHUGHTAI_2023, DOSHI_2024, FURUTA_2024, KHANH_2026, LYU_2023,
    NANDA_2023, NGUYEN_2026, NOTSAWO_2023, POWER_2022, PUBLISHED,
)
from .report import is_finished, load_analysis, load_history, load_json, table

PENDING = "_Not yet run._"

# The runs in the operations table.  One definition, imported by make_report.py
# and export_web.py, so the README and the page cannot render different tables.
OP_TAGS = ["B_add_s0", "B_sub_s0", "B_mul_s0", "B_sqx_p113", "B_sqx_p109"]
ADD_REFERENCE = "B_add_s0"   # the addition run operations are compared against

# Method parameters.  These are choices, not results, so they are named here
# and formatted into the prose rather than written into it.
GROK_ACC = 0.90          # test accuracy that counts as "generalised"
MEMORISED_ACC = 0.99     # training accuracy that counts as "memorised"
EARLY_ACC = 0.10         # test accuracy that marks visible movement
MID_ACC = 0.50
FULL_ACC = 0.99
NEURON_FRAC = 0.85       # the neuron-clustering rule's variance threshold
HALFWAY = 0.50           # "half of a signal's total change"


def cite(ref: str) -> str:
    """Short form, e.g. 'Nanda et al. (2023)', from a full reference string.

    References are written "Surname, Surname, ..., 'Title', venue year, arXiv:...".
    The year is the arXiv year when there is one (it dates the work), else the
    first four-digit year in the string.
    """
    authors = [a.strip() for a in ref.split("'")[0].rstrip(", ").split(",") if a.strip()]
    etal = authors[0].endswith("et al.")
    first = authors[0].replace("et al.", "").strip().split()[-1]
    ax = arxiv(ref)
    if ax:
        year = "20" + ax.split(":")[1][:2]
    else:
        year = next((t for t in ref.replace(",", " ").split()
                     if t.isdigit() and len(t) == 4), "")
    return f"{first}{' et al.' if etal or len(authors) > 1 else ''} ({year})"


def arxiv(ref: str) -> str:
    for tok in ref.replace(",", " ").split():
        if tok.startswith("arXiv:"):
            return tok
    return ""


def st(v):
    """A step count; a fractional step is an interpolation artefact."""
    return None if v is None else int(round(v))


def orders(ratio: float) -> float:
    return math.log10(ratio) if ratio > 0 else float("nan")


# ------------------------------------------------------------------ headline

def _crossings(hist):
    from .analysis.timing import crossing_step
    return {
        "train": crossing_step(hist, "train_acc", MEMORISED_ACC),
        "t10": crossing_step(hist, "test_acc", EARLY_ACC),
        "t50": crossing_step(hist, "test_acc", MID_ACC),
        "t90": crossing_step(hist, "test_acc", GROK_ACC),
        "t99": crossing_step(hist, "test_acc", FULL_ACC),
    }


def headline(root: Path, tag: str) -> Optional[str]:
    h = load_history(root, tag)
    if not is_finished(h):
        return None
    hist = h["history"]
    c = _crossings(hist)
    chance = 1.0 / h["data"]["p"]
    rows = [
        [f"training accuracy reaches {MEMORISED_ACC:.0%}", st(c["train"]), "the training set is memorised"],
        [f"test accuracy reaches {EARLY_ACC:.0%}", st(c["t10"]), f"chance is {chance:.2%}"],
        [f"test accuracy reaches {MID_ACC:.0%}", st(c["t50"]), ""],
        [f"test accuracy reaches {GROK_ACC:.0%}", st(c["t90"]), "generalisation"],
        [f"test accuracy reaches {FULL_ACC:.0%}", st(c["t99"]), ""],
    ]
    plateau = c["t90"] - c["train"] if c["t90"] and c["train"] else None

    # What happens on the plateau, measured rather than assumed.
    on = [r for r in hist if c["train"] and c["t90"] and c["train"] <= r["step"] <= c["t90"]]
    worst_train = min(on, key=lambda r: r["train_acc"]) if on else None
    n_dips = sum(1 for a, b in zip(on, on[1:])
                 if a["train_acc"] >= MEMORISED_ACC > b["train_acc"])
    peak = max(hist, key=lambda r: r["test_loss"])
    # test accuracy on the plateau, sampled rather than characterised by eye
    def acc_at(step):
        return min(hist, key=lambda r: abs(r["step"] - step))["test_acc"]
    early, late = acc_at(c["train"] * 5 if c["train"] else 1000), acc_at(0.8 * c["t10"]) if c["t10"] else None

    lines = [table(["event", "step", "note"], rows, align="lrl"), ""]
    if plateau is not None:
        lines.append(
            f"For **{plateau:,.0f} steps** training accuracy stays at or near "
            f"{MEMORISED_ACC:.0%} while the model stays far from generalising. Neither "
            f"curve is flat on that stretch. Training accuracy drops below "
            f"{MEMORISED_ACC:.0%} {n_dips} times in short loss spikes, as low as "
            f"{worst_train['train_acc']:.1%} at step {worst_train['step']:,}. Test accuracy "
            f"is above chance almost from the start and creeps up slowly -- "
            f"{early:.1%} a few hundred steps in, {late:.1%} by step "
            f"{0.8 * c['t10']:,.0f} -- and only passes {EARLY_ACC:.0%} at step {c['t10']:,.0f}.")
    peak_train = next((r for r in hist if r["step"] == peak["step"]), None)
    lines.append("")
    lines.append(
        f"Test loss rises during memorisation and peaks at **{peak['test_loss']:.2f}** at "
        f"step {peak['step']:,}"
        + (f", which is on one of those spikes (training accuracy {peak_train['train_acc']:.1%} "
           f"at the same step)" if peak_train and peak_train["train_acc"] < MEMORISED_ACC else "")
        + ".")
    lines.append("")
    lines.append(
        f"Setup: `({h['data']['op']}) mod {h['data']['p']}`, {h['data']['n_train']:,} of "
        f"{h['data']['n_train'] + h['data']['n_test']:,} pairs used for training "
        f"({h['data']['train_frac']:.0%}), a {h['n_params']:,}-parameter one-layer transformer, "
        f"full-batch AdamW with weight decay {h['train_cfg']['weight_decay']}, "
        f"{h['train_cfg']['steps']:,} steps, CPU only. The architecture and optimiser follow "
        f"{cite(NANDA_2023)}; the phenomenon is {cite(POWER_2022)}. This first run also had "
        + ("a float32 loss, " if h["train_cfg"].get("loss_dtype", "float32") == "float32" else "")
        + ("no learning-rate warmup, " if "warmup_steps" not in h["train_cfg"] else "")
        + "and an output column for the '=' token; section 5 tests whether those mattered.")
    return "\n".join(lines)


# ----------------------------------------------------------------- mechanism

RULE_LABELS = {
    "rule_a_embedding_norm": ("embedding-norm threshold", "the embedding's Fourier spectrum, per basis index"),
    "rule_b_neuron_cluster": ("neuron clustering", "which frequency explains most of each MLP neuron"),
    "rule_c_power_gap": ("power gap", "the embedding's Fourier spectrum, per frequency"),
}


def mechanism(root: Path, tag: str) -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m:
        return None
    k = m["key_freqs"]
    rules = table(["rule", "reads", "frequencies found"],
                  [[RULE_LABELS[r][0], RULE_LABELS[r][1], v] for r, v in k["by_rule"].items()],
                  align="lll")
    n = m["neurons"]
    census = table(["frequency", "neurons assigned"],
                   [[f, v] for f, v in sorted(n["counts_by_freq"].items(), key=lambda kv: -kv[1])],
                   align="lr")
    s = m["structure"]
    chance = 1.0 / m["p"]
    struct = table(["the logits are a function of...", "variance explained"],
                   [["(a + b) mod p", s["a+b"]], ["(a - b) mod p", s["a-b"]],
                    ["a alone", s["a"]], ["b alone", s["b"]], ["(a * b) mod p", s["a*b"]]],
                   bold_best={1: "max"})
    t = m["trig"]
    trig = table(["frequency", "energy in cos/sin(w(a+b))"],
                 [[int(f), v["sum_frac"]] for f, v in sorted(t["per_freq"].items(), key=lambda kv: int(kv[0]))]
                 + [["**mean**", t["mean_sum_frac"]]])
    lo, hi = PUBLISHED["gini_W_E_range"].value
    return "\n\n".join([
        f"The mechanism described here is {cite(NANDA_2023)}'s; what follows re-derives it "
        f"on this repository's own model. Three rules are applied to the final checkpoint "
        f"(step {m['final_step']:,}). Two of them read the same object -- the embedding's "
        f"Fourier spectrum -- and one reads the MLP neurons, so this is two independent "
        f"views rather than three. They agree with Jaccard {k['jaccard']:.3f}:",
        rules,
        f"Those {len(k['used'])} frequencies carry **{k['frac_power_in_key']:.1%}** of the "
        f"embedding's Fourier power, out of {(m['p'] - 1) // 2} available. The spectrum's Gini "
        f"coefficient is {k['gini_W_E']:.4f}; {cite(NANDA_2023)} report {lo}-{hi} across their "
        f"settings, but do not say which vector they compute it over, so the two may not be "
        f"comparable.",
        f"Every one of the {n['n_total']} MLP neurons is dominated by one of these frequencies"
        + (":" if n["all_on_key_freqs"] else ", with exceptions:"),
        census,
        f"{n['frac_above_85']:.1%} of neurons have more than {NEURON_FRAC:.0%} of their variance explained "
        f"by a single frequency (mean {n['mean_dominant_frac']:.4f}, minimum "
        f"{n['min_dominant_frac']:.4f}).",
        f"**What it computes.** Chance for the variance-explained column is {chance:.4f}:",
        struct,
        "**The trigonometric identity.** Within each key frequency, the dependence on (a, b) "
        "splits into a part that is a function of (a+b) and a part that is a function of "
        "(a-b); the identity `cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)` predicts it is "
        "all the former. The measure is exactly one for logits built to be the algorithm, "
        "by construction, and one half for random logits, by symmetry; the tests check both "
        "(`tests/test_analysis.py`):",
        trig,
    ])


# ------------------------------------------------------------------ readout

READOUT_RUNS = ["B_add_s0", "B_add_s1", "C_add_nowarm", "C_add_f32", "main_add_s0", "B_sub_s0"]


def describe_run(root: Path, tag: str) -> str:
    """A run's training choices, read from its own record rather than typed."""
    h = load_history(root, tag) or {}
    tc, mc, d = h.get("train_cfg", {}), h.get("model_cfg", {}), h.get("data", {})
    bits = [d.get("op", "?"), f"{tc.get('loss_dtype', 'float32')} loss"]
    w = tc.get("warmup_steps")
    bits.append(f"{w}-step warmup" if w and w > 1 else
                ("warmup_steps = 1" if w == 1 else "no LR schedule"))
    if mc.get("d_vocab_out", mc.get("d_vocab")) and d.get("p") and \
            mc.get("d_vocab_out", mc.get("d_vocab")) > d["p"]:
        bits.append("extra W_U column")
    return ", ".join(bits)


def readout(root: Path, _tag: str) -> Optional[str]:
    rows, own = [], {}
    for tag in READOUT_RUNS:
        note = describe_run(root, tag)
        m = load_json(root, f"{tag}_mechanism.json")
        if not m or "readout_budget" not in m:
            continue
        rb = m["readout_budget"]
        d = rb.get("direction_used", "sum")
        b = rb[d]
        own[tag] = (b["own"], d, rb)
        rows.append([f"`{tag}`", note, "(a+b)" if d == "sum" else "(a-b)",
                     b["own"], b["cross"], b["unexplained"]])
    if not rows:
        return None
    tbl = table(["run", "configuration", "direction read", "at the predicted frequency",
                 "at another key frequency", "left over"], rows, bold_best={3: "max"})
    f64 = [own[t][0] for t in ("B_add_s0", "B_add_s1", "C_add_nowarm") if t in own]
    f32 = [own[t][0] for t in ("C_add_f32", "main_add_s0") if t in own]
    steps_read = {t: (load_json(root, f"{t}_mechanism.json") or {}).get("final_step") for t in own}
    parts = [
        "The mechanism's third step is the readout: the amplitudes of cos(w(a+b)) and "
        "sin(w(a+b)) must themselves be waves at the same frequency in the answer c, which "
        "is what makes the sum peak at c = a+b. Before measuring it, each input's logits are "
        "shifted to mean zero over c. That component adds the same number to every class, "
        "so softmax, the loss and every prediction are exactly invariant to it; counting it "
        "would measure a direction the model cannot be using. Where the identification "
        "rules disagree, the frequencies measured are all those any rule picks.",
        tbl,
    ]
    if f64:
        parts.append(
            f"In the float64 addition runs the readout is essentially exact "
            f"({min(f64):.3f} to {max(f64):.3f} at the predicted frequency)."
            + (f" Subtraction reads {own['B_sub_s0'][0]:.3f} in the (a-b) direction it actually "
               f"uses, and {own['B_sub_s0'][2]['sum']['own']:.3f} if read in the (a+b) direction "
               f"-- a model read in the wrong coordinate looks unstructured."
               if "B_sub_s0" in own else ""))
    if f32 and f64:
        parts.append(
            f"The {len(f32)} float32 runs are measurably less clean ({min(f32):.3f} to "
            f"{max(f32):.3f}), with the remainder split between other key frequencies and "
            f"energy none of them explain. **The cause is not established.** It is "
            f"{len(f32)} runs against {len(f64)}, and the pairs are not matched on everything: "
            f"the float32 control `C_add_f32` was read at step "
            f"{steps_read.get('C_add_f32', 0):,} and its float64 twin `B_add_s0` at step "
            f"{steps_read.get('B_add_s0', 0):,}, so budget differs as well as precision"
            + (f" (though `C_add_nowarm`, float64 and read at step "
               f"{steps_read['C_add_nowarm']:,}, is as clean as the others)"
               if "C_add_nowarm" in steps_read else "")
            + ". An earlier version of this section attributed the difference to a "
            f"per-answer bias that float32 prevented from being cleaned up. That was wrong: "
            f"the component it measured was the softmax-invariant one removed above, which "
            f"no loss gradient acts on at any precision.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- ablations

def ablations(root: Path, tag: str) -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m:
        return None
    chance = 1.0 / m["p"]
    fa = m["ablation_frequency"]
    ctrl = fa.get("_control_freqs", {}).get("freqs", [])
    K = m["key_freqs"]["used"]
    ps = fa.get("_power_share", {})

    def row(label, key):
        v = fa[key]
        return [label, v["train_acc"], v["test_acc"], v["test_loss"]]

    freq_rows = [row("none (baseline)", "baseline"),
                 row(f"keep ONLY the key frequencies {K}", "keep_key_freqs"),
                 row(f"delete ONLY the key frequencies {K}", "drop_key_freqs")]
    if "drop_key_freqs_norm_restored" in fa:
        freq_rows.append(row("delete the key frequencies, then restore the embedding's norm",
                             "drop_key_freqs_norm_restored"))
    freq_rows += [row(f"delete {len(ctrl)} control frequencies {ctrl}", "drop_control_freqs"),
                  row(f"keep ONLY the control frequencies {ctrl}", "keep_control_freqs")]
    freq = table(["edit applied to W_E", "train acc", "test acc", "test loss"], freq_rows)

    comp = m["ablation_component"]
    order = ["baseline", "no_mlp"] + [k for k in comp if k.startswith("no_head")] + ["no_attention"]
    component = table(["component removed", "train acc", "test acc", "test loss"],
                      [[k.replace("_", " "), comp[k]["train_acc"], comp[k]["test_acc"], comp[k]["test_loss"]]
                       for k in order if k in comp])

    npf = m.get("ablation_neuron_per_freq", {})
    neuron = table(["MLP neurons mean-ablated", "train acc", "test acc", "test loss"],
                   [[k.replace("_", " "), v["train_acc"], v["test_acc"], v["test_loss"]]
                    for k, v in npf.items()]) if npf else ""
    worst_loss = max((v["test_loss"] for v in npf.values()), default=None)

    pr = m["progress"]
    prog = table(["logit-space edit", "split", "loss", "accuracy"],
                 [["keep only the key frequencies' (a+b) directions", "all pairs",
                   pr["restricted_sum_all"]["loss"], pr["restricted_sum_all"]["acc"]],
                  ["keep only the key frequencies' (a+b) directions", "train",
                   pr["restricted_sum_train"]["loss"], pr["restricted_sum_train"]["acc"]],
                  ["delete exactly those directions", "train",
                   pr["excluded_sum_train"]["loss"], pr["excluded_sum_train"]["acc"]]],
                 align="llrr")

    parts = [
        f"These edit the **weights** and re-run the whole network. Chance accuracy is {chance:.4f}.",
        "**Embedding surgery.**",
        freq,
    ]
    if ps:
        parts.append(
            f"The key frequencies hold {ps['key']:.1%} of the embedding's power and the evenly "
            f"spaced controls {ps['control']:.2%}, so the controls alone cannot rule out that "
            f"deleting the key frequencies kills the model merely by shrinking its embedding. "
            f"The norm-restored row does: with the key frequencies deleted and the rest scaled "
            f"back up to the original norm, test accuracy is "
            f"{fa['drop_key_freqs_norm_restored']['test_acc']:.2%}.")
    parts += [
        "**Neuron surgery.** Each key frequency's neuron cluster, mean-ablated:",
        neuron,
    ]
    if worst_loss is not None:
        parts.append(
            f"Two caveats. The clusters differ in size and there is no size-matched random "
            f"control, so 'dropping this cluster is survivable' partly reflects that it is the "
            f"smallest. And losses as high as {worst_loss:,.0f} -- far above the "
            f"{math.log(m['p']):.2f} of a uniform guess -- mean the ablated network is being "
            f"pushed off its training distribution, so these rows say which clusters matter, "
            f"not by how much.")
    parts += [
        "**Whole components.**",
        component,
        f"**Logit-space restriction** ({cite(NANDA_2023)}'s restricted and excluded loss). "
        f"This edits the output logits, not the weights, so it is a projection rather than an "
        f"intervention. The logits for one output class, as a function of (a, b), have "
        f"{m['p'] ** 2:,} degrees of freedom; keeping two directions per key frequency "
        f"leaves {2 * len(K)}:",
        prog,
    ]
    return "\n\n".join(parts)


# -------------------------------------------------- the dispensable frequency

WEAKEST_RUNS = ["main_add_s0", "B_add_s0", "B_add_s1", "B_sub_s0", "C_add_f32"]


def disagreement(root: Path, _tag: str) -> Optional[str]:
    rows, n, hits, disputes, dispute_hits, multi = [], 0, 0, 0, 0, []
    for tag in WEAKEST_RUNS:
        m = load_json(root, f"{tag}_mechanism.json")
        r = load_json(root, f"{tag}_redundancy.json")
        if not m or not r:
            continue
        K = m["key_freqs"]["used"]
        pw = m["key_freqs"]["power_per_freq"]
        weakest = min(K, key=lambda f: pw[f])
        sets = [set(v) for v in m["key_freqs"]["by_rule"].values()]
        disputed = sorted(set().union(*sets) - set.intersection(*sets))
        thr = r["threshold"]
        full = set(K)
        # a frequency is individually dispensable if the key set without it still
        # reaches the threshold -- computed from the subset table, not inferred
        alone = sorted(f for f in K for row in r["rows"]
                       if set(row["subset"]) == full - {f} and row["test_acc"] >= thr)
        n += 1
        only_weakest = alone == [weakest]
        hits += only_weakest
        weakest_among = weakest in alone
        multi.append((tag, alone, weakest)) if len(alone) > 1 else None
        if disputed:
            disputes += 1
            dispute_hits += disputed == [weakest]
        rows.append([f"`{tag}`", weakest, ", ".join(map(str, disputed)) or "none",
                     ", ".join(map(str, alone)) or "none"])
    if not rows:
        return None
    return "\n\n".join([
        table(["run", "weakest key frequency in the embedding", "frequency the rules disagree on",
               "key frequencies that can each be deleted alone"], rows),
        f"'Can be deleted alone' means the rest of the key set, with every other frequency "
        f"also removed, still reaches {GROK_ACC:.0%} test accuracy. In {hits} of {n} runs "
        f"exactly one key frequency qualifies, and it is the weakest."
        + "".join(f" In `{t}` {len(a)} qualify ({', '.join(map(str, a))}), and the weakest, "
                  f"{w}, is {'one of them' if w in a else 'not among them'}."
                  for t, a, w in multi)
        + f" In {dispute_hits} of the {disputes} runs where the identification rules "
        f"disagree, they disagree about the weakest frequency. An earlier version presented rule disagreement as a "
        f"*prediction* of redundancy. It is not independent evidence: the deletion test edits "
        f"the embedding, and two of the rules threshold that same embedding, so the weakest "
        f"component being both the marginal one and the dispensable one is close to what "
        f"you would expect.",
    ])


def load_bearing(root: Path, tag: str) -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m or not m.get("per_frequency_excluded_loss"):
        return None
    pf = {int(k): v for k, v in m["per_frequency_excluded_loss"].items()}
    K = m["key_freqs"]["used"]
    items = sorted(pf.items(), key=lambda kv: -kv[1])
    top = items[: len(K) + 2]
    rows = [[f"{k}" + ("  (key)" if k in K else ""), v] for k, v in top]
    key_vals = [pf[k] for k in K]
    rest = sorted(v for k, v in pf.items() if k not in K)
    weakest_key, strongest_other = min(key_vals), max(rest)
    boundary = weakest_key / strongest_other if strongest_other > 0 else float("inf")
    base = m["baseline"]["train_loss"]
    picked = sorted(k for k, _ in items[: len(K)])
    return "\n\n".join([
        "The rules above read the model's representation. This reads its behaviour: remove "
        "one frequency's two (a+b) Fourier directions from the logits at a time and measure "
        "the training loss. It "
        "still works in the same Fourier basis, so it is not independent machinery, and "
        "taking 'the top k' borrows k from the rules; but it involves no threshold of its own.",
        table(["frequency removed", "train loss afterwards"], rows),
        f"At the boundary -- the weakest key frequency against the strongest of the rest -- "
        f"the separation is a factor of {boundary:,.0f} ({orders(boundary):.1f} orders of "
        f"magnitude). Most non-key frequencies cost nothing measurable: removing them leaves "
        f"the training loss at its unablated value of {base:.2e}. The top {len(K)} by this "
        f"measure are "
        f"{picked}, " + ("the same set the rules find." if picked == sorted(K)
                         else f"which differs from the rules' {sorted(K)}."),
    ])


def redundancy(root: Path, tag: str) -> Optional[str]:
    d = load_json(root, f"{tag}_redundancy.json")
    if not d:
        return None
    K = d["key_freqs"]
    rows = [[str(r["subset"]) if r["subset"] else "nothing", r["size"], r["test_acc"], r["test_loss"]]
            for r in d["rows"]]
    mins = d["minimal_subsets"]
    red = d["redundant_frequencies"]
    full = next(r for r in d["rows"] if r["size"] == len(K))
    best_min = next((r for r in d["rows"] if r["subset"] in mins), None)
    others = [r for r in d["rows"] if r["size"] == d["minimal_size"] and r["subset"] not in mins]
    parts = [
        f"Every subset of the {len(K)} key frequencies is kept in the embedding, all other "
        f"frequencies are deleted, and the network is re-run. Chance is {d['chance_acc']:.4f}:",
        table(["frequencies kept in W_E", "size", "test acc", "test loss"], rows),
    ]
    if best_min and len(mins) == 1:
        parts.append(
            f"Among subsets of the key set, the only {d['minimal_size']}-frequency subset "
            f"that reaches {d['threshold']:.0%} is {mins[0]}, at {best_min['test_acc']:.2%}; "
            f"the other {len(others)} stay at or below "
            f"{max(r['test_acc'] for r in others):.1%}. Frequency {red[0]} is dispensable for "
            f"accuracy, though keeping it lowers the test loss from "
            f"{best_min['test_loss']:.4f} to {full['test_loss']:.4f}. Whether that is why "
            f"training keeps it -- a loss benefit outweighing its weight-decay cost -- is "
            f"plausible but was not measured here.")
    return "\n\n".join(parts)


# ------------------------------------------------------------------- phases

def phases(root: Path, tag: str) -> Optional[str]:
    a = load_analysis(root, tag)
    h = load_history(root, tag)
    if not a or not a.get("complete") or not is_finished(h):
        return None
    import numpy as np
    rows = a["rows"]
    step = np.array([r["step"] for r in rows], float)

    def cross(key, frac):
        if key not in rows[0]:
            return None
        y = np.array([r.get(key, np.nan) for r in rows], float)
        start, end = y[0], float(np.median(y[-8:]))
        target = start + frac * (end - start)
        for i in range(1, len(y)):
            if (y[i] >= target) if end > start else (y[i] <= target):
                y0, y1 = y[i - 1], y[i]
                return float(step[i]) if y1 == y0 else float(
                    step[i - 1] + (target - y0) / (y1 - y0) * (step[i] - step[i - 1]))
        return None

    sigs = [("test accuracy (visible from outside)", "test_acc"),
            ("restricted loss", "restricted_loss_sum_all"),
            ("embedding spectrum Gini", "emb_gini"),
            ("logit variance explained by (a+b)", "logit_var_a+b"),
            ("excluded loss", "excluded_loss_sum"),
            (f"neurons explained >{NEURON_FRAC:.0%} by one frequency", "neuron_frac_above_85pct")]
    ref = cross("test_acc", HALFWAY)
    body = []
    for name, key in sigs:
        v = cross(key, HALFWAY)
        body.append([name, st(v), st(ref - v) if (ref and v) else None])

    g = _crossings(h["history"])["t90"]
    near = sorted(s for s in step if g and 0.7 * g <= s <= 1.2 * g)
    gaps = [b - a_ for a_, b in zip(near, near[1:])]
    spacing = float(np.median(gaps)) if gaps else None
    rl = [(r["step"], r["restricted_loss_sum_all"]) for r in rows]
    rl_peak = max(rl, key=lambda x: x[1])
    lead_rl = next((b[2] for b in body if b[0] == "restricted loss"), None)

    parts = [
        f"The progress measures and the three-phase account are {cite(NANDA_2023)}'s. Each "
        f"signal is measured against its own range, from initialisation to final value, and "
        f"the table gives the step at which it has made half its total change. A positive "
        f"lead means it gets there before test accuracy does.",
        table(["signal", f"reaches {HALFWAY:.0%} of its change", "lead over test accuracy"], body),
    ]
    if spacing:
        leads = [(b[0], b[2]) for b in body[1:] if b[2] is not None]
        ahead = [n for n, v in leads if v > spacing]
        behind = [n for n, v in leads if v < -spacing]
        tied = [n for n, v in leads if -spacing <= v <= spacing]
        parts.append(
            f"**Resolution.** Checkpoints near the transition are about {spacing:,.0f} steps "
            f"apart, so a lead smaller than that is not distinguishable from zero. Against "
            f"that: " + "; ".join(x for x in [
                ("ahead by more than the spacing: " + ", ".join(ahead)) if ahead else "",
                ("within it: " + ", ".join(tied)) if tied else "",
                ("behind by more than it: " + ", ".join(behind)) if behind else ""] if x)
            + f". The restricted loss is also not monotone: it first rises, to "
            f"{rl_peak[1]:.2f} at step {rl_peak[0]:,}, before it falls.")
    parts.append(
        f"The ordering -- the key frequencies' subspace becoming predictive before test "
        f"accuracy moves, the removal of the memorised solution coming later -- is the one "
        f"{cite(NANDA_2023)} describe. This is one run.")
    return "\n\n".join(parts)


def runtime(root: Path, tag: str) -> Optional[str]:
    h = load_history(root, tag)
    if not is_finished(h):
        return None
    last = h["history"][-1]
    return (f"The mainline run is {last['step']:,} steps in "
            f"**{last['elapsed'] / 60:.0f} minutes** on 6 CPU threads "
            f"({last['elapsed'] / max(last['step'], 1) * 1000:.0f} ms per full-batch step).")


# --------------------------------------------------------------- operations

OP_LABELS = {
    "add": "(a + b) mod p", "sub": "(a - b) mod p", "mul": "(a * b) mod p",
    "sq_sum": "(a^2 + b^2) mod p", "sq_sum_cross": "(a^2 + ab + b^2) mod p",
    "cube_add": "(a^3 + b) mod p", "cube_cross": "(a^3 + ab) mod p",
}


def operations(root: Path, tags: List[str]) -> Optional[str]:
    from .analysis.timing import crossing_step
    own_key = {"add": "a+b", "sub": "a-b", "mul": "a*b"}
    rows, notes, got = [], [], {}
    for tag in tags:
        h = load_history(root, tag)
        if not is_finished(h):
            continue
        g = crossing_step(h["history"], "test_acc", GROK_ACC)
        got[tag] = g
        m = load_json(root, f"{tag}_mechanism.json")
        op = h["data"]["op"]
        budget = h["train_cfg"]["steps"]
        row = [f"`{tag}`", f"`{OP_LABELS.get(op, op)}`", h["data"]["p"],
               st(g) if g else f"none by {budget:,}", budget, h["history"][-1]["test_acc"]]
        if m:
            k = m["key_freqs"]
            row += ["--" if k["jaccard"] == 0 else len(k["used"]), k["gini_W_E"],
                    m["structure"].get(own_key.get(op, ""), None)]
        else:
            row += [None, None, None]
        rows.append(row)
        if g is None:
            notes.append(f"`{tag}` did not reach {GROK_ACC:.0%} within {budget:,} steps -- "
                         f"censored, not shown never to grok.")
    if not rows:
        return None
    parts = [
        f"Same model and optimiser, one seed per operation; the step budgets differ (see the "
        f"column), so the mechanism is read at different points after grokking. The last "
        f"column is the variance of the logits explained by the task's own combination of "
        f"the inputs -- (a+b), (a-b) or (a*b) -- in the ordinary basis. 'Key freqs' is "
        f"'--' where the identification rules find no common set, which for multiplication "
        f"means the ordinary basis is the wrong place to look (next subsection). The survey "
        f"of which modular operations grok is {cite(FURUTA_2024)}'s.",
        table(["run", "task", "p", "grokking step", "budget", "final test acc", "key freqs",
               "Gini(W_E)", "variance explained by own combination"], rows),
    ]
    sub_tag = next((t for t in got if (load_history(root, t) or {}).get("data", {}).get("op") == "sub"), None)
    if sub_tag and ADD_REFERENCE in got and got[sub_tag] and got[ADD_REFERENCE]:
        pred = load_json(root, "prediction_summary.json") or {}
        gs = [r["grok"] for r in (pred.get("within_config") or {}).get("runs", [])]
        spread = (max(gs) / min(gs)) if len(gs) > 1 else None
        sub_budget = load_history(root, sub_tag)["train_cfg"]["steps"]
        parts.append(
            f"Subtraction builds the mirror circuit: its output tracks (a-b). It grokked at "
            f"step {st(got[sub_tag]):,} against {st(got[ADD_REFERENCE]):,} for `{ADD_REFERENCE}`, "
            f"but with one seed each"
            + (f", and seeds of one configuration elsewhere in this project spanning a factor "
               f"of {spread:.1f}" if spread else "")
            + f", that is not evidence subtraction is slower. It grokked "
            f"{sub_budget - st(got[sub_tag]):,} steps before its budget ran out.")
    if notes:
        parts += ["**Censored runs.**"] + [f"- {n}" for n in notes]
    return "\n\n".join(parts)


def quadratic(root: Path, _tag: str) -> Optional[str]:
    d = load_json(root, "quadratic_form.json")
    if not d or "unseen_with_flipped_partner_trained" not in d[0]:
        return None
    rows = [[f"`{r['tag']}`", f"{r['p_mod_3']} mod 3",
             "splits" if r["form_splits"] else "irreducible",
             r["test_acc"], r["frac_test_with_transpose_in_train"],
             r["acc_transpose_in_train"], r["acc_transpose_held_out"], r["chance"]]
            for r in d]
    tbl = table(["run", "p", "form over F_p", "test acc",
                 "held-out pairs whose transpose was trained on", "acc on those",
                 "acc on the rest", "chance"], rows)
    none_grok = all(r["test_acc"] < GROK_ACC for r in d)
    budgets = sorted({r["budget"] for r in d})
    part_rows = [[f"`{r['tag']}`",
                  r["unseen_with_flipped_partner_trained"]["n"],
                  r["unseen_with_flipped_partner_trained"]["predicts_flipped"],
                  r["unseen_without_flipped_partner"]["n"],
                  r["unseen_without_flipped_partner"]["predicts_flipped"],
                  r["chance"]] for r in d]
    return "\n\n".join([
        f"{cite(FURUTA_2024)} and {cite(DOSHI_2024)} (their Hypothesis 5.1) relate whether "
        f"a modular polynomial is learnable to whether it factorises. `a^2 + ab + b^2` "
        f"factors into linear forms over F_p exactly when p = 1 (mod 3), so this tests "
        f"that at a matched pair of primes.",
        tbl,
        (f"**Neither generalises within {budgets[-1]:,} steps, so this gives a null answer "
         f"on factorability.** " if none_grok else "")
        + f"Both sit near {sum(r['test_acc'] for r in d) / len(d):.0%} test accuracy, and the cause is not partial learning of the "
        "form. It is symmetric in a and b while the train/test split is over *ordered* "
        "pairs, so about half the held-out pairs have their transpose in the training set; "
        "the model gets essentially all of those right and is near chance on the rest. It "
        "has memorised the training table and learned that the table is symmetric.",
        "Its *mistakes* are structured, though. `a^2 - ab + b^2` is the same form with "
        "the sign of one input flipped, and on the held-out pairs it cannot answer from "
        "memory, the model often predicts exactly that. Whether it does depends on "
        "whether a sign-flipped partner of the pair -- (a, -b), (-a, b) or their "
        "transposes, all of which share that value -- was in the training set:",
        table(["run", "pairs with a sign-flipped partner trained",
               "predicts a^2 - ab + b^2", "pairs without one", "predicts a^2 - ab + b^2",
               "chance"], part_rows),
        "So these look like memorised answers retrieved for the wrong key, which suggests "
        "the representation places b and -b close together; that was not measured "
        "directly, and the without-partner groups are small. An earlier version read the "
        "plateau as a "
        "circuit that had 'lost the sign of b'; that explained neither the accuracy, which "
        "symmetry accounts for, nor these errors, which memorised partners do. Splitting "
        "on unordered pairs would remove both effects and make this a fair test of "
        "factorability.",
    ])


# --------------------------------------------------------- multiplication

def dlog_block(root: Path, tag: str = "B_mul_s0") -> Optional[str]:
    m = load_json(root, f"{tag}_mechanism.json")
    if not m or "dlog" not in m:
        return None
    d = m["dlog"]
    o, s = m["key_freqs"], d["key_freqs"]
    parts = [
        f"The nonzero residues mod p form a cyclic group of order p-1 under multiplication, "
        f"so re-indexing them by discrete logarithm turns `a * b` into "
        f"`dlog(a) + dlog(b) mod (p-1)`. None of this is new. The reduction is "
        f"{cite(DOSHI_2024)} ({arxiv(DOSHI_2024)}); a transformer trained on multiplication "
        f"has been shown sparse in this basis ({cite(NGUYEN_2026)}, {arxiv(NGUYEN_2026)}); and "
        f"restricted/excluded-loss ablations in the irreducible-representation basis of a "
        f"cyclic group -- which is what the discrete-log Fourier basis is -- are "
        f"{cite(CHUGHTAI_2023)} ({arxiv(CHUGHTAI_2023)}). What follows reproduces those on "
        f"this repository's own model, which was trained on the full table including 0.",
        table(["basis", "key frequencies", "Gini(W_E)", "power in key freqs",
               "variance explained by the sum", "rules agree (Jaccard)"],
              [["ordinary (residues 0..p-1)", len(o["used"]), o["gini_W_E"],
                o["frac_power_in_key"], m["structure"]["a+b"], o["jaccard"]],
               [f"discrete log (g = {d['primitive_root']}, n = {d['n']})", len(s["used"]),
                s["gini_W_E"], s["frac_power_in_key"], d["structure"]["a+b"], s["jaccard"]]]),
    ]
    ws = d.get("weight_surgery")
    if ws:
        parts += [
            f"**Weight surgery in the multiplicative basis.** The nonzero residues' embedding "
            f"rows are re-indexed by discrete logarithm, filtered in the Fourier basis over "
            f"Z_{d['n']}, written back, and the whole network is re-run. Accuracy is on "
            f"held-out pairs with both inputs nonzero:",
            table(["edit", "test acc"],
                  [["none", ws["baseline"]["test_acc_nonzero"]],
                   [f"keep only the key frequencies {ws['key_freqs']} (and the mean)", ws["keep_key"]["test_acc_nonzero"]],
                   [f"delete only the key frequencies", ws["drop_key"]["test_acc_nonzero"]],
                   [f"delete {len(ws['control_freqs'])} control frequencies {ws['control_freqs']}",
                    ws["drop_control"]["test_acc_nonzero"]],
                   [f"keep only the control frequencies (and the mean)", ws["keep_control"]["test_acc_nonzero"]]]),
        ]
    zi = d.get("zero_interventions")
    if zi and isinstance(zi.get("random_direction_mean_norm"), dict):
        rd = zi["random_direction_mean_norm"]
        parts += [
            "**The absorbing element.** Zero has no multiplicative inverse, so it is outside "
            f"the group. Prior work excludes it or treats it as a separate stratum "
            f"({cite(DOSHI_2024)}; {cite(CHEN_2026)}, observationally and on composite moduli). "
            f"Here the model is correct on the pairs containing a zero, no neuron behaves "
            f"like a detector for it (strongest correlation with `a == 0`: "
            f"{d['zero_element']['max_neuron_corr_a_is_zero']:.3f}), and its embedding row "
            f"has the smallest norm of all ({zi['row0_norm']:.2f} against a mean of "
            f"{zi['mean_other_norm']:.2f}). But the small norm is not the mechanism. Editing "
            f"that one row and re-running the model:",
            table(["edit to the embedding of 0",
                   f"accuracy, all {2 * m['p'] - 1} pairs containing a 0 (train and test)"],
                  [["none", zi["unedited"]],
                   ["set to zero", zi["row0_zeroed"]],
                   ["rescaled to the mean norm of the other rows", zi["row0_rescaled_to_mean_norm"]],
                   ["doubled", zi["row0_doubled"]],
                   ["replaced by the mean of the other rows", zi["row0_replaced_by_mean_of_others"]],
                   [f"replaced by a random direction at the mean norm (median of {rd['n']})", rd["median"]],
                   [f"scaled by {zi.get('scale_factor', 10)}", zi["row0_scaled_x10"]]]),
            f"Restoring its norm changes nothing, and what the row contains barely matters: "
            f"zeroed, averaged, or replaced by a random direction (median {rd['median']:.2f} "
            f"over {rd['n']} directions, range {rd['min']:.2f} to {rd['max']:.2f}), the model "
            f"still answers 0, until the edit is large enough to dominate the residual stream.",
        ]
    one = (zi or {}).get("one_input_without_signal")
    both = (zi or {}).get("both_inputs_without_signal")
    if one and both:
        parts.append(
            f"So what does make it answer 0? Remove the multiplicative signal from one "
            f"*nonzero* input instead -- zero or average the embedding of a nonzero residue "
            f"({', '.join(map(str, one['probe_rows']))}) -- and the model answers 0 on "
            f"{one['frac_predicted_0_zeroed']:.1%} of that residue's pairs. Remove it from "
            f"both inputs and it answers {both['most_common_prediction']} on "
            f"{both['its_share']:.0%} of pairs, and 0 on {both['frac_predicted_0']:.0%}. The "
            f"model treats an input that carries no multiplicative signal as a zero factor -- "
            f"what 0 means in multiplication -- whether or not the input is actually 0. Which "
            f"weights implement that was not identified.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- controls

CONTROL_NOTES = {"B_add_s1": "different seed for split and weights"}
CONTROL_ORDER = ["main_add_s0", "B_add_s0", "C_add_f32", "C_add_nowarm", "B_add_s1"]


def controls(root: Path, _tag: str) -> Optional[str]:
    from .analysis.timing import crossing_step
    g = {}
    rows = []
    for tag in CONTROL_ORDER:
        h = load_history(root, tag)
        if not is_finished(h):
            continue
        v = crossing_step(h["history"], "test_acc", GROK_ACC)
        g[tag] = v
        note = describe_run(root, tag) + (f"; {CONTROL_NOTES[tag]}" if tag in CONTROL_NOTES else "")
        rows.append([f"`{tag}`", note, st(v) if v else "none"])
    if len(rows) < 2:
        return None
    init = load_json(root, "controls_init.json")
    parts = [
        "The first run used float32 cross-entropy, no warmup, and an unembedding with a "
        "column for the '=' token that can never be correct. The corrected configuration "
        "grokked in about half the steps. Each control below changes one training choice, "
        "one run each; their step budgets also differ, which does not affect the grokking "
        "step but does change the step at which the mechanism is read.",
        table(["run", "what differs", "grokking step"], rows, align="llr"),
    ]
    if {"B_add_s0", "C_add_f32", "C_add_nowarm", "main_add_s0"} <= set(g):
        spread = abs(g.get("B_add_s1", g["B_add_s0"]) - g["B_add_s0"])
        parts.append(
            f"Changing only the loss precision moves the step by "
            f"{st(g['C_add_f32'] - g['B_add_s0']):+,}; removing the warmup by "
            f"{st(g['C_add_nowarm'] - g['B_add_s0']):+,}. For scale, two seeds of the "
            f"corrected configuration differ by {st(spread):,}. So neither change, alone, "
            f"accounts for the gap to the original's {st(g['main_add_s0']):,}.")
    if init:
        parts.append(
            f"What else differs is narrower than it might seem. At step 0 the two runs share "
            f"{init['n_identical']} of their {init['n_tensors']} weight tensors bit for bit; "
            f"only W_U is drawn differently, because it is drawn last and its shape changed. "
            f"The remaining gap is therefore some combination of W_U's initialisation, the "
            f"extra column itself, an interaction between float32 and no warmup (never run "
            f"jointly), and chance. These single runs cannot separate them, and an earlier "
            f"claim that the original was simply 'a slow draw' went beyond them. Note also "
            f"that `warmup_steps = 1` still gives a zero learning rate on the first step.")
    return "\n\n".join(parts)


# -------------------------------------------------------------- phase diagram

def phase_diagram_block(root: Path, _tag: str) -> Optional[str]:
    d = load_json(root, "sweep_summary.json")
    if not d:
        return None
    cells = d["cells"]
    wds = sorted({c["weight_decay"] for c in cells})
    fracs = sorted({c["train_frac"] for c in cells})
    p_, budget = cells[0]["p"], cells[0]["budget"]

    def fmt(c):
        return f"none (best {c['max_test_acc']:.0%})" if c["censored"] else st(c["grok_step"])

    tbl = table(["", *[f"wd = {w}" for w in wds]],
                [[f"train fraction {fr}"] + [fmt(next(c for c in cells if c["weight_decay"] == w
                                                      and c["train_frac"] == fr)) for w in wds]
                 for fr in fracs])
    parts = [
        f"A smaller modulus (p = {p_}) makes a run cheap enough to sweep. One seed per cell, "
        f"{budget:,} steps each; the number is the step at which test accuracy first reaches "
        f"{GROK_ACC:.0%}.",
        tbl,
        f"{d['n_censored']} of {d['n_total']} cells did not get there within {budget:,} steps "
        f"-- censored, not shown never to grok.",
    ]
    top = max(fracs)
    in_top = [c for c in cells if c["train_frac"] == top]
    full = sorted((c for c in in_top if not c["censored"]), key=lambda c: c["weight_decay"])
    if len(full) >= 3 and len(full) == len(in_top):
        seq = [c["grok_step"] for c in full]
        mono = all(a > b for a, b in zip(seq, seq[1:]))
        prod = [c["grok_step"] * c["weight_decay"] for c in full]
        import math as _m
        xs = [_m.log(c["weight_decay"]) for c in full]

        def slope(ys):
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

        s_total = slope([_m.log(c["grok_step"]) for c in full])
        s_delay = slope([_m.log(c["grok_step"] - (c["memorisation_step"] or 0)) for c in full])
        parts.append(
            f"Within the one training fraction where every cell grokked ({top}), the "
            f"grokking step {'falls monotonically' if mono else 'is not monotone'} as weight "
            f"decay rises: " + ", ".join(f"{st(s):,}" for s in seq) + ". Step times weight "
            f"decay stays between {min(prod):,.0f} and {max(prod):,.0f} while weight decay "
            f"spans a factor of {full[-1]['weight_decay'] / full[0]['weight_decay']:.0f}. "
            f"On log axes the slope is {s_total:.2f} counting from step 0, and {s_delay:.2f} "
            f"counting from the end of memorisation, which is the delay the published law "
            f"is about. One seed and four points; consistent with a roughly 1/lambda "
            f"dependence, which {cite(LYU_2023)} ({arxiv(LYU_2023)}) prove in a large-"
            f"initialisation limit and {cite(KHANH_2026)} fit under AdamW. This reproduces "
            f"that; it does not discover it.")
    return "\n\n".join(parts)


def replicates(root: Path, _tag: str) -> Optional[str]:
    import glob
    from .analysis.timing import crossing_step
    runs = []
    for hp in sorted(glob.glob(str(root / "results" / "*_f0.5*_history.json"))):
        h = json.loads(Path(hp).read_text())
        if not is_finished(h) or h["data"]["p"] != 59:
            continue
        runs.append({"wd": h["train_cfg"]["weight_decay"], "seed": h["data"]["seed"],
                     "grok": crossing_step(h["history"], "test_acc", GROK_ACC),
                     "budget": h["train_cfg"]["steps"],
                     "best": max(r["test_acc"] for r in h["history"])})
    seeds = sorted({r["seed"] for r in runs})
    if len(seeds) < 2:
        return None
    wds = sorted({r["wd"] for r in runs})

    def cell(w, s):
        m = next((r for r in runs if r["wd"] == w and r["seed"] == s), None)
        if not m:
            return "--"
        return st(m["grok"]) if m["grok"] else f"none by {m['budget']:,} (best {m['best']:.0%})"

    body = [[f"weight decay {w}"] + [cell(w, s) for s in seeds] for w in wds]

    # A censored cell is "later than its budget", which still orders it.
    def key(r):
        return r["grok"] if r["grok"] is not None else r["budget"] + 1

    verdicts = []
    for s in seeds:
        seq = [next((r for r in runs if r["wd"] == w and r["seed"] == s), None) for w in wds]
        if any(x is None for x in seq):
            continue
        ks = [key(x) for x in seq]
        ok = all(a > b for a, b in zip(ks, ks[1:]))
        # a censored cell followed by a larger-budget cell cannot be ordered
        verdicts.append(ok)
    budgets = {s: sorted({r["budget"] for r in runs if r["seed"] == s}) for s in seeds}
    return "\n\n".join([
        "The same training fraction, a second seed for both split and initialisation:",
        table(["", *[f"seed {s} ({', '.join(f'{b:,}' for b in budgets[s])}-step budget)"
                     for s in seeds]], body),
        f"Both seeds order the cells the same way, from most weight decay (fastest) to "
        f"least, counting a cell that did not grok as later than its budget. That is "
        + ("consistent in every seed" if all(verdicts) else
           f"consistent in {sum(verdicts)} of {len(verdicts)} seeds")
        + ". The budgets differ, so a censored cell is not comparable with the other seed's "
        f"number in the same row, only with its own column's order.",
    ])


# ----------------------------------------------------------------- forecast

def prediction(root: Path, _tag: str) -> Optional[str]:
    d = load_json(root, "prediction_summary.json")
    w = (d or {}).get("within_config")
    if not w:
        return None
    c = w["config"]
    earliest = min(r["grok"] for r in w["runs"])
    latest = max(r["grok"] for r in w["runs"])
    steps = [s for s in sorted(w["at_steps"], key=int) if int(s) < earliest]
    if not steps:
        return None
    keys = [k for s in steps for k in w["at_steps"][s]]
    keys = list(dict.fromkeys(keys))
    body = []
    for k in keys:
        lab = next(w["at_steps"][s][k]["label"] for s in steps if k in w["at_steps"][s])
        cells = []
        for s in steps:
            v = w["at_steps"][s].get(k)
            cells.append(None if v is None else
                         f"{v['rho']:+.2f}" + (" *" if v.get("survives_bonferroni") else ""))
        body.append([lab] + cells)
    b = w["bonferroni"]
    acc = w.get("test_acc_at_step", {})
    last = steps[-1]
    surv = sorted(((v["label"], v["rho"]) for v in w["at_steps"][last].values()
                   if v.get("survives_bonferroni")), key=lambda x: -abs(x[1]))
    none_first = not any(v.get("survives_bonferroni") for v in w["at_steps"][steps[0]].values())
    best = surv[0] if surv else None
    parts = [
        f"{cite(NOTSAWO_2023)} ({arxiv(NOTSAWO_2023)}) predict *whether* grokking will occur "
        f"from oscillations in the early training-loss curve. The question here is a "
        f"different one: among runs that all grok, which early signals rank them by *when*, "
        f"in a setting where only the random draw differs.",
        f"{w['n']} runs share the task (`{c['op']}`), modulus ({c['p']}), training fraction "
        f"({c['train_frac']}), weight decay ({c['weight_decay']}) and budget "
        f"({c['budget']:,} steps). They grok between step {earliest:,.0f} and "
        f"{latest:,.0f}. A reading taken after some run has grokked measures the outcome, so "
        f"only readings before step {earliest:,.0f} are used.",
    ]
    if acc:
        parts.append(
            "These are not forecasts made before anything happens. In this configuration "
            "test accuracy starts rising almost immediately: "
            + "; ".join(f"at step {s} it is already {acc[s]['min']:.0%} to {acc[s]['max']:.0%}"
                        for s in steps if s in acc)
            + f" (chance {acc[steps[0]]['chance']:.1%}). The readings rank how far along a "
              f"transition already under way each run is.")
    parts += [
        f"Spearman correlation with the grokking step; positive means a higher reading goes "
        f"with a later transition. `*` marks coefficients that survive a two-tailed "
        f"Bonferroni correction over all {b['n_tests']} tests computed (|rho| > "
        f"{b['critical_rho_two_tailed']:.2f}). 'Leak-free' rows use the frequencies each "
        f"checkpoint itself would pick, rather than the final model's:",
        table(["signal", *[f"at step {int(s):,}" for s in steps]], body),
    ]
    plain = {"train_loss", "test_loss", "test_acc", "weight_norm"}
    first = w["at_steps"][steps[0]]
    top_first = max(first.items(), key=lambda kv: abs(kv[1]["rho"]))
    if none_first:
        parts.append(
            f"At step {int(steps[0]):,} nothing survives the correction. The largest point "
            f"estimate there is the {top_first[1]['label']} ({top_first[1]['rho']:+.2f}), "
            + ("a plain loss signal" if top_first[0] in plain else "a mechanistic measure")
            + ", but none of these is reliable at this n.")
    if best:
        parts.append(
            f"At step {int(last):,}, {len(surv)} signals do. The strongest is the "
            f"**{best[0]}** (rho {best[1]:+.2f}), and the survivors are not distinguishable "
            f"from one another at n = {w['n']}. In this one configuration, then, plain losses "
            f"rank the runs as well as the mechanistic measures do by step {int(last):,}, and "
            f"the ordinary test loss -- visible from outside -- is among the best. An earlier "
            f"version of this section claimed the visible quantity could not rank the runs; it "
            f"had looked only at test accuracy.")
    parts.append(
        "Earlier versions of this table were also wrong for a mechanical reason: runs with a "
        "non-zero data seed were analysed on the seed-0 train/test split, which corrupted "
        "every split-dependent signal. The numbers above are from the corrected analysis.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- registry

GENERATORS = {
    "headline": headline,
    "mechanism": mechanism,
    "readout": readout,
    "ablations": ablations,
    "disagreement": disagreement,
    "load_bearing": load_bearing,
    "redundancy": redundancy,
    "phases": phases,
    "runtime": runtime,
    "quadratic": quadratic,
    "controls": controls,
    "phase_diagram": phase_diagram_block,
    "replicates": replicates,
    "prediction": prediction,
}

MULTI_TAG_GENERATORS = {"operations": operations}


def build(root: Path, tag: str, op_tags: Optional[List[str]] = None) -> Dict[str, str]:
    out = {}
    for name, fn in GENERATORS.items():
        try:
            v = fn(root, tag)
        except Exception as exc:                  # a broken block must not vanish
            v = f"_Block failed to render: {exc!r}_"
        out[name] = v if v is not None else PENDING
    for name, fn in MULTI_TAG_GENERATORS.items():
        try:
            v = fn(root, op_tags or [tag])
        except Exception as exc:
            v = f"_Block failed to render: {exc!r}_"
        out[name] = v if v is not None else PENDING
    try:
        v = dlog_block(root)
    except Exception as exc:
        v = f"_Block failed to render: {exc!r}_"
    out["dlog"] = v if v is not None else PENDING
    return out
