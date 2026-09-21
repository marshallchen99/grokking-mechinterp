#!/usr/bin/env python3
"""Can an early internal signal forecast when -- or whether -- a run will grok?

This is the question a progress measure is ultimately for.  Section 4 shows the
measures move before the accuracy does *within one run*; that is a much weaker
claim than being able to look at step 1,000 of an unseen run and say what will
happen at step 10,000.

Two questions, kept separate because they have different sample sizes:

  classification  at an early step, does the signal separate runs that will
                  grok inside their budget from runs that will not?
  regression      among the runs that do grok, does the signal rank them by how
                  long they take?

Both are reported with the number of runs behind them, because that number is
small and the reader has to be able to discount accordingly.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.timing import crossing_step     # noqa: E402
from grokking.report import is_finished                # noqa: E402

FEATURES = [
    # Everything a forecast may use must be computable from TRAINING pairs, or
    # be declared visible from outside.  Restricted loss is taken on the
    # training split; the all-pairs version reads held-out labels.
    ("restricted_loss_sum_train", "restricted loss (train pairs)", False),
    ("excluded_loss_sum", "excluded loss (train pairs)", True),
    ("emb_gini", "embedding Gini (no frequency choice)", True),
    ("emb_key_frac", "power in key frequencies", True),
    ("logit_var_a+b", "(a+b) variance explained (no frequency choice)", True),
    ("weight_norm", "weight norm", False),
    ("train_loss", "train loss", False),
    ("test_acc", "test accuracy (visible from outside)", True),
    ("test_loss", "test loss (visible from outside)", False),
    # the frequency-dependent measures again, with the set each checkpoint
    # itself would pick -- no information from after the transition
    ("restricted_loss_sum_train_live", "restricted loss (train pairs), leak-free", False),
    ("excluded_loss_sum_live", "excluded loss (train pairs), leak-free", True),
    ("emb_key_frac_live", "power in key frequencies, leak-free", True),
]


def spearman(x, y):
    """Rank correlation, written out so the repo needs no scipy."""
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def spearman_p_two_tailed(rho, n):
    """Two-tailed p for a Spearman coefficient, via the t approximation.

    Two-tailed because the table reports signals in both directions; a one-
    tailed test would need the direction fixed in advance for every signal.
    """
    import math
    if rho is None or n < 4:
        return None
    if abs(rho) >= 1:
        return 0.0
    t = abs(rho) * math.sqrt((n - 2) / (1 - rho * rho))
    df = n - 2
    # survival function of Student's t, by numerical integration of the pdf
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    hi, N = t + 60.0, 60000
    h = (hi - t) / N
    area = sum(c * (1 + (t + i * h) ** 2 / df) ** (-(df + 1) / 2) * (0.5 if i in (0, N) else 1)
               for i in range(N + 1)) * h
    return min(1.0, 2 * area)


def critical_rho(n, alpha):
    """Smallest |rho| that is significant at two-tailed level alpha."""
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = (lo + hi) / 2
        pv = spearman_p_two_tailed(mid, n)
        if pv is not None and pv < alpha:
            hi = mid
        else:
            lo = mid
    return hi


def permutation_null(n, draws=200_000, seed=0):
    """|rho| under the null of no association, by Monte Carlo over permutations.

    For untied ranks the null distribution of Spearman's rho depends only on n,
    so it is computed once and reused for every signal.  This replaces a t
    approximation that is anti-conservative at n around 16.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n + 1, dtype=float)
    perms = np.argsort(rng.random((draws, n)), axis=1) + 1.0
    d2 = ((perms - ranks) ** 2).sum(axis=1)
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return np.sort(np.abs(rho))


def perm_p_two_tailed(rho, null):
    import numpy as np
    if rho is None:
        return None
    k = len(null) - np.searchsorted(null, abs(rho) - 1e-12, side="left")
    return float((k + 1) / (len(null) + 1))


def perm_critical(null, alpha):
    import numpy as np
    return float(np.quantile(null, 1.0 - alpha))


def auc(pos, neg):
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    if not pos or not neg:
        return None
    wins = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at-steps", type=int, nargs="+", default=[200, 500, 1000, 2000])
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    runs = []
    for hp in sorted(glob.glob(str(root / "results" / f"{args.pattern}_history.json"))):
        tag = Path(hp).name[: -len("_history.json")]
        h = json.loads(Path(hp).read_text())
        if not is_finished(h):
            continue
        ap_ = root / "results" / f"{tag}_analysis.json"
        if not ap_.exists():
            continue
        a = json.loads(ap_.read_text())
        if not a.get("complete"):
            continue
        grok = crossing_step(h["history"], "test_acc", 0.90)
        runs.append({"tag": tag, "p": h["data"]["p"], "op": h["data"]["op"],
                     "wd": h["train_cfg"]["weight_decay"],
                     "frac": h["data"]["train_frac"],
                     "budget": h["train_cfg"]["steps"],
                     "grok": grok, "censored": grok is None,
                     "rows": a["rows"]})
    if len(runs) < 4:
        raise SystemExit(f"only {len(runs)} runs have a complete trajectory analysis")

    print(f"{len(runs)} runs with complete trajectories "
          f"({sum(r['censored'] for r in runs)} censored)\n")

    out = {"n_runs": len(runs), "n_censored": sum(r["censored"] for r in runs),
           "runs": [{k: r[k] for k in ("tag", "p", "wd", "frac", "grok", "censored", "budget")}
                    for r in runs],
           "at_steps": {}}

    for at in args.at_steps:
        usable = [r for r in runs if r["rows"][-1]["step"] >= at]
        if len(usable) < 4:
            continue
        rec = {"n": len(usable), "features": {}}
        print(f"=== measured at step {at:,}  ({len(usable)} runs) ===")
        print(f"  {'feature':34s} {'AUC grok/censored':>18s} {'rho vs grok step':>18s}")
        for key, label, higher_is_earlier in FEATURES:
            vals, groks = [], []
            for r in usable:
                row = min(r["rows"], key=lambda x: abs(x["step"] - at))
                v = row.get(key)
                if v is None:
                    continue
                vals.append((r, v))
            if len(vals) < 4:
                continue
            pos = [v for r, v in vals if not r["censored"]]
            neg = [v for r, v in vals if r["censored"]]
            a_ = auc(pos, neg) if higher_is_earlier else (
                auc(neg, pos) if pos and neg else None)
            g = [(v, r["grok"]) for r, v in vals if not r["censored"]]
            rho = spearman([x for x, _ in g], [y for _, y in g]) if len(g) >= 3 else None
            rec["features"][key] = {"label": label, "auc": a_, "rho": rho,
                                    "n_grok": len(pos), "n_censored": len(neg)}
            print(f"  {label:34s} {('%.3f' % a_) if a_ is not None else '--':>18s} "
                  f"{('%+.3f' % rho) if rho is not None else '--':>18s}")
        out["at_steps"][str(at)] = rec
        print()

    # ---- the unconfounded version --------------------------------------
    # The censored runs are mostly low-train-fraction sweep cells, so any signal
    # that merely tracks the training fraction scores a high AUC above.  The
    # clean question is asked inside a single configuration, where the only thing
    # that differs is the random draw.
    # Group by the OPERATION too: two runs on different tasks do not differ only
    # in their random draw, and subtraction's grokking step is driven by the task.
    groups = {}
    for r in runs:
        # the budget is part of the configuration: a run with a different budget
        # has a different checkpoint schedule and its "final" key frequencies
        # come from a different step
        groups.setdefault((r["op"], r["p"], r["frac"], r["wd"], r["budget"]), []).append(r)
    best = max(groups.items(), key=lambda kv: sum(not x["censored"] for x in kv[1]))
    cfg, members = best
    members = [m for m in members if not m["censored"]]
    if len(members) >= 4:
        members.sort(key=lambda m: m["grok"])
        within = {"config": {"op": cfg[0], "p": cfg[1], "train_frac": cfg[2],
                             "weight_decay": cfg[3], "budget": cfg[4]},
                  "n": len(members),
                  "runs": [{"tag": m["tag"], "grok": m["grok"]} for m in members],
                  "at_steps": {}}
        print(f"=== within one configuration: {cfg[0]}, p={cfg[1]}, frac={cfg[2]}, wd={cfg[3]}, budget={cfg[4]} "
              f"({len(members)} runs, only the random draw differs) ===")
        for at in args.at_steps:
            rec = {}
            for key, label, _ in FEATURES:
                xs, ys = [], []
                for m in members:
                    row = min(m["rows"], key=lambda x: abs(x["step"] - at))
                    if row.get(key) is None:
                        continue
                    xs.append(row[key]); ys.append(m["grok"])
                rho = spearman(xs, ys) if len(xs) >= 3 else None
                if rho is not None:
                    rec[key] = {"label": label, "rho": rho,
                                "p_two_tailed": spearman_p_two_tailed(rho, len(xs))}
            within["at_steps"][str(at)] = rec
            accs = [min(m["rows"], key=lambda x: abs(x["step"] - at))["test_acc"] for m in members]
            within.setdefault("test_acc_at_step", {})[str(at)] = {
                "min": min(accs), "max": max(accs), "chance": 1.0 / cfg[1]}
        hdr = "  %-28s" % "signal" + "".join("%12s" % f"step {a:,}" for a in args.at_steps)
        print(hdr)
        for key, label, _ in FEATURES:
            line = "  %-28s" % label
            for at in args.at_steps:
                v = within["at_steps"][str(at)].get(key)
                line += "%12s" % (f"{v['rho']:+.2f}" if v else "--")
            print(line)
        # Significance by permutation, and Bonferroni over the tests that are
        # actually reported: readings before the earliest transition only.  A
        # reading after some run has grokked measures the outcome and is not
        # shown, so it must not inflate the denominator either.
        earliest = min(m["grok"] for m in members)
        reported = [s for s in within["at_steps"] if int(s) < earliest]
        null = permutation_null(len(members))
        n_tests = sum(len(within["at_steps"][s]) for s in reported)
        alpha = 0.05 / max(n_tests, 1)
        for at, rec in within["at_steps"].items():
            for v in rec.values():
                v["p_two_tailed"] = perm_p_two_tailed(v["rho"], null)
                v["survives_bonferroni"] = (at in reported and v["p_two_tailed"] is not None
                                            and v["p_two_tailed"] < alpha)
        within["bonferroni"] = {
            "n_tests": n_tests, "alpha": alpha, "reported_steps": reported,
            "method": f"two-tailed permutation test, {len(null):,} draws",
            "critical_rho_two_tailed": perm_critical(null, alpha)}
        print(f"  Bonferroni over {n_tests} reported tests (steps {reported}), alpha "
              f"{alpha:.5f}; permutation critical |rho| = "
              f"{within['bonferroni']['critical_rho_two_tailed']:.3f}")
        out["within_config"] = within
        print()

    path = root / "results" / "prediction_summary.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    print("\nAUC is the probability that a run which will grok scores above one that will "
          "not; 0.5 is chance.\nrho is the rank correlation between the signal and the "
          "grokking step among runs that grokked;\nnegative means a higher signal predicts "
          "an earlier grok.")


if __name__ == "__main__":
    main()
