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

Only the second is used in the write-up.  The classification across
configurations is computed and saved (``at_steps``) but not reported: which runs
fail to grok is decided mostly by the training fraction and the operation, so a
signal that merely tracks those scores well.  It is kept so that the tests run
on these data are all visible.
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
from grokking.runinfo import provenance                # noqa: E402

PLAIN = {"train_loss", "test_loss", "test_acc", "weight_norm"}   # no mechanism needed
BOOT_DRAWS = 10_000

FEATURES = [
    # Everything a forecast may use must be computable from TRAINING pairs, or
    # be declared visible from outside.  Restricted loss is taken on the
    # training split; the all-pairs version reads held-out labels.
    ("restricted_loss_sum_train", "restricted loss (train pairs) (final)", False),
    ("excluded_loss_sum", "excluded loss (train pairs) (final)", True),
    ("emb_gini", "embedding Gini (no frequency choice)", True),
    ("emb_key_frac", "power in key frequencies (final)", True),
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


def bootstrap_rhos(xs_by_key, ys, draws=BOOT_DRAWS, seed=0):
    """Spearman rho of every signal on the same bootstrap resamples of the runs.

    Resampling runs (not signals) keeps the pairing, so differences between two
    signals' coefficients get an honest interval.  Returns {key: array of rho}.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(ys)
    idx = rng.integers(0, n, size=(draws, n))
    out = {k: np.full(draws, np.nan) for k in xs_by_key}
    ys = list(ys)
    for d in range(draws):
        ii = idx[d]
        yy = [ys[i] for i in ii]
        for k, xs in xs_by_key.items():
            r = spearman([xs[i] for i in ii], yy)
            if r is not None:
                out[k][d] = r
    return out


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
    censored_members = [m["tag"] for m in members if m["censored"]]
    members = [m for m in members if not m["censored"]]
    # runs of the same configuration left out only because their budget differs
    other_budget = [{"tag": r["tag"], "budget": r["budget"], "grok": r["grok"]}
                    for r in runs if (r["op"], r["p"], r["frac"], r["wd"]) == cfg[:4]
                    and r["budget"] != cfg[4]]
    if len(members) >= 4:
        members.sort(key=lambda m: m["grok"])
        within = {"config": {"op": cfg[0], "p": cfg[1], "train_frac": cfg[2],
                             "weight_decay": cfg[3], "budget": cfg[4]},
                  "n": len(members),
                  "runs": [{"tag": m["tag"], "grok": m["grok"]} for m in members],
                  # A censored member would be dropped by its outcome; say so.
                  "censored_members_dropped": censored_members,
                  "excluded_other_budget": other_budget,
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
                    rec[key] = {"label": label, "rho": rho, "n": len(xs)}
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
        #
        # That family was chosen after the results were seen.  The first version
        # corrected over every test computed, at every step; both are recorded
        # so the write-up can say what the narrowing changed.
        earliest = min(m["grok"] for m in members)
        reported = [s for s in within["at_steps"] if int(s) < earliest]
        nulls = {}

        def null_for(n):
            if n not in nulls:
                nulls[n] = permutation_null(n)
            return nulls[n]

        n_tests = sum(len(within["at_steps"][s]) for s in reported)
        n_all = sum(len(r) for r in within["at_steps"].values())
        alpha, alpha_all = 0.05 / max(n_tests, 1), 0.05 / max(n_all, 1)
        for at, rec in within["at_steps"].items():
            for v in rec.values():
                v["p_two_tailed"] = perm_p_two_tailed(v["rho"], null_for(v["n"]))
                v["survives_bonferroni"] = (at in reported and v["p_two_tailed"] < alpha)
                v["survives_bonferroni_all_computed"] = v["p_two_tailed"] < alpha_all
        null = null_for(len(members))
        within["bonferroni"] = {
            "n_tests": n_tests, "alpha": alpha, "reported_steps": reported,
            "method": f"two-tailed permutation test, {len(null):,} draws",
            "critical_rho_two_tailed": perm_critical(null, alpha),
            "all_computed": {"n_tests": n_all, "alpha": alpha_all,
                             "critical_rho_two_tailed": perm_critical(null, alpha_all)}}

        # How precisely are the coefficients known?  Bootstrap the runs and
        # compare the best plain signal with the best mechanistic one.
        import numpy as np
        within["bootstrap"] = {}
        for at in reported:
            rec = within["at_steps"][at]
            xs_by_key = {}
            for key in rec:
                xs = [min(m["rows"], key=lambda x: abs(x["step"] - int(at)))[key] for m in members]
                xs_by_key[key] = xs
            boot = bootstrap_rhos(xs_by_key, [m["grok"] for m in members])
            ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                  for k, v in boot.items()}
            plain = [k for k in rec if k in PLAIN]
            mech = [k for k in rec if k not in PLAIN and "(final)" not in rec[k]["label"]]
            bp = max(plain, key=lambda k: abs(rec[k]["rho"]))
            bm = max(mech, key=lambda k: abs(rec[k]["rho"]))
            diff = np.abs(boot[bp]) - np.abs(boot[bm])
            within["bootstrap"][at] = {
                "draws": BOOT_DRAWS, "rho_ci95": ci,
                "best_plain": bp, "best_mechanistic_leak_free": bm,
                "abs_rho_difference": float(abs(rec[bp]["rho"]) - abs(rec[bm]["rho"])),
                "abs_rho_difference_ci95": [float(np.nanpercentile(diff, 2.5)),
                                            float(np.nanpercentile(diff, 97.5))]}
        print(f"  Bonferroni over {n_tests} reported tests (steps {reported}), alpha "
              f"{alpha:.5f}; permutation critical |rho| = "
              f"{within['bonferroni']['critical_rho_two_tailed']:.3f}")
        out["within_config"] = within
        print()

    out["_provenance"] = provenance(root)
    path = root / "results" / "prediction_summary.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    print("\nAUC is the probability that a run which will grok scores above one that will "
          "not; 0.5 is chance.\nrho is the rank correlation between the signal and the "
          "grokking step among runs that grokked;\nnegative means a higher signal predicts "
          "an earlier grok.")


if __name__ == "__main__":
    main()
