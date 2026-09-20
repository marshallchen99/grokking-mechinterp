#!/usr/bin/env python3
"""Export a compact JSON for the interactive page.

Everything the page shows has to come from the results files, same as the
README, so this is the only place the web data is assembled.  It is kept small
by thinning the training history (which has thousands of points at a cadence no
chart needs) while keeping every checkpoint's spectrum.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.analysis.timing import crossing_step, phase_boundaries   # noqa: E402
from grokking.report import is_finished                                # noqa: E402


def load(root, name):
    p = root / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def thin(hist, target=420):
    """Keep every point up to step 200, then log-thin the rest."""
    if len(hist) <= target:
        return hist
    import math
    out, seen = [], set()
    for r in hist:
        if r["step"] <= 200:
            out.append(r); seen.add(r["step"])
    rest = [r for r in hist if r["step"] > 200]
    n = target - len(out)
    lo, hi = math.log(rest[0]["step"]), math.log(rest[-1]["step"])
    for i in range(n):
        want = math.exp(lo + (hi - lo) * i / max(n - 1, 1))
        r = min(rest, key=lambda x: abs(x["step"] - want))
        if r["step"] not in seen:
            out.append(r); seen.add(r["step"])
    return sorted(out, key=lambda r: r["step"])


def r4(x):
    return None if x is None else round(float(x), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main_add_s0")
    ap.add_argument("--op-tags", nargs="*",
                    default=["main_add_s0", "B_add_s0", "B_sub_s0", "B_mul_s0",
                             "B_sqx_p113", "B_sqx_p109", "B_add_s1"])
    ap.add_argument("--out", default="web_data.json")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    hist = load(root, f"{args.tag}_history.json")
    analysis = load(root, f"{args.tag}_analysis.json")
    mech = load(root, f"{args.tag}_mechanism.json")
    red = load(root, f"{args.tag}_redundancy.json")
    sweep = load(root, "sweep_summary.json")
    if not (hist and analysis and analysis.get("complete")):
        raise SystemExit("need a finished run with a complete trajectory analysis")

    pb = phase_boundaries(hist["history"])
    curve = [{"s": r["step"], "tr": r4(r["train_acc"]), "te": r4(r["test_acc"]),
              "trl": r4(r["train_loss"]), "tel": r4(r["test_loss"]),
              "w": r4(r["weight_norm"])}
             for r in thin(hist["history"])]

    frames = [{"s": r["step"], "te": r4(r["test_acc"]), "tr": r4(r["train_acc"]),
               "pw": [round(v, 6) for v in r["emb_power_per_freq"]],
               "gini": r4(r.get("emb_gini")),
               "restr": r4(r.get("restricted_loss_sum_all")),
               "excl": r4(r.get("excluded_loss_sum")),
               "ab": r4(r.get("logit_var_a+b")),
               "nf": r4(r.get("neuron_frac_above_85pct"))}
              for r in analysis["rows"]]

    out = {
        "meta": {
            "p": hist["data"]["p"], "op": hist["data"]["op"],
            "n_train": hist["data"]["n_train"], "n_test": hist["data"]["n_test"],
            "train_frac": hist["data"]["train_frac"],
            "n_params": hist["n_params"], "steps": hist["train_cfg"]["steps"],
            "weight_decay": hist["train_cfg"]["weight_decay"],
            "chance": 1.0 / hist["data"]["p"],
            "wall_clock_s": round(hist["history"][-1]["elapsed"]),
        },
        "phases": {k: (round(v) if v is not None else None) for k, v in pb.items()},
        "curve": curve,
        "frames": frames,
        "key_freqs": analysis["key_freqs"],
        "key_freq_rules": analysis.get("key_freq_rules"),
    }
    if mech:
        out["mechanism"] = {
            "final_step": mech["final_step"],
            "gini": r4(mech["key_freqs"]["gini_W_E"]),
            "frac_power": r4(mech["key_freqs"]["frac_power_in_key"]),
            "structure": {k: r4(v) for k, v in mech["structure"].items()},
            "trig": {"sum": r4(mech["trig"]["mean_sum_frac"]),
                     "readout": r4(mech["trig"]["mean_readout_frac"])},
            "neurons": mech["neurons"],
            "ablation": {k: {kk: r4(vv) for kk, vv in v.items()}
                         for k, v in mech["ablation_frequency"].items()
                         if not k.startswith("_")},
            "control_freqs": mech["ablation_frequency"].get("_control_freqs", {}).get("freqs", []),
            "components": {k: {kk: r4(vv) for kk, vv in v.items()}
                           for k, v in mech["ablation_component"].items()},
            "progress": {k: {kk: r4(vv) for kk, vv in v.items()}
                         for k, v in mech["progress"].items()},
        }
    if red:
        out["redundancy"] = {"rows": [{"subset": r["subset"], "acc": r4(r["test_acc"]),
                                       "loss": r4(r["test_loss"])} for r in red["rows"]],
                             "minimal": red["minimal_subsets"],
                             "redundant": red["redundant_frequencies"]}
    mul = load(root, "B_mul_s0_mechanism.json")
    if mul and "dlog" in mul:
        d = mul["dlog"]
        out["dlog"] = {
            "n": d["n"], "g": d["primitive_root"],
            "ordinary": {"nkeys": len(mul["key_freqs"]["used"]),
                         "gini": r4(mul["key_freqs"]["gini_W_E"]),
                         "power": r4(mul["key_freqs"]["frac_power_in_key"]),
                         "sumvar": r4(mul["structure"]["a+b"]),
                         "jaccard": r4(mul["key_freqs"]["jaccard"])},
            "star": {"nkeys": len(d["key_freqs"]["used"]),
                     "keys": d["key_freqs"]["used"],
                     "gini": r4(d["key_freqs"]["gini_W_E"]),
                     "power": r4(d["key_freqs"]["frac_power_in_key"]),
                     "sumvar": r4(d["structure"]["a+b"]),
                     "jaccard": r4(d["key_freqs"]["jaccard"])},
            "causal": {k: {kk: r4(vv) for kk, vv in v.items()}
                       for k, v in d.get("progress", {}).items()},
            "zero": {k: r4(v) for k, v in d["zero_element"].items()},
        }

    controls = []
    for tag in ["main_add_s0", "B_add_s0", "C_add_f32", "C_add_nowarm", "B_add_s1"]:
        h = load(root, f"{tag}_history.json")
        if not is_finished(h):
            continue
        controls.append({"tag": tag, "grok": r4(crossing_step(h["history"], "test_acc", 0.90)),
                         "budget": h["train_cfg"]["steps"]})
    out["controls"] = controls

    pred = load(root, "prediction_summary.json")
    if pred and pred.get("within_config"):
        out["prediction"] = pred["within_config"]

    import glob as _glob
    repl = []
    for hp in sorted(_glob.glob(str(root / "results" / "*_f0.5*_history.json"))):
        h = json.loads(Path(hp).read_text())
        if not is_finished(h) or h["data"]["p"] != 59:
            continue
        repl.append({"wd": h["train_cfg"]["weight_decay"], "seed": h["data"]["seed"],
                     "grok": r4(crossing_step(h["history"], "test_acc", 0.90)),
                     "budget": h["train_cfg"]["steps"]})
    out["replicates"] = repl

    ops = []
    for tag in args.op_tags:
        h = load(root, f"{tag}_history.json")
        if not is_finished(h):
            continue
        ops.append({"tag": tag, "op": h["data"]["op"], "p": h["data"]["p"],
                    "grok": r4(crossing_step(h["history"], "test_acc", 0.90)),
                    "budget": h["train_cfg"]["steps"],
                    "final_test": r4(h["history"][-1]["test_acc"]),
                    "mech": (lambda m: None if not m else {
                        "nkeys": len(m["key_freqs"]["used"]),
                        "gini": r4(m["key_freqs"]["gini_W_E"]),
                        "sumvar": r4(m["structure"]["a+b"]),
                        "diffvar": r4(m["structure"]["a-b"]),
                        "trig": r4(m["trig"]["mean_sum_frac"])})(
                            load(root, f"{tag}_mechanism.json")),
                    "curve": [{"s": r["step"], "te": r4(r["test_acc"]),
                               "tr": r4(r["train_acc"])} for r in thin(h["history"], 220)]})
    out["operations"] = ops
    if sweep:
        out["sweep"] = sweep["cells"]

    path = root / "results" / args.out
    path.write_text(json.dumps(out, separators=(",", ":")))
    kb = path.stat().st_size / 1024
    print(f"wrote {path}  ({kb:.0f} KB; {len(curve)} curve points, "
          f"{len(frames)} spectra, {len(ops)} finished operation runs)")


if __name__ == "__main__":
    main()
