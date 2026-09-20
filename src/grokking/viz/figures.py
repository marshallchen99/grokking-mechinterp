"""The figures.

Design notes that apply throughout:

* Identity colours are fixed once and never reassigned: train is always
  categorical slot 1, test always slot 2, in every figure in the project.  A
  reader learns the mapping in figure 1 and it holds to the end.
* Where one line is the point and the others are context, the figure switches
  to emphasis -- accent hue for the subject, de-emphasis grey for the rest --
  rather than spending eight categorical hues on a story about one of them.
* The x axis is logarithmic almost everywhere, because the phenomenon spans
  three orders of magnitude in training step and a linear axis would compress
  everything interesting into the left edge.
* Magnitude grids use one hue light-to-dark; signed quantities use the blue/red
  pair with a neutral grey midpoint.  No rainbows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, categorical, diverging_cmap, emphasis, sequential_cmap


def _logx(ax, lo: float = 1.0, xmax: Optional[float] = None):
    """Log step axis that still shows step 0.

    symlog is the right scale here -- the phenomenon spans three orders of
    magnitude and step 0 has to be on the plot -- but by default it also draws
    the negative branch, which is meaningless for a step count.
    """
    ax.set_xscale("symlog", linthresh=lo)
    ax.set_xlabel("training step")
    if xmax is not None:
        ax.set_xlim(0, xmax)


def _endpoint_labels(ax, x, items, dx=1.04, fontsize=8, min_gap_frac=0.055):
    """Direct labels at the right edge, nudged apart when they would collide.

    Two series that both end at 100% would otherwise print on top of each
    other, which is the most common way a direct label stops being a label.
    """
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"

    def to_frac(y):
        if log:
            import math
            return (math.log10(max(y, 1e-300)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (y - lo) / (hi - lo)

    def from_frac(f):
        if log:
            return 10 ** (f * (math.log10(hi) - math.log10(lo)) + math.log10(lo))
        return f * (hi - lo) + lo

    import math
    placed = sorted(((to_frac(y), text, color) for y, text, color in items))
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < min_gap_frac:
            placed[i] = (placed[i - 1][0] + min_gap_frac, placed[i][1], placed[i][2])
    for f, text, color in placed:
        ax.annotate(text, xy=(x, from_frac(min(f, 1.0))), color=color,
                    va="center", ha="left", fontsize=fontsize,
                    xytext=(x * dx, from_frac(min(f, 1.0))),
                    annotation_clip=False, fontweight="bold")


def grokking_curve(hist: List[Dict], mode: str = "light",
                   grok_step: Optional[int] = None,
                   title: str = "Grokking on modular addition",
                   figsize=(6.4, 4.6)):
    """The headline figure: train and test, accuracy above, loss below.

    Two series, so: legend present, and both direct-labelled at their right
    end.  The gap between the two lines IS the phenomenon, so nothing else
    competes for attention -- no markers, no annotations except the one
    vertical rule marking where generalisation happens.
    """
    t = PALETTE[mode]
    c_train, c_test = categorical(mode, 2)
    step = np.array([r["step"] for r in hist], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   height_ratios=[1, 1])

    for ax, keys, ylab in (
        (ax1, ("train_acc", "test_acc"), "accuracy"),
        (ax2, ("train_loss", "test_loss"), "cross-entropy loss"),
    ):
        for key, color, name in zip(keys, (c_train, c_test), ("train", "test")):
            y = np.array([r[key] for r in hist], dtype=float)
            ax.plot(step, y, color=color, label=name, linewidth=2.0)
        ax.set_ylabel(ylab)

    ax1.set_ylim(-0.04, 1.06)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_yscale("log")

    if grok_step:
        for ax in (ax1, ax2):
            ax.axvline(grok_step, color=t["deemph"], linewidth=1.0, zorder=0)
        ax1.annotate(f"generalisation\nat step {grok_step:,}",
                     xy=(grok_step, 0.5), xytext=(grok_step * 1.15, 0.42),
                     color=t["text_secondary"], fontsize=8, ha="left")

    # selective direct labels at the right edge, de-overlapped
    for ax, keys in ((ax1, ("train_acc", "test_acc")), (ax2, ("train_loss", "test_loss"))):
        _endpoint_labels(ax, step[-1],
                         [(hist[-1][k], n, c)
                          for k, c, n in zip(keys, (c_train, c_test), ("train", "test"))])

    ax1.set_title(title, color=t["text_primary"])
    # legend to the right of the left-aligned title so the two never collide
    ax1.legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.0),
               frameon=False, borderaxespad=0.2)
    _logx(ax2, xmax=float(step[-1]) * 1.35)
    ax1.set_xlim(0, float(step[-1]) * 1.35)
    return fig


def spectrum_bars(power: Sequence[float], key_freqs: Sequence[int],
                  mode: str = "light", title: str = "",
                  ylabel: str = "share of embedding power",
                  figsize=(6.4, 2.4)):
    """Per-frequency power, with the key frequencies picked out.

    One series whose story is "these few, not those fifty", so this is the
    emphasis form: accent hue for the selected frequencies, de-emphasis grey
    for the rest.  A value ramp here would double-encode bar height as colour.
    """
    t = PALETTE[mode]
    pw = np.asarray(power, dtype=float)[1:]          # drop the constant term
    share = pw / pw.sum() if pw.sum() else pw
    freqs = np.arange(1, len(share) + 1)
    colors = [categorical(mode, 1)[0] if f in set(key_freqs) else t["deemph"]
              for f in freqs]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(freqs, share, color=colors, width=0.72)
    ax.set_xlabel("frequency  k")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.2, len(share) + 0.8)
    ax.grid(axis="x", visible=False)
    for f in key_freqs:
        ax.annotate(str(f), xy=(f, share[f - 1]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=categorical(mode, 1)[0], fontweight="bold")
    if title:
        ax.set_title(title, color=t["text_primary"])
    return fig


def heatmap(grid: np.ndarray, mode: str = "light", signed: bool = False,
            title: str = "", xlabel: str = "b", ylabel: str = "a",
            cbar_label: str = "", figsize=(4.2, 3.6), vmax=None):
    """A (p, p) grid. One hue for magnitude; blue/red with a grey midpoint for signed."""
    t = PALETTE[mode]
    fig, ax = plt.subplots(figsize=figsize)
    if signed:
        m = float(np.abs(grid).max()) if vmax is None else vmax
        im = ax.imshow(grid, cmap=diverging_cmap(mode), vmin=-m, vmax=m,
                       origin="lower", interpolation="nearest")
    else:
        im = ax.imshow(grid, cmap=sequential_cmap(mode), origin="lower",
                       vmax=vmax, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, colors=t["text_muted"])
    if cbar_label:
        cb.set_label(cbar_label, color=t["text_secondary"])
    if title:
        ax.set_title(title, color=t["text_primary"])
    return fig


def progress_panel(rows: List[Dict], mode: str = "light",
                   grok_step: Optional[int] = None,
                   phases: Optional[Dict[str, float]] = None,
                   figsize=(6.6, 6.4)):
    """The argument that the transition is not sudden.

    Top panel is what an observer sees from outside: flat, then a cliff.  The
    panels below are measured inside the same checkpoints and move thousands of
    steps earlier.  Each panel uses emphasis -- in each one a single line is
    the claim -- rather than spending categorical hues on a story about one of
    them.
    """
    t = PALETTE[mode]
    c1, c2, c3 = categorical(mode, 3)
    step = np.array([r["step"] for r in rows], dtype=float)

    def col(key, default=np.nan):
        return np.array([r.get(key, default) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(step, col("train_acc"), color=t["deemph"], label="train", linewidth=1.5)
    ax.plot(step, col("test_acc"), color=c2, label="test", linewidth=2.2)
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title("what an observer sees from outside", color=t["text_primary"])
    ax.legend(loc="upper left", ncols=2, frameon=False)

    ax = axes[1]
    ax.plot(step, col("restricted_loss_sum_all"), color=c1,
            label="restricted loss  (keep only the key frequencies)")
    ax.plot(step, col("excluded_loss_sum"), color=c3,
            label="excluded loss  (delete only the key frequencies)")
    ax.plot(step, col("train_loss"), color=t["deemph"], linewidth=1.2,
            label="train loss")
    ax.set_yscale("log")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("what the progress measures see", color=t["text_primary"])
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    ax = axes[2]
    ax.plot(step, col("logit_var_a+b"), color=c1,
            label="logit variance explained by (a+b)")
    ax.plot(step, col("emb_gini"), color=c3,
            label="embedding spectrum concentration (Gini)")
    ax.plot(step, col("neuron_frac_above_85pct"), color=t["deemph"], linewidth=1.5,
            label="neurons explained >85% by one frequency")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("structure forming inside the model", color=t["text_primary"])
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)

    # Phases are marked by their boundaries, with only the middle one shaded.
    # Shading all three in the same tone just paints the whole plot.
    if phases:
        items = sorted(phases.items(), key=lambda kv: kv[1][0])
        for i, (name, (a0, a1)) in enumerate(items):
            if i == 1:
                for ax in axes:
                    ax.axvspan(a0, a1, color=t["grid"], alpha=0.45, zorder=0, linewidth=0)
            if a0 > 0:
                for ax in axes:
                    ax.axvline(a0, color=t["axis"], linewidth=0.7, zorder=1)
        for name, (a0, a1) in items:
            mid = (max(a0, 0.6) * max(a1, 1)) ** 0.5
            axes[0].annotate(name, xy=(mid, 0.72), xycoords=("data", "axes fraction"),
                             fontsize=7.5, color=t["text_muted"], ha="center")
    if grok_step:
        for ax in axes:
            ax.axvline(grok_step, color=c2, linewidth=1.2, zorder=1, alpha=0.8)
    _logx(axes[-1], xmax=float(step[-1]) * 1.02)
    for ax in axes:
        ax.set_xlim(0, float(step[-1]) * 1.02)
    return fig
