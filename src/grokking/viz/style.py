"""One place where every figure's typography and colour is decided.

Colour policy
-------------
The palette is the validated reference instance from the data-visualisation
guidance, used verbatim -- categorical slots in their fixed published order,
a single-hue blue ramp for magnitude, and a blue/red pair with a neutral grey
midpoint for signed quantities.  Two consequences are worth stating because
they show up in almost every figure here:

* Categorical hues are assigned by *identity* and never cycled or reordered.
  "Train" is always slot 1 and "test" is always slot 2, in every figure, so the
  reader learns the mapping once.
* When the story is "one of these is the point", the right form is emphasis --
  one accent hue against a de-emphasis grey -- not eight categorical colours.
  `emphasis()` exists so that is the easy thing to do.

Forms that put every pair of series side by side (scatter, small multiples)
are capped at the first three slots, which are the ones that clear the
all-pairs colour-vision gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- tokens

PALETTE: Dict[str, Dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#7a7975",
        "grid": "#e6e5e1",
        "axis": "#c9c8c3",
        "deemph": "#b8b7b2",
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "sequential": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                       "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                       "#184f95", "#104281", "#0d366b"],
        "diverging_mid": "#f0efec",
        "diverging_neg": "#0d366b",
        "diverging_pos": "#8f1f1f",
        "status": {"good": "#0ca30c", "warning": "#fab219",
                   "serious": "#ec835a", "critical": "#d03b3b"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#8e8d85",
        "grid": "#2e2e2c",
        "axis": "#464643",
        "deemph": "#5a5a56",
        "categorical": ["#3987e5", "#d95926", "#199e70", "#c98500",
                        "#d55181", "#008300", "#9085e9", "#e66767"],
        "sequential": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
                       "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
                       "#9ec5f4", "#b7d3f6", "#cde2fb"],
        "diverging_mid": "#383835",
        "diverging_neg": "#86b6ef",
        "diverging_pos": "#e66767",
        "status": {"good": "#0ca30c", "warning": "#fab219",
                   "serious": "#ec835a", "critical": "#d03b3b"},
    },
}

# Forms where every pair of series can end up adjacent (scatter, small
# multiples) are limited to these slots -- see the palette's all-pairs gate.
ALL_PAIRS_CAP = 3

_WINDOWS_FONTS = Path("/mnt/c/Windows/Fonts")
_PREFERRED_SERIF = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def _register_windows_fonts() -> None:
    """Make Times New Roman available when running under WSL."""
    if not _WINDOWS_FONTS.is_dir():
        return
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = _WINDOWS_FONTS / name
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
            except Exception:
                pass


def use_style(mode: str = "light", scale: float = 1.0) -> Dict[str, object]:
    """Install the rcParams for `mode` and return that mode's tokens."""
    if mode not in PALETTE:
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    _register_windows_fonts()
    t = PALETTE[mode]
    available = {f.name for f in fm.fontManager.ttflist}
    serif = [f for f in _PREFERRED_SERIF if f in available] or ["DejaVu Serif"]

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",
        "font.size": 9 * scale,
        "axes.titlesize": 10 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 8 * scale,

        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["text_secondary"],
        "axes.titlecolor": t["text_primary"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,

        # recessive, solid hairline grid -- never dashed
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,

        "xtick.color": t["text_muted"],
        "ytick.color": t["text_muted"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.labelcolor": t["text_secondary"],
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.2,

        "axes.prop_cycle": mpl.cycler(color=list(t["categorical"])),
        "text.color": t["text_primary"],
        "figure.constrained_layout.use": True,
    })
    return t


# ------------------------------------------------------------------- helpers

def categorical(mode: str = "light", n: int | None = None,
                all_pairs: bool = False) -> List[str]:
    """Categorical hues in their fixed published order.

    `all_pairs=True` for forms where any two series can end up side by side;
    that caps the list at the slots which clear the all-pairs gates.
    """
    hues = list(PALETTE[mode]["categorical"])
    if all_pairs:
        hues = hues[:ALL_PAIRS_CAP]
    if n is None:
        return hues
    if n > len(hues):
        raise ValueError(
            f"{n} series requested but only {len(hues)} slots are validated; "
            "fold the tail into 'Other', facet into small multiples, or use "
            "composite encoding -- never generate another hue"
        )
    return hues[:n]


def emphasis(n: int, highlight, mode: str = "light",
             accents: Sequence[int] = (0, 1, 2)) -> List[str]:
    """Colours for 'one (or a few) of these is the point'.

    Everything that is not highlighted gets the de-emphasis grey.
    """
    t = PALETTE[mode]
    hi = {highlight} if isinstance(highlight, int) else set(highlight)
    hues = list(t["categorical"])
    out, k = [], 0
    for i in range(n):
        if i in hi:
            out.append(hues[accents[k % len(accents)]])
            k += 1
        else:
            out.append(t["deemph"])
    return out


def sequential_cmap(mode: str = "light", name: str = "grok_seq"):
    """One hue, light -> dark (reversed for the dark surface)."""
    return LinearSegmentedColormap.from_list(name, list(PALETTE[mode]["sequential"]))


def diverging_cmap(mode: str = "light", name: str = "grok_div"):
    """Two opposite hues with a neutral grey midpoint."""
    t = PALETTE[mode]
    return LinearSegmentedColormap.from_list(
        name, [t["diverging_neg"], t["diverging_mid"], t["diverging_pos"]])


def save(fig, name: str, outdir: str | Path = "figures", mode: str = "light",
         formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Write a figure as raster (for the README) and vector (for the report)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = name if mode == "light" else f"{name}_dark"
    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        # no creation date in the PDF, so an unchanged figure regenerates byte for byte
        fig.savefig(p, format=ext, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    return paths
