"""Generate the README's tables from the results files.

Every number in the write-up is produced by this module, reading the JSON that
the experiments actually wrote.  Nothing is typed in by hand, so a table can
never drift away from the run it describes, and re-running an experiment
re-generates its numbers.  The README carries

    <!-- BEGIN:name --> ... <!-- END:name -->

markers; `render_readme` replaces what is between them and leaves the prose
alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MARK = "<!-- {kind}:{name} -->"


# ----------------------------------------------------------------- loading

def load_history(root: Path, tag: str) -> Optional[Dict]:
    p = root / "results" / f"{tag}_history.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_analysis(root: Path, tag: str) -> Optional[Dict]:
    p = root / "results" / f"{tag}_analysis.json"
    return json.loads(p.read_text()) if p.exists() else None


def is_finished(h) -> bool:
    """Has this run reached the step count it was configured for?

    History files are flushed periodically while a run is in progress, so a
    partial one is perfectly well-formed and will happily render as a finished
    experiment.  Everything that consumes a history must check.
    """
    return bool(h) and h["history"] and h["history"][-1]["step"] >= h["train_cfg"]["steps"]


def load_json(root: Path, name: str) -> Optional[Dict]:
    p = root / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------ table making

def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 1000):
        return f"{int(v):+,}" if isinstance(v, int) and v < 0 else f"{int(v):,}"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        # respect the precision a caller already rounded to, rather than
        # padding 0.6 out to 0.6000
        dec = len(str(v).split(".")[1]) if "." in str(v) else 0
        return f"{v:.{min(nd, max(dec, 1))}f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def table(headers: Sequence[str], rows: Sequence[Sequence], align: Optional[str] = None,
          bold_best: Optional[Dict[int, str]] = None) -> str:
    """A GitHub-flavoured markdown table.

    `bold_best` maps a column index to "min" or "max"; that column's best
    numeric cell is emboldened.
    """
    body = [[_fmt(c) for c in r] for r in rows]
    if bold_best:
        for col, how in bold_best.items():
            vals = []
            for i, r in enumerate(rows):
                v = r[col]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append((v, i))
            if vals:
                best = (min if how == "min" else max)(vals)[1]
                body[best][col] = f"**{body[best][col]}**"
    al = align or ("l" + "r" * (len(headers) - 1))
    sep = {"l": ":--", "r": "--:", "c": ":-:"}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(sep[a] for a in al) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


# --------------------------------------------------------------- rendering

def render_readme(readme_path: Path, blocks: Dict[str, str]) -> List[str]:
    """Replace each marked block in the README; return the names replaced."""
    text = readme_path.read_text()
    replaced = []
    for name, content in blocks.items():
        begin = MARK.format(kind="BEGIN", name=name)
        end = MARK.format(kind="END", name=name)
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            continue
        text = pattern.sub(f"{begin}\n{content}\n{end}", text)
        replaced.append(name)
    readme_path.write_text(text)
    return replaced


def missing_blocks(readme_path: Path, blocks: Dict[str, str]) -> List[str]:
    """Marked blocks in the README that no generator produced -- a sign of drift."""
    text = readme_path.read_text()
    present = set(re.findall(r"<!-- BEGIN:([A-Za-z0-9_.-]+) -->", text))
    return sorted(present - set(blocks))
