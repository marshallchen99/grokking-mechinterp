#!/usr/bin/env python3
"""Regenerate every table in the README from the results files.

The README carries <!-- BEGIN:name --> ... <!-- END:name --> markers; this
replaces what is between them and leaves the prose alone.  No number in the
write-up is typed by hand, so a table can never drift from the run it
describes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grokking.report import missing_blocks, render_readme   # noqa: E402
from grokking.report_blocks import PENDING, build           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main_add_s0")
ap.add_argument("--op-tags", nargs="*", default=None)   # default: report_blocks.OP_TAGS
ap.add_argument("--root", default=str(ROOT))
args = ap.parse_args()

root = Path(args.root)
from grokking.report_blocks import OP_TAGS
blocks = build(root, args.tag, op_tags=args.op_tags or OP_TAGS)
readme = root / "README.md"
done = render_readme(readme, blocks)
filled = [b for b in done if blocks[b] != PENDING]
pending = [b for b in done if blocks[b] == PENDING]
print(f"rendered {len(done)} blocks; {len(filled)} filled, {len(pending)} pending")
if pending:
    print(f"  pending: {pending}")
broken = [b for b, v in blocks.items() if v.startswith("_Block failed")]
if broken:
    print(f"  FAILED TO RENDER: {broken}")
    for b in broken:
        print(f"    {b}: {blocks[b]}")
miss = missing_blocks(readme, blocks)
if miss:
    print(f"  WARNING: markers with no generator: {miss}")
