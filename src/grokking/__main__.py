"""`python -m grokking` regenerates the README tables from results/."""
import argparse
from pathlib import Path

from .report import render_readme, missing_blocks
from .report_blocks import OP_TAGS, build

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main_add_s0")
ap.add_argument("--op-tags", nargs="*", default=None)   # default: OP_TAGS
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
args = ap.parse_args()

root = Path(args.root)
blocks = build(root, args.tag, op_tags=args.op_tags or OP_TAGS)
readme = root / "README.md"
done = render_readme(readme, blocks)
print(f"rendered {len(done)} blocks: {done}")
miss = missing_blocks(readme, blocks)
if miss:
    print(f"WARNING: marked blocks with no generator: {miss}")
