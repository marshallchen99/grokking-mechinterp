#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
for f in results/S_p59_*_history.json; do
  tag=$(basename "$f" _history.json)
  frac=$(python3 -c "import json;print(json.load(open('$f'))['data']['train_frac'])")
  python3 scripts/analyze_run.py --tag "$tag" --p 59 --train-frac "$frac" \
    --threads 1 --no-neurons > "logs/traj_$tag.log" 2>&1
  echo "done $tag"
done
