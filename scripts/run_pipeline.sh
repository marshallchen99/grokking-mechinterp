#!/bin/bash
# Run one or more job files in sequence, then analyse everything they produced.
#
# Sequence, not concurrency: on this machine a single training run saturates
# around six threads and adding more processes lowers total throughput, because
# the limit is memory bandwidth rather than cores.  Three concurrent runs took
# the mainline from 118 ms/step to 618.  So each wave finishes before the next
# starts, and only the jobs *within* a wave run in parallel.
#
#   ./scripts/run_pipeline.sh jobs_a.json jobs_b.json
#
set -euo pipefail
cd "$(dirname "$0")/.."
PARALLEL=${PARALLEL:-2}
THREADS=${THREADS:-5}

for spec in "$@"; do
  echo "=== $spec  $(date +%T) ==="
  python3 scripts/run_many.py --jobs "scripts/$spec" --parallel "$PARALLEL" --threads "$THREADS"
done

echo "=== analysing  $(date +%T) ==="
for f in results/*_history.json; do
  tag=$(basename "$f" _history.json)
  [ -f "results/${tag}_analysis.json" ] && continue
  read -r op p frac <<<"$(python3 - "$f" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
print(h["data"]["op"], h["data"]["p"], h["data"]["train_frac"])
PY
)"
  python3 scripts/analyze_run.py --tag "$tag" --op "$op" --p "$p" --train-frac "$frac" \
      --threads 2 --no-neurons > "logs/traj_${tag}.log" 2>&1 || echo "  analysis failed: $tag"
  echo "  analysed $tag"
done

python3 scripts/analyze_prediction.py --at-steps 200 500 1000 2000 > logs/prediction.log 2>&1 || true
python3 scripts/make_report.py  > logs/report.log  2>&1
python3 scripts/make_figures.py > logs/figures.log 2>&1
python3 scripts/export_web.py   > logs/export.log  2>&1
echo "=== done  $(date +%T) ==="
