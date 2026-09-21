#!/bin/bash
# Train the runs in one or more job files, then regenerate everything.
#
#   ./scripts/run_pipeline.sh jobs_sweep.json jobs_repl.json
#
# Sequence, not concurrency: on this machine a single training run saturates
# around six threads and adding more processes lowers total throughput, because
# the limit is memory bandwidth rather than cores.  Three concurrent runs took
# the mainline from 118 ms/step to 618.  So each job file finishes before the
# next starts, and only the jobs *within* a file run in parallel.
#
# A tag that already has checkpoints is refused rather than overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PARALLEL=${PARALLEL:-2}
THREADS=${THREADS:-5}

for spec in "$@"; do
  echo "=== $spec  $(date +%T) ==="
  python3 scripts/run_many.py --jobs "scripts/$spec" --parallel "$PARALLEL" --threads "$THREADS"
done

exec ./scripts/regenerate.sh
