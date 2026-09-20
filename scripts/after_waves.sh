#!/bin/bash
# Wait on a PID, not on a command-line pattern: pgrep -f matches any process
# whose command line contains the string, including the shell that launched
# this one, which makes a pattern-based wait loop hang forever.
cd /mnt/d/Portfolio/grokking-mechinterp
while kill -0 102769 2>/dev/null; do sleep 30; done
echo "waves finished $(date +%T); starting the sweep"
python3 scripts/run_many.py --jobs scripts/jobs_sweep.json --parallel 3 --threads 3
echo "sweep finished $(date +%T)"
