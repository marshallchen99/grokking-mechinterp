#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
while kill -0 102769 2>/dev/null; do sleep 30; done
echo "waves finished $(date +%T)"
echo "=== wave 4: controlled comparison $(date +%T) ==="
python3 scripts/run_many.py --jobs scripts/jobs_w4.json --parallel 2 --threads 5
echo "=== sweep $(date +%T) ==="
python3 scripts/run_many.py --jobs scripts/jobs_sweep.json --parallel 3 --threads 3
echo "ALL COMPUTE DONE $(date +%T)"
