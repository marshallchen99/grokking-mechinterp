#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
echo "=== 16 forecast seeds $(date +%T) ==="
python3 scripts/run_many.py --jobs scripts/jobs_forecast.json --parallel 4 --threads 3
echo "=== quadratic form, 60k budget $(date +%T) ==="
python3 scripts/run_many.py --jobs scripts/jobs_sqx2.json --parallel 2 --threads 5
echo "ROUND 2 DONE $(date +%T)"
