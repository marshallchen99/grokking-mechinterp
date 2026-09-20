#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
for w in w1 w2 w3; do
  echo "=== wave $w starting $(date +%T) ==="
  python3 scripts/run_many.py --jobs scripts/jobs_$w.json --parallel 2
  echo "=== wave $w done $(date +%T) ==="
done
echo "ALL WAVES DONE $(date +%T)"
