#!/bin/bash
# Wait for run A to finish, then start wave 1.  Running them concurrently on
# this machine lowers total throughput (memory bandwidth, not cores, is the
# limit), so the waves are strictly sequential.
cd /mnt/d/Portfolio/grokking-mechinterp
while pgrep -f "run_train.py --tag main_add_s0" > /dev/null; do sleep 20; done
echo "run A finished at $(date +%T); starting wave 1"
python3 scripts/run_many.py --jobs scripts/jobs_w1.json --parallel 2
echo "wave 1 finished at $(date +%T); starting wave 2"
python3 scripts/run_many.py --jobs scripts/jobs_w2.json --parallel 2
echo "wave 2 finished at $(date +%T); starting wave 3"
python3 scripts/run_many.py --jobs scripts/jobs_w3.json --parallel 2
echo "all waves finished at $(date +%T)"
