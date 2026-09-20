#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
while kill -0 118505 2>/dev/null; do sleep 20; done
echo "sweep trajectories done $(date +%T)"
for spec in "B_sub_s0 sub 113" "B_mul_s0 mul 113" "B_add_s1 add 113" "C_add_f32 add 113" "C_add_nowarm add 113" "B_sqx_p113 sq_sum_cross 113" "B_sqx_p109 sq_sum_cross 109"; do
  set -- $spec
  python3 scripts/analyze_run.py --tag $1 --op $2 --p $3 --threads 2 --no-neurons > logs/traj_$1.log 2>&1
  echo "done $1 $(date +%T)"
done
echo "ALL TRAJECTORIES DONE $(date +%T)"
