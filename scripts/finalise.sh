#!/bin/bash
cd /mnt/d/Portfolio/grokking-mechinterp
while kill -0 118941 2>/dev/null || kill -0 118502 2>/dev/null; do sleep 25; done
echo "all compute finished $(date +%T)"
python3 scripts/analyze_prediction.py --at-steps 200 500 1000 2000 > logs/prediction.log 2>&1
echo "prediction done"
python3 scripts/analyze_sweep.py --pattern "S_p59_" --out sweep_summary.json --threads 4 > logs/sweep2.log 2>&1
python3 scripts/make_report.py --tag main_add_s0 --op-tags B_add_s0 B_sub_s0 B_mul_s0 B_sqx_p113 B_sqx_p109 > logs/report.log 2>&1
python3 scripts/make_figures.py --tag main_add_s0 --op-tags main_add_s0 B_add_s0 B_sub_s0 B_mul_s0 B_sqx_p113 B_sqx_p109 > logs/figures.log 2>&1
python3 scripts/export_web.py --tag main_add_s0 --op-tags B_add_s0 B_sub_s0 B_mul_s0 B_sqx_p113 B_sqx_p109 > logs/export.log 2>&1
echo "FINALISED $(date +%T)"
