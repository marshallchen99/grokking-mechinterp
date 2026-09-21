#!/bin/bash
# Rebuild every shipped results file from the checkpoints, then the README,
# the figures and the page -- the exact commands that produced them.
#
#   ./scripts/regenerate.sh              # everything (about an hour on a laptop CPU)
#   ./scripts/regenerate.sh --report     # only README, figures and page, from results/
#
# The checkpoints are not in the repository (3.7 GB).  Without them only the
# second form works.  Any failure stops the script: a half-regenerated
# results/ must not be mistaken for a fresh one.
set -euo pipefail
cd "$(dirname "$0")/.."
R=${ROOT_DIR:-$PWD}   # where results/ and checkpoints/ live (default: this checkout)
mkdir -p "$R/logs"
T=${THREADS:-2}   # the shipped mechanism files reproduce byte for byte at 2 threads

step() {   # step <log name> <command...>
  local log="$R/logs/$1.log"; shift
  echo "  $*"
  "$@" > "$log" 2>&1 || { echo "FAILED (see $log): $*"; exit 1; }
}

if [ "${1:-}" != "--report" ]; then
  [ -d "$R/checkpoints" ] || { echo "no checkpoints/ -- only --report is possible"; exit 1; }
  echo "=== trajectories  $(date +%T) ==="
  NEURON_RUNS=" main_add_s0 B_add_s0 "        # the runs whose neuron measures are reported
  for f in "$R"/results/*_history.json; do
    tag=$(basename "$f" _history.json)
    case "$tag" in R_*) continue ;; esac   # seed replicates: only their training curves are used
    case "$NEURON_RUNS" in
      *" $tag "*) step "traj_$tag" python3 scripts/analyze_run.py --root "$R" --tag "$tag" --threads "$T" ;;
      *)          step "traj_$tag" python3 scripts/analyze_run.py --root "$R" --tag "$tag" --threads "$T" --no-neurons ;;
    esac
  done

  echo "=== final-checkpoint mechanism  $(date +%T) ==="
  for tag in main_add_s0 B_add_s0 B_add_s1 B_sub_s0 C_add_f32 C_add_nowarm; do
    step "mech_$tag" python3 scripts/analyze_mechanism.py --root "$R" --tag "$tag" --threads "$T"
  done
  step mech_B_mul_s0 python3 scripts/analyze_mechanism.py --root "$R" --tag B_mul_s0 --dlog --threads "$T"
  for tag in main_add_s0 B_add_s0 B_add_s1 B_sub_s0 C_add_f32; do
    step "redund_$tag" python3 scripts/analyze_redundancy.py --root "$R" --tag "$tag" --threads "$T"
  done
  step quadratic python3 scripts/analyze_quadratic.py --root "$R" --tags Q_sqx_p59 Q_sqx_p61 --threads "$T"
  step sweep python3 scripts/analyze_sweep.py --root "$R" --pattern S_p59_ --threads "$T"
  step controls_init python3 scripts/analyze_controls_init.py --root "$R"
  step prediction python3 scripts/analyze_prediction.py --root "$R"
fi

echo "=== report  $(date +%T) ==="
step report python3 scripts/make_report.py --root "$R"        # exits non-zero on a failed or pending block
step figures python3 scripts/make_figures.py --root "$R"
step web python3 scripts/export_web.py --root "$R"
echo "=== done  $(date +%T) ==="
