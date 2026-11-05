#!/usr/bin/env bash
# Influence diagnostics and the influence-weighted online gate on BERT, BP and ePC.
# Usage: scripts/run_influence_gates.sh <command>
#   diagnostics  ablation and perturbation at each stage onset of g9_* and epc_*
#                lr 3e-5, both orders, seeds 0-2; needs their stage snapshots and
#                rollback.json (no training)
#   live         online_influence, both orders, seeds 0-2 per learner (12 runs)
#   matched      BP calibration (uniform c in {0.25,0.5,0.75} and none, both orders,
#                seed 0; 8 runs), then one uniform c* per learner, stage 2-3 path
#                matched to live, both orders, seeds 0-2 (12 runs); ePC calibrates
#                on epcsuite_uniform_s* and epcsuite_none (run_epc_suite.sh
#                gates_calibrate)
#   past         online_influence probing the previous domain, reported separately
#                as replay-using, both orders, seeds 0-2 per learner (12 runs)
#   commutator   news/wiki 2-stage: online_influence and its matched uniform, both
#                orders, seeds 0-2 per learner (24 runs), BP calibration (8 runs),
#                then infl_commutator_<learner>/commutator.json; ePC calibrates on
#                epcsuite_commutator_uniform_s* and epcsuite_commutator_none
#   commutator_surprise  news/wiki 2-stage: online_surprise and its matched uniform,
#                both orders, seeds 0-2 per learner (24 runs), then
#                infl_commutator_surprise_<learner>/commutator.json; reuses the
#                calibration of the commutator step
#   surprise_bp  BP online_surprise at BP_LR (6 runs), run before verdict: the
#                reused online_surprise_* runs are at lr 1e-4, so BP's comparison
#                with surprise is judged on these
#   verdict      infl_frontier_<learner>/frontier.json with all three comparisons
#                under both computations; the past variant and any cross-lr row
#                are descriptive, and the past variant's path ratio to its
#                control is printed
#   prune        delete stage_*.npz of finished live, past, matched, surprise and
#                calibration runs; keeps those of infl_*_pair_{live,matched}_*
# Env: ROOT, SEEDS, PYTHON, LEARNERS ("bp epc"), BP_LR (3e-5), PRUNE=1 prunes each
# finished run as soon as its summary.json exists.
# A run with a summary.json and no snapshots is skipped: --resume would retrain it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
LEARNERS=${LEARNERS:-"bp epc"}
BP_LR=${BP_LR:-3e-5}
EPC_LR=3e-5
PRUNE=${PRUNE:-0}
SCALES=(0.25 0.5 0.75)
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000
)
BP=(--mode backprop --lr "$BP_LR")
EPC=(
  --mode pc --pc_inference epc --pc_eta_infer 1e-4 --pc_infer_steps 5
  --pc_forward_updates none --pc_eta_control energy --pc_energy_target 0.17
  --pc_alignment_batches 2 --lr "$EPC_LR"
)
ONLINE=(--gate_interval 50 --gate_ema 0.3 --gate_map_windows 256)
GATE=(--gate online_influence "${ONLINE[@]}")
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
    pair_fwd) echo "news wiki" ;;
    pair_rev) echo "wiki news" ;;
  esac
}

learner_args() {
  case "$1" in
    bp) echo "${BP[*]}" ;;
    epc) echo "${EPC[*]}" ;;
  esac
}

gate_args() {
  case "$1" in
    bp) echo "${GATE[*]}" ;;
    epc) echo "${GATE[*]} --gate_map_source learner" ;;
  esac
}

prune_run() {
  if [ -f "$1/summary.json" ]; then
    rm -f "$1"/stage_*.npz
  else
    echo "not pruning $1: no summary.json" >&2
  fi
}

run() {
  local learner=$1 name=$2 order=$3 seed=$4 prunable=$5
  shift 5
  local out="$ROOT/infl_${learner}_${name}_${order}_seed${seed}"
  if [ -f "$out/summary.json" ] && [ ! -f "$out/stage_0.npz" ]; then
    echo "skipping $out: finished, snapshots pruned"
    return
  fi
  # shellcheck disable=SC2046
  "$PYTHON" examples/mlm_domain_forgetting.py "${COMMON[@]}" $(learner_args "$learner") \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee -a "$out.log"
  if [ "$PRUNE" = 1 ] && [ "$prunable" = yes ]; then
    prune_run "$out"
  fi
}

calibration_run() {
  local learner=$1 prefix=$2 scale=$3 order=$4
  if [ "$learner" = bp ]; then
    if [ "$scale" = 1 ]; then
      echo "infl_bp_${prefix}none_${order}_seed0"
    else
      echo "infl_bp_${prefix}uniform_s${scale}_${order}_seed0"
    fi
    return
  fi
  local suite=""
  if [ -n "$prefix" ]; then
    suite=commutator_
  fi
  if [ "$scale" = 1 ]; then
    echo "epcsuite_${suite}none_${order}_seed0"
  else
    echo "epcsuite_${suite}uniform_s${scale}_${order}_seed0"
  fi
}

matched_scale() {
  local learner=$1 prefix=$2 stages=$3 gated=${4:-live} order runs=() scale
  local orders=(fwd rev)
  if [ "$stages" = 2 ]; then
    orders=(pair_fwd pair_rev)
  fi
  for order in "${orders[@]}"; do
    for scale in "${SCALES[@]}" 1; do
      runs+=("$scale=$(calibration_run "$learner" "$prefix" "$scale" "$order")")
    done
  done
  "$PYTHON" - "$ROOT" "infl_${learner}_${prefix}${gated}" "$stages" "$SEEDS" "${orders[*]}" \
    "${runs[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np

from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, gated, stages, seeds, orders = sys.argv[1:6]
stages, seeds, orders = int(stages), seeds.split(), orders.split()


def path(run):
    if not (Path(root) / run / "summary.json").is_file():
        sys.exit(f"{root}/{run} is not a finished run; run its calibration first")
    return sum(s["total"] for s in read_paths(f"{root}/{run}", stages)[1:])


target = np.mean([path(f"{gated}_{o}_seed{s}") for o in orders for s in seeds])
by_scale = {}
for item in sys.argv[6:]:
    scale, run = item.split("=")
    by_scale.setdefault(float(scale), []).append(path(run))
scales = sorted(by_scale)
paths = [float(np.mean(by_scale[c])) for c in scales]
if np.any(np.diff(paths) <= 0):
    sys.exit(f"calibration paths {paths} do not increase with the scale")
if not paths[0] <= target <= paths[-1]:
    sys.exit(f"{gated} path {target:.4g} outside calibration {paths}")
print(f"{float(np.interp(target, paths, scales)):.4f}")
PY
}

label_of() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

from fabricpc.experiments.domain_forgetting.frontier import describe_gate

config = json.loads((Path(sys.argv[1]) / "run_config.json").read_text())["config"]
print(describe_gate(config)[2])
PY
}

calibrate_bp() {
  local prefix=$1 order c prunable=yes
  shift
  for order in "$@"; do
    run bp "${prefix}none" "$order" 0 "$prunable"
    for c in "${SCALES[@]}"; do
      run bp "${prefix}uniform_s$c" "$order" 0 "$prunable" --gate uniform --gate_scale "$c"
    done
  done
}

first_seed() {
  local seeds=($SEEDS)
  echo "${seeds[0]}"
}

case "${1:-}" in
  diagnostics)
    for learner in $LEARNERS; do
      if [ "$learner" = bp ]; then base=g9; else base=epc; fi
      dirs=()
      for order in fwd rev; do
        for seed in $SEEDS; do
          dirs+=("$ROOT/${base}_${order}_lr3e-5_seed${seed}")
        done
      done
      "$PYTHON" examples/domain_forgetting_analysis.py influence --run_dirs "${dirs[@]}" \
        --output_dir "$ROOT/infl_diagnostics_$learner" 2>&1 |
        tee -a "$ROOT/infl_diagnostics_$learner.log"
    done
    ;;
  live)
    for learner in $LEARNERS; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2046
          run "$learner" live "$order" "$seed" yes $(gate_args "$learner")
        done
      done
    done
    ;;
  matched)
    for learner in $LEARNERS; do
      if [ "$learner" = bp ]; then
        calibrate_bp "" fwd rev
      fi
      c=$(matched_scale "$learner" "" 3)
      echo "$learner online_influence matched gate_scale=$c"
      for order in fwd rev; do
        for seed in $SEEDS; do
          run "$learner" matched "$order" "$seed" yes --gate uniform --gate_scale "$c"
        done
      done
    done
    ;;
  past)
    for learner in $LEARNERS; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2046
          run "$learner" past "$order" "$seed" yes $(gate_args "$learner") --gate_variant past
        done
      done
    done
    ;;
  commutator)
    for learner in $LEARNERS; do
      for order in pair_fwd pair_rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2046
          run "$learner" pair_live "$order" "$seed" no $(gate_args "$learner")
        done
      done
      if [ "$learner" = bp ]; then
        calibrate_bp pair_ pair_fwd pair_rev
      fi
      c=$(matched_scale "$learner" pair_ 2)
      echo "$learner commutator matched gate_scale=$c"
      for order in pair_fwd pair_rev; do
        for seed in $SEEDS; do
          run "$learner" pair_matched "$order" "$seed" no --gate uniform --gate_scale "$c"
        done
      done
      seed=$(first_seed)
      gate=$(label_of "$ROOT/infl_${learner}_pair_live_pair_fwd_seed${seed}")
      control=$(label_of "$ROOT/infl_${learner}_pair_matched_pair_fwd_seed${seed}")
      "$PYTHON" examples/domain_forgetting_analysis.py commutator \
        --run_dirs "$ROOT/infl_${learner}_pair_live"_pair_*_seed? \
          "$ROOT/infl_${learner}_pair_matched"_pair_*_seed? \
        --commutator_test "$gate" "$control" \
        --output_dir "$ROOT/infl_commutator_$learner" 2>&1 |
        tee -a "$ROOT/infl_commutator_$learner.log"
    done
    ;;
  commutator_surprise)
    for learner in $LEARNERS; do
      source_args=""
      if [ "$learner" = epc ]; then
        source_args="--gate_map_source learner"
      fi
      for order in pair_fwd pair_rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2086
          run "$learner" pair_surprise "$order" "$seed" no --gate online_surprise \
            "${ONLINE[@]}" $source_args
        done
      done
      c=$(matched_scale "$learner" pair_ 2 surprise)
      echo "$learner surprise commutator matched gate_scale=$c"
      for order in pair_fwd pair_rev; do
        for seed in $SEEDS; do
          run "$learner" pair_surprise_matched "$order" "$seed" no --gate uniform --gate_scale "$c"
        done
      done
      seed=$(first_seed)
      gate=$(label_of "$ROOT/infl_${learner}_pair_surprise_pair_fwd_seed${seed}")
      control=$(label_of "$ROOT/infl_${learner}_pair_surprise_matched_pair_fwd_seed${seed}")
      "$PYTHON" examples/domain_forgetting_analysis.py commutator \
        --run_dirs "$ROOT/infl_${learner}_pair_surprise"_pair_*_seed? \
          "$ROOT/infl_${learner}_pair_surprise_matched"_pair_*_seed? \
        --commutator_test "$gate" "$control" \
        --output_dir "$ROOT/infl_commutator_surprise_$learner" 2>&1 |
        tee -a "$ROOT/infl_commutator_surprise_$learner.log"
    done
    ;;
  verdict)
    shopt -s nullglob
    seed=$(first_seed)
    for learner in $LEARNERS; do
      gate=$(label_of "$ROOT/infl_${learner}_live_fwd_seed${seed}")
      control=$(label_of "$ROOT/infl_${learner}_matched_fwd_seed${seed}")
      runs=(
        "$ROOT/infl_${learner}_live"_{fwd,rev}_seed?
        "$ROOT/infl_${learner}_past"_{fwd,rev}_seed?
        "$ROOT/infl_${learner}_matched"_{fwd,rev}_seed?
      )
      if [ "$learner" = bp ]; then
        lr=$BP_LR
        reused="$ROOT/online_surprise_fwd_seed${seed}"
        runs+=(
          "$ROOT"/online_surprise_{fwd,rev}_seed?
          "$ROOT"/g3_bp_t5000000_lr1e-4 "$ROOT"/g4_bp_seed? "$ROOT"/g8_rev_bp_seed?
          "$ROOT"/g9_*_seed? "$ROOT"/g8_joint_seed?
        )
        reference=(--reference_mode backprop)
      else
        lr=$EPC_LR
        reused="$ROOT/epcsuite_online_surprise_fwd_seed${seed}"
        runs+=(
          "$ROOT"/epcsuite_online_surprise_{fwd,rev}_seed?
          "$ROOT"/epcsuite_curve_*_seed? "$ROOT"/epc_fwd_lr${EPC_LR}_seed?
          "$ROOT"/epc_rev_lr${EPC_LR}_seed?
        )
        reference=(--reference_mode pc --pc_inference epc --map_source learner)
      fi
      surprise=$(label_of "$reused")
      judged=$surprise
      verdicts=()
      if [ "$learner" = bp ]; then
        same_lr=("$ROOT/infl_bp_surprise"_{fwd,rev}_seed?)
        if [ ${#same_lr[@]} -eq 0 ]; then
          echo "no BP surprise runs at BP_LR: run the surprise_bp step first" >&2
          exit 1
        fi
        runs+=("${same_lr[@]}")
        judged=$(label_of "${same_lr[0]}")
        verdicts+=(--descriptive_verdict "$gate" "$control" "$surprise")
      fi
      verdicts+=(--influence_verdict "$gate" "$control" "$judged")
      past_runs=("$ROOT/infl_${learner}_past"_{fwd,rev}_seed?)
      if [ ${#past_runs[@]} -gt 0 ]; then
        past=$(label_of "${past_runs[0]}")
        verdicts+=(--descriptive_verdict "$past" "$control" "$judged")
        "$PYTHON" - "$ROOT" "$learner" <<'PY' | tee -a "$ROOT/infl_frontier_$learner.log"
import sys
from pathlib import Path

from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, learner = sys.argv[1:]


def path(run):
    return sum(s["total"] for s in read_paths(run, 3)[1:])


for past in sorted(Path(root).glob(f"infl_{learner}_past_*_seed?")):
    control = past.with_name(past.name.replace("_past_", "_matched_"))
    print(f"path {past.name} / matched: {path(past) / path(control):.4f}")
PY
      fi
      # shellcheck disable=SC2086
      "$PYTHON" examples/domain_forgetting_analysis.py frontier --run_dirs "${runs[@]}" \
        "${reference[@]}" --seq_lr "$lr" --expected_seeds $SEEDS "${verdicts[@]}" \
        --output_dir "$ROOT/infl_frontier_$learner" 2>&1 |
        tee -a "$ROOT/infl_frontier_$learner.log"
    done
    ;;
  surprise_bp)
    for order in fwd rev; do
      for seed in $SEEDS; do
        run bp surprise "$order" "$seed" yes --gate online_surprise "${ONLINE[@]}"
      done
    done
    ;;
  prune)
    shopt -s nullglob
    for learner in $LEARNERS; do
      for out in \
        "$ROOT/infl_${learner}"_{live,past,matched,surprise,none}_{fwd,rev}_seed? \
        "$ROOT/infl_${learner}"_uniform_s*_{fwd,rev}_seed? \
        "$ROOT/infl_${learner}"_pair_none_pair_*_seed? \
        "$ROOT/infl_${learner}"_pair_uniform_s*_pair_*_seed?; do
        prune_run "$out"
      done
    done
    ;;
  *)
    sed -n 2,32p "$0"
    exit 1
    ;;
esac
