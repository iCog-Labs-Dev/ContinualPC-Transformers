#!/usr/bin/env bash
# The Objective 2-3 set under ePC on BERT, at the benchmark ePC setting (T=5, η0=1e-4, d*=0.17).
# Usage: scripts/run_epc_suite.sh <command>
#   curve            lr 1e-5, 2e-5, 5e-5, 1e-4, both orders, seeds 0-2 (24 runs;
#                    3e-5 is the benchmark epc_{fwd,rev}_lr3e-5_seed*)
#   orders           lr 3e-5 in nwr, rnw, rwn, wnr, seeds 0-2 (12 runs)
#   rollback         rollback.json of the 12 orders runs
#   maps             spike-only (BP) and learner (ePC) onset maps and predictors of
#                    all six orders' runs (after rollback; stops unless every
#                    stage_*.npz exists; merges into, never overwrites, existing
#                    error_maps.json and predictors.json), then the frozen
#                    predictor test with --map_source learner and bp
#   gates_calibrate  uniform c in {0.25,0.5,0.75} and none (c = 1, logs update_norm),
#                    both orders, seed 0 (8 runs)
#   gates            β=2 map (learner), map (bp), type, shuffled, freeze MLPs,
#                    freeze bottom, both orders, seeds 0-2 (36 runs)
#   gates_matched    one uniform c* (stage 2-3 path matched to the learner map
#                    gate), both orders, seeds 0-2 (6 runs)
#   online           online_protect β=2 and online_surprise on the learner map,
#                    live + replay, then each rule's matched uniform, both
#                    orders, seeds 0-2 (36 runs; needs gates_calibrate)
#   commutator       news/wiki 2-stage: none, learner map, its matched uniform,
#                    online protect (learner), its matched uniform, both orders,
#                    seeds 0-2, plus calibration (36 runs) and commutator.json
#   frontier         frontier.json against the ePC lr curve (diverged curve runs
#                    listed under failed_reference_runs), primary test on the
#                    learner map gates and on the bp map gates; bp: primary test
#                    only (no contrasts, no matched control) (2 analyses)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
LR=3e-5
CURVE_LRS=(1e-5 2e-5 5e-5 1e-4)
SIX_ORDERS=(fwd rev nwr rnw rwn wnr)
SCALES=(0.25 0.5 0.75)
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000
)
EPC=(
  --mode pc --pc_inference epc --pc_eta_infer 1e-4 --pc_infer_steps 5
  --pc_forward_updates none --pc_eta_control energy --pc_energy_target 0.17
  --pc_alignment_batches 2
)
LEARNER=(--gate_map_source learner)
MAP=(--gate map --gate_beta 2 --gate_map_windows 256)
ONLINE=(--gate_interval 50 --gate_ema 0.3 --gate_map_windows 256)
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
    nwr) echo "news wiki reviews" ;;
    rnw) echo "reviews news wiki" ;;
    rwn) echo "reviews wiki news" ;;
    wnr) echo "wiki news reviews" ;;
    pair_fwd) echo "news wiki" ;;
    pair_rev) echo "wiki news" ;;
  esac
}

rule_args() {
  case "$1" in
    protect) echo "--gate online_protect --gate_beta 2" ;;
    surprise) echo "--gate online_surprise" ;;
  esac
}

run() {
  local name=$1 order=$2 seed=$3 lr=$4
  shift 4
  local out="$ROOT/epcsuite_${name}_${order}_seed${seed}"
  # shellcheck disable=SC2046
  "$PYTHON" examples/mlm_domain_forgetting.py "${COMMON[@]}" "${EPC[@]}" --lr "$lr" \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee -a "$out.log"
}

sequential_dir() {
  case "$1" in
    fwd | rev) echo "$ROOT/epc_$1_lr${LR}_seed$2" ;;
    *) echo "$ROOT/epcsuite_seq_$1_seed$2" ;;
  esac
}

next_seed() {
  local seeds=($SEEDS) i
  for i in "${!seeds[@]}"; do
    if [[ ${seeds[$i]} == "$1" ]]; then
      echo "${seeds[$(((i + 1) % ${#seeds[@]}))]}"
      return
    fi
  done
  exit 1
}

matched_scale() {
  "$PYTHON" - "$ROOT" "$1" "$2" "$3" "$SEEDS" <<'EOF'
import sys
import numpy as np
from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, gated, prefix, stages, seeds = sys.argv[1:6]
stages, seeds = int(stages), seeds.split()
orders = ("pair_fwd", "pair_rev") if stages == 2 else ("fwd", "rev")


def path(run):
    return sum(s["total"] for s in read_paths(f"{root}/{run}", stages)[1:])


target = np.mean([path(f"epcsuite_{gated}_{o}_seed{s}") for o in orders for s in seeds])
scales = [0.25, 0.5, 0.75, 1.0]
runs = {c: f"epcsuite_{prefix}uniform_s{c:g}" for c in scales[:-1]} | {
    1.0: f"epcsuite_{prefix}none"
}
paths = [np.mean([path(f"{runs[c]}_{o}_seed0") for o in orders]) for c in scales]
if np.any(np.diff(paths) <= 0):
    sys.exit(f"calibration paths {paths} do not increase with the scale")
if not paths[0] <= target <= paths[-1]:
    sys.exit(f"{gated} path {target:.4g} outside calibration {paths}")
print(f"{float(np.interp(target, paths, scales)):.4f}")
EOF
}

check_scale_free() {
  "$PYTHON" - "$ROOT" "$1" "$2" "$3" <<'EOF'
import json
import sys
from pathlib import Path

root, scale, name, prefix = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
own = f"epcsuite_{prefix}matched_{name}_"
for config in sorted(root.glob(f"epcsuite_{prefix}matched_*_seed0/run_config.json")):
    other = config.parent.name
    if other.startswith(own):
        continue
    if json.loads(config.read_text())["config"].get("gate_scale") == scale:
        sys.exit(
            f"matched gate_scale {scale:g} of {name} equals that of {other}; their "
            f"frontier and commutator labels (uniform_s{scale:g}) would collide, stopping"
        )
EOF
}

require_snapshots() {
  local out stage missing=0
  for out in "$@"; do
    for stage in 0 1 2 3; do
      if [ ! -f "$out/stage_$stage.npz" ]; then
        echo "missing $out/stage_$stage.npz" >&2
        missing=1
      fi
    done
  done
  if [ "$missing" = 1 ]; then
    echo "maps needs every run's stage snapshots; stopping before any analysis" >&2
    exit 1
  fi
}

case "${1:-}" in
  curve)
    for lr in "${CURVE_LRS[@]}"; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          run "curve_lr$lr" "$order" "$seed" "$lr"
        done
      done
    done
    ;;
  orders)
    for order in "${SIX_ORDERS[@]:2}"; do
      for seed in $SEEDS; do
        run seq "$order" "$seed" "$LR"
      done
    done
    ;;
  rollback)
    for order in "${SIX_ORDERS[@]:2}"; do
      for seed in $SEEDS; do
        out=$(sequential_dir "$order" "$seed")
        "$PYTHON" examples/domain_forgetting_analysis.py rollback --run_dir "$out" \
          2>&1 | tee -a "$out.rollback.log"
      done
    done
    ;;
  maps)
    dirs=()
    for order in "${SIX_ORDERS[@]}"; do
      for seed in $SEEDS; do
        dirs+=("$(sequential_dir "$order" "$seed")")
      done
    done
    require_snapshots "${dirs[@]}"
    for out in "${dirs[@]}"; do
        "$PYTHON" examples/domain_forgetting_analysis.py error_maps --run_dir "$out" \
          --settings spike_only --map_source learner 2>&1 | tee -a "$out.maps.log"
        "$PYTHON" examples/domain_forgetting_analysis.py predictors --run_dir "$out" \
          2>&1 | tee -a "$out.maps.log"
    done
    for source in learner bp; do
      "$PYTHON" examples/domain_forgetting_analysis.py architecture --run_dirs "${dirs[@]}" \
        --map_source "$source" --output_dir "$ROOT/epcsuite_architecture_$source" \
        2>&1 | tee -a "$ROOT/epcsuite_architecture_$source.log"
    done
    ;;
  gates_calibrate)
    for order in fwd rev; do
      run none "$order" 0 "$LR"
      for c in "${SCALES[@]}"; do
        run "uniform_s$c" "$order" 0 "$LR" --gate uniform --gate_scale "$c"
      done
    done
    ;;
  gates)
    for order in fwd rev; do
      for seed in $SEEDS; do
        run map_b2_learner "$order" "$seed" "$LR" "${MAP[@]}" "${LEARNER[@]}"
        run map_b2_bp "$order" "$seed" "$LR" "${MAP[@]}"
        run type_b2 "$order" "$seed" "$LR" --gate type --gate_beta 2 \
          --gate_map_windows 256 "${LEARNER[@]}"
        run random_b2 "$order" "$seed" "$LR" --gate random --gate_beta 2 \
          --gate_map_windows 256 --gate_seed 0 "${LEARNER[@]}"
        run freezemlp "$order" "$seed" "$LR" --gate freeze --gate_modules mlp
        run freezebottom "$order" "$seed" "$LR" --gate freeze \
          --gate_modules embeddings blocks_0-1
      done
    done
    ;;
  gates_matched)
    c=$(matched_scale map_b2_learner "" 3)
    echo "map (learner) matched gate_scale=$c"
    check_scale_free "$c" map ""
    for order in fwd rev; do
      for seed in $SEEDS; do
        run matched_map "$order" "$seed" "$LR" --gate uniform --gate_scale "$c"
      done
    done
    ;;
  online)
    for rule in protect surprise; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2046
          run "online_${rule}" "$order" "$seed" "$LR" $(rule_args "$rule") \
            "${ONLINE[@]}" "${LEARNER[@]}"
        done
      done
      for order in fwd rev; do
        for seed in $SEEDS; do
          source_seed=$(next_seed "$seed")
          # shellcheck disable=SC2046
          run "online_${rule}_replay" "$order" "$seed" "$LR" $(rule_args "$rule") \
            "${ONLINE[@]}" "${LEARNER[@]}" --gate_variant replay \
            --gate_source "$ROOT/epcsuite_online_${rule}_${order}_seed${source_seed}"
        done
      done
      c=$(matched_scale "online_$rule" "" 3)
      echo "online $rule matched gate_scale=$c"
      check_scale_free "$c" "online_$rule" ""
      for order in fwd rev; do
        for seed in $SEEDS; do
          run "matched_online_$rule" "$order" "$seed" "$LR" --gate uniform --gate_scale "$c"
        done
      done
    done
    ;;
  commutator)
    for order in pair_fwd pair_rev; do
      run commutator_none "$order" 0 "$LR"
      for c in "${SCALES[@]}"; do
        run "commutator_uniform_s$c" "$order" 0 "$LR" --gate uniform --gate_scale "$c"
      done
      for seed in $SEEDS; do
        if [ "$seed" != 0 ]; then
          run commutator_none "$order" "$seed" "$LR"
        fi
        run commutator_map_b2 "$order" "$seed" "$LR" "${MAP[@]}" "${LEARNER[@]}"
        run commutator_online_protect "$order" "$seed" "$LR" \
          --gate online_protect --gate_beta 2 "${ONLINE[@]}" "${LEARNER[@]}"
      done
    done
    c_map=$(matched_scale commutator_map_b2 commutator_ 2)
    c_online=$(matched_scale commutator_online_protect commutator_ 2)
    echo "commutator matched gate_scale: map_b2 $c_map, online_protect $c_online"
    if [ "$c_map" = "$c_online" ]; then
      echo "both matched scales are $c_map; the two matched conditions would share" \
        "the label uniform_s$c_map in commutator.json, stopping" >&2
      exit 1
    fi
    for gated in map_b2 online_protect; do
      if [ "$gated" = map_b2 ]; then c=$c_map; else c=$c_online; fi
      for order in pair_fwd pair_rev; do
        for seed in $SEEDS; do
          run "commutator_matched_$gated" "$order" "$seed" "$LR" \
            --gate uniform --gate_scale "$c"
        done
      done
    done
    shopt -s nullglob
    "$PYTHON" examples/domain_forgetting_analysis.py commutator \
      --run_dirs "$ROOT"/epcsuite_commutator_none_*_seed? \
        "$ROOT"/epcsuite_commutator_map_b2_*_seed? \
        "$ROOT"/epcsuite_commutator_online_protect_*_seed? \
        "$ROOT"/epcsuite_commutator_matched_*_seed? \
      --output_dir "$ROOT/epcsuite_commutator_bert" 2>&1 |
      tee -a "$ROOT/epcsuite_commutator_bert.log"
    ;;
  frontier)
    shopt -s nullglob
    runs=(
      "$ROOT"/epcsuite_curve_*_seed? "$ROOT"/epc_fwd_lr${LR}_seed? "$ROOT"/epc_rev_lr${LR}_seed?
      "$ROOT"/epcsuite_map_b2_*_seed? "$ROOT"/epcsuite_type_b2_*_seed?
      "$ROOT"/epcsuite_random_b2_*_seed? "$ROOT"/epcsuite_freeze*_seed?
      "$ROOT"/epcsuite_matched_*_seed? "$ROOT"/epcsuite_online_*_seed?
    )
    if [ ${#runs[@]} -eq 0 ]; then
      echo "no runs under $ROOT" >&2
      exit 1
    fi
    for source in learner bp; do
      "$PYTHON" examples/domain_forgetting_analysis.py frontier --run_dirs "${runs[@]}" \
        --reference_mode pc --pc_inference epc --map_source "$source" --seq_lr "$LR" \
        --output_dir "$ROOT/epcsuite_frontier_$source" 2>&1 |
        tee -a "$ROOT/epcsuite_frontier_$source.log"
    done
    ;;
  *)
    sed -n 2,29p "$0"
    exit 1
    ;;
esac
