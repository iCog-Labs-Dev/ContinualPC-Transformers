#!/usr/bin/env bash
# Retention baselines (L2-SP, EWC) and the optimizer-restart control on BERT, BP and ePC.
# Usage: scripts/run_retention_baselines.sh <command>
#   pilot      seed 0, both orders, four strengths per method and learner (32 runs)
#   select     per learner, ret_selection_<learner>/retention_selection.json: lowest
#              mean final CE over both orders among strengths with mean gain >= 90%
#              of the learner's ungated seed-0 run (g9_* for BP, epc_*_lr3e-5 for ePC);
#              prints every strength and flags an endpoint or no qualifying strength
#   confirm    the selected strength, seeds other than 0, both orders (16 runs)
#   calibrate  trains uniform_all c in 0.25 0.5 0.75 and EXTRA_SCALES on seed 0, both
#              orders, as ret_<learner>_uniformall_s<c>_* (12 runs), then checks them
#              and the ungated c = 1 runs: BP infl_bp_none_*, ePC epcsuite_none_*
#   matched    one uniform_all c* per method and learner, stage 2-3 path matched to
#              the selected strength, both orders, seeds 0-2 (24 runs)
#   restart    --keep_optimizer_state, both orders, seeds 0-2 per learner (12 runs)
#   verdict    ret_frontier_<learner>_<method>/frontier.json: baseline against its
#              frontier and matched uniform under both computations, all seeds and
#              seeds other than 0; ret_frontier_<learner>_restart/frontier.json:
#              keep-optimizer runs, descriptive against the restarted runs
#   prune      delete stage_*.npz and optimizer_state_stage_*.npz of finished ret_* runs
# Env: ROOT, SEEDS, PYTHON, LEARNERS ("bp epc"), METHODS ("l2sp ewc"), EXTRA_SCALES
# (""), PRUNE=1 prunes each finished run as soon as its summary.json exists.
# A run with a summary.json and no snapshots is skipped: --resume would retrain it.
# A method and learner with no qualifying strength is skipped with an error and the
# step exits 3 after finishing the others; the method is reported as failing to learn.
# A selected path the calibrated c do not bracket gets no matched runs, matched exits 3 after
# the others, and verdict reports that comparison as undefined.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
LEARNERS=${LEARNERS:-"bp epc"}
METHODS=${METHODS:-"l2sp ewc"}
EXTRA_SCALES=${EXTRA_SCALES:-}
PRUNE=${PRUNE:-0}
LR=3e-5
PILOT_SEED=0
L2SP_GRID="1e-3 1e-2 1e-1 1"
EWC_GRID="1e0 1e2 1e4 1e6"
SCALES="0.25 0.5 0.75"
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000
)
BP=(--mode backprop --lr "$LR")
EPC=(
  --mode pc --pc_inference epc --pc_eta_infer 1e-4 --pc_infer_steps 5
  --pc_forward_updates none --pc_eta_control energy --pc_energy_target 0.17
  --pc_alignment_batches 2 --lr "$LR"
)
SKIPPED=0
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
  esac
}

learner_args() {
  case "$1" in
    bp) echo "${BP[*]}" ;;
    epc) echo "${EPC[*]}" ;;
  esac
}

grid_of() {
  case "$1" in
    l2sp) echo "$L2SP_GRID" ;;
    ewc) echo "$EWC_GRID" ;;
  esac
}

confirmation_seeds() {
  local seed out=""
  for seed in $SEEDS; do
    if [ "$seed" != "$PILOT_SEED" ]; then
      out="$out $seed"
    fi
  done
  echo "$out"
}

confirmation_flag() {
  local seeds
  seeds=$(confirmation_seeds)
  if [ -n "$seeds" ]; then
    echo "--confirmation_seeds $seeds"
  fi
}

prune_run() {
  if [ -f "$1/summary.json" ]; then
    rm -f "$1"/stage_*.npz "$1"/optimizer_state_stage_*.npz
  else
    echo "not pruning $1: no summary.json" >&2
  fi
}

run() {
  local learner=$1 name=$2 order=$3 seed=$4
  shift 4
  local out="$ROOT/ret_${learner}_${name}_${order}_seed${seed}"
  if [ -f "$out/summary.json" ] && [ ! -f "$out/stage_0.npz" ]; then
    echo "skipping $out: finished, snapshots pruned"
    return
  fi
  # shellcheck disable=SC2046
  "$PYTHON" examples/mlm_domain_forgetting.py "${COMMON[@]}" $(learner_args "$learner") \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee -a "$out.log"
  if [ "$PRUNE" = 1 ]; then
    prune_run "$out"
  fi
}

ungated_run() {
  case "$1" in
    bp) echo "g9_$2_lr${LR}_seed${PILOT_SEED}" ;;
    epc) echo "epc_$2_lr${LR}_seed${PILOT_SEED}" ;;
  esac
}

calibration_run() {
  local learner=$1 scale=$2 order=$3
  case "$learner:$scale" in
    bp:1) echo "infl_bp_none_${order}_seed0" ;;
    epc:1) echo "epcsuite_none_${order}_seed0" ;;
    *) echo "ret_${learner}_uniformall_s${scale}_${order}_seed0" ;;
  esac
}

check_calibration() {
  local learner=$1 order scale items=""
  for order in fwd rev; do
    for scale in $SCALES $EXTRA_SCALES 1; do
      items="$items $scale=$(calibration_run "$learner" "$scale" "$order")"
    done
  done
  # shellcheck disable=SC2086
  "$PYTHON" - "$ROOT" "$learner" "$LR" $items <<'PY'
import json
import math
import sys
from pathlib import Path

root, learner, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
hint = {
    "bp": "scripts/run_influence_gates.sh matched and the calibrate step",
    "epc": "scripts/run_epc_suite.sh gates_calibrate and the calibrate step",
}[learner]
mode, inference = {"bp": ("backprop", None), "epc": ("pc", "epc")}[learner]
for item in sys.argv[4:]:
    scale, run = item.split("=")
    path = Path(root) / run
    if not (path / "summary.json").is_file():
        sys.exit(f"{path} is not a finished run; produce it with {hint}")
    if not (path / "training_log_stage_3.jsonl").is_file():
        sys.exit(f"{path} has no training logs to read its path from")
    config = json.loads((path / "run_config.json").read_text())["config"]
    gate = config.get("gate", "none")
    matches = (
        math.isclose(float(config["lr"]), lr)
        and config.get("mode") == mode
        and (inference is None or config.get("pc_inference") == inference)
        and (
            gate == "none"
            if float(scale) == 1.0
            else gate == "uniform_all" and float(config["gate_scale"]) == float(scale)
        )
    )
    if not matches:
        sys.exit(f"{path} does not match {learner} lr {lr:g} uniform_all c = {scale}")
print(f"{learner}: {len(sys.argv) - 4} calibration runs finished and match lr {lr:g}")
PY
}

matched_scale() {
  local learner=$1 method=$2 strength=$3 order scale runs=""
  for order in fwd rev; do
    for scale in $SCALES $EXTRA_SCALES 1; do
      runs="$runs $scale=$(calibration_run "$learner" "$scale" "$order")"
    done
  done
  # shellcheck disable=SC2086
  "$PYTHON" - "$ROOT" "ret_${learner}_${method}_l${strength}" "$SEEDS" "$learner" $runs <<'PY'
import sys
from pathlib import Path

import numpy as np

from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, gated, seeds, learner = sys.argv[1:5]
seeds = seeds.split()


def path(run):
    if not (Path(root) / run / "summary.json").is_file():
        sys.exit(f"{root}/{run} is not a finished run; run its step first")
    return sum(s["total"] for s in read_paths(f"{root}/{run}", 3)[1:])


target = np.mean([path(f"{gated}_{o}_seed{s}") for o in ("fwd", "rev") for s in seeds])
by_scale = {}
for item in sys.argv[5:]:
    scale, run = item.split("=")
    by_scale.setdefault(float(scale), []).append(path(run))
scales = sorted(by_scale)
paths = [float(np.mean(by_scale[c])) for c in scales]
if np.any(np.diff(paths) <= 0):
    sys.exit(f"calibration paths {paths} do not increase with the scale")
if not paths[0] <= target <= paths[-1]:
    side = "below" if target < paths[0] else "above"
    print(
        f"ERROR {gated} path {target:.4g} is {side} the calibration {paths} at "
        f"c = {scales}; not extrapolating, no matched runs. To bracket it: "
        f"EXTRA_SCALES='<c>' LEARNERS={learner} scripts/run_retention_baselines.sh "
        f"calibrate with a c beyond the range, then rerun matched with the same "
        f"EXTRA_SCALES. Otherwise verdict reports this comparison as undefined.",
        file=sys.stderr,
    )
    sys.exit(4)
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

# Prints the matched label, or a placeholder that leaves the comparison undefined.
matched_label() {
  if [ -f "$1/run_config.json" ]; then
    label_of "$1"
  else
    echo "note: $1 is absent, its comparison is undefined" >&2
    echo "no_matched_control"
  fi
}

# Prints the grid token of the selected strength. Exit 3: nothing qualified.
selected_strength() {
  local learner=$1 method=$2
  # shellcheck disable=SC2046
  "$PYTHON" - "$ROOT/ret_selection_$learner/retention_selection.json" "$learner" "$method" \
    $(grid_of "$method") <<'PY'
import json
import sys
from pathlib import Path

path, learner, method, *grid = sys.argv[1:]
if not Path(path).is_file():
    sys.exit(f"{path} is missing: run the select step first")
entry = json.loads(Path(path).read_text())["methods"].get(method)
if entry is None:
    sys.exit(f"{path} has no {method}: run its pilot and the select step first")
if entry["selected"] is None:
    print(
        f"ERROR {method} {learner}: no qualifying strength, skipped; report it as "
        f"failing to learn, with no confirmation",
        file=sys.stderr,
    )
    sys.exit(3)
if entry["endpoint"]:
    print(
        f"note: {method} {learner} selected the grid endpoint {entry['selected']:g}",
        file=sys.stderr,
    )
tokens = [t for t in grid if float(t) == float(entry["selected"])]
if len(tokens) != 1:
    sys.exit(f"{method} {learner}: {entry['selected']} is not on the grid {grid}")
print(tokens[0])
PY
}

exit_if_skipped() {
  if [ "$SKIPPED" != 0 ]; then
    echo "step finished with $SKIPPED method-learner combinations skipped" >&2
    exit 3
  fi
}

case "${1:-}" in
  pilot)
    for learner in $LEARNERS; do
      for method in $METHODS; do
        for strength in $(grid_of "$method"); do
          for order in fwd rev; do
            run "$learner" "${method}_l${strength}" "$order" "$PILOT_SEED" \
              --retention "$method" --retention_lambda "$strength"
          done
        done
      done
    done
    ;;
  select)
    for learner in $LEARNERS; do
      dirs=()
      for order in fwd rev; do
        dirs+=("$ROOT/$(ungated_run "$learner" "$order")")
        for method in $METHODS; do
          for strength in $(grid_of "$method"); do
            dirs+=("$ROOT/ret_${learner}_${method}_l${strength}_${order}_seed${PILOT_SEED}")
          done
        done
      done
      if [ "$learner" = bp ]; then
        reference=(--reference_mode backprop)
      else
        reference=(--reference_mode pc --pc_inference epc)
      fi
      "$PYTHON" examples/domain_forgetting_analysis.py retention_select \
        --run_dirs "${dirs[@]}" "${reference[@]}" --seq_lr "$LR" --seed "$PILOT_SEED" \
        --output_dir "$ROOT/ret_selection_$learner" 2>&1 |
        tee -a "$ROOT/ret_selection_$learner.log"
    done
    ;;
  confirm)
    for learner in $LEARNERS; do
      for method in $METHODS; do
        strength=$(selected_strength "$learner" "$method") || {
          code=$?
          if [ "$code" != 3 ]; then exit "$code"; fi
          SKIPPED=$((SKIPPED + 1))
          continue
        }
        for order in fwd rev; do
          for seed in $(confirmation_seeds); do
            run "$learner" "${method}_l${strength}" "$order" "$seed" \
              --retention "$method" --retention_lambda "$strength"
          done
        done
      done
    done
    exit_if_skipped
    ;;
  calibrate)
    for learner in $LEARNERS; do
      for c in $SCALES $EXTRA_SCALES; do
        for order in fwd rev; do
          run "$learner" "uniformall_s$c" "$order" 0 --gate uniform_all --gate_scale "$c"
        done
      done
      check_calibration "$learner"
    done
    ;;
  matched)
    for learner in $LEARNERS; do
      check_calibration "$learner"
      for method in $METHODS; do
        strength=$(selected_strength "$learner" "$method") || {
          code=$?
          if [ "$code" != 3 ]; then exit "$code"; fi
          SKIPPED=$((SKIPPED + 1))
          continue
        }
        c=$(matched_scale "$learner" "$method" "$strength") || {
          code=$?
          if [ "$code" != 4 ]; then exit "$code"; fi
          SKIPPED=$((SKIPPED + 1))
          continue
        }
        echo "$learner $method lambda=$strength matched gate_scale=$c"
        for order in fwd rev; do
          for seed in $SEEDS; do
            run "$learner" "${method}_matched" "$order" "$seed" --gate uniform_all --gate_scale "$c"
          done
        done
      done
    done
    exit_if_skipped
    ;;
  restart)
    for learner in $LEARNERS; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          run "$learner" keepopt "$order" "$seed" --keep_optimizer_state
        done
      done
    done
    ;;
  verdict)
    shopt -s nullglob
    for learner in $LEARNERS; do
      if [ "$learner" = bp ]; then
        curve=(
          "$ROOT"/g3_bp_t5000000_lr1e-4 "$ROOT"/g4_bp_seed? "$ROOT"/g8_rev_bp_seed?
          "$ROOT"/g9_*_seed? "$ROOT"/g8_joint_seed?
        )
        reference=(--reference_mode backprop)
      else
        curve=(
          "$ROOT"/epcsuite_curve_*_seed? "$ROOT"/epc_fwd_lr${LR}_seed?
          "$ROOT"/epc_rev_lr${LR}_seed?
        )
        reference=(--reference_mode pc --pc_inference epc --map_source learner)
      fi
      if [ "${#curve[@]}" -eq 0 ]; then
        echo "no $learner reference curve runs under $ROOT" >&2
        exit 1
      fi
      for method in $METHODS; do
        strength=$(selected_strength "$learner" "$method") || {
          code=$?
          if [ "$code" != 3 ]; then exit "$code"; fi
          SKIPPED=$((SKIPPED + 1))
          continue
        }
        name="ret_${learner}_${method}_l${strength}"
        control="ret_${learner}_${method}_matched"
        gate=$(label_of "$ROOT/${name}_fwd_seed${PILOT_SEED}")
        matched=$(matched_label "$ROOT/${control}_fwd_seed${PILOT_SEED}")
        # shellcheck disable=SC2046,SC2086
        "$PYTHON" examples/domain_forgetting_analysis.py frontier \
          --run_dirs "$ROOT/${name}"_{fwd,rev}_seed? "$ROOT/${control}"_{fwd,rev}_seed? \
          "${curve[@]}" "${reference[@]}" --seq_lr "$LR" --expected_seeds $SEEDS \
          --baseline_verdict "$gate" "$matched" \
          $(confirmation_flag) \
          --output_dir "$ROOT/ret_frontier_${learner}_${method}" 2>&1 |
          tee -a "$ROOT/ret_frontier_${learner}_${method}.log"
      done
      keep=$(label_of "$ROOT/ret_${learner}_keepopt_fwd_seed${PILOT_SEED}")
      restarted=$(label_of "$ROOT/$(ungated_run "$learner" fwd)")
      # shellcheck disable=SC2046,SC2086
      "$PYTHON" examples/domain_forgetting_analysis.py frontier \
        --run_dirs "$ROOT/ret_${learner}_keepopt"_{fwd,rev}_seed? "${curve[@]}" \
        "${reference[@]}" --seq_lr "$LR" --expected_seeds $SEEDS \
        --descriptive_baseline_verdict "$keep" "$restarted" \
        $(confirmation_flag) \
        --output_dir "$ROOT/ret_frontier_${learner}_restart" 2>&1 |
        tee -a "$ROOT/ret_frontier_${learner}_restart.log"
    done
    exit_if_skipped
    ;;
  prune)
    shopt -s nullglob
    for learner in $LEARNERS; do
      for out in "$ROOT/ret_${learner}"_*_{fwd,rev}_seed?; do
        prune_run "$out"
      done
    done
    ;;
  *)
    sed -n 2,26p "$0"
    exit 1
    ;;
esac
