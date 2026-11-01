#!/usr/bin/env bash
# Online per-step gates on BERT (news/reviews/wiki), BP mode, 3 stages.
# Usage: scripts/run_online_gate.sh {live|replay|calibrate|matched|tier2 <rule>|frontier}
#   live       online_protect and online_surprise, both orders, seeds 0-2 (12 runs)
#   replay     each rule, seed s replays seed (s+1) mod 3 of its order (12 runs)
#   calibrate  uniform c in {0.25,0.5,0.75} and none (c = 1; the g3/g4/g8 lr 1e-4
#              runs log no update_norm), both orders, seed 0 (8 runs)
#   matched    per rule one uniform c* (mean stage 2-3 path over both orders and
#              seeds matched to the live runs), both orders, seeds 0-2 (12 runs)
#   tier2      type, shuffle (gate_seed 0) and inverse variants of one rule
#              (protect|surprise), both orders, seeds 0-2 (18 runs)
#   frontier   frontier.json over the online, matched and reference runs
set -euo pipefail

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --mode backprop --tokens_per_domain 5000000 --lr 1e-4
)
ONLINE=(--gate_interval 50 --gate_ema 0.3 --gate_map_windows 256)
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
  esac
}

rule_args() {
  case "$1" in
    protect) echo "--gate online_protect --gate_beta 2" ;;
    surprise) echo "--gate online_surprise" ;;
    *)
      echo "unknown rule $1 (protect|surprise)" >&2
      exit 1
      ;;
  esac
}

run() {
  local label=$1 order=$2 seed=$3
  shift 3
  local out="$ROOT/online_${label}_${order}_seed${seed}"
  # shellcheck disable=SC2046
  python examples/mlm_domain_forgetting.py "${COMMON[@]}" \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee "$out.log"
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
  python - "$ROOT" "$1" "$SEEDS" <<'EOF'
import sys
import numpy as np
from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, rule, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
orders = ("fwd", "rev")


def path(run):
    stages = read_paths(f"{root}/{run}", 3)
    return stages[1]["total"] + stages[2]["total"]


target = np.mean([path(f"online_{rule}_{o}_seed{s}") for o in orders for s in seeds])
scales = [0.25, 0.5, 0.75, 1.0]
runs = {c: f"online_uniform_s{c:g}" for c in scales[:-1]} | {1.0: "online_none"}
paths = [np.mean([path(f"{runs[c]}_{o}_seed0") for o in orders]) for c in scales]
if np.any(np.diff(paths) <= 0):
    sys.exit(f"calibration paths {paths} do not increase with the scale")
if not paths[0] <= target <= paths[-1]:
    sys.exit(f"{rule} stage 2-3 path {target:.4g} outside calibration {paths}")
print(f"{float(np.interp(target, paths, scales)):.4f}")
EOF
}

case "${1:-}" in
  live)
    for rule in protect surprise; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          # shellcheck disable=SC2046
          run "$rule" "$order" "$seed" $(rule_args "$rule") "${ONLINE[@]}"
        done
      done
    done
    ;;
  replay)
    for rule in protect surprise; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          source_seed=$(next_seed "$seed")
          # shellcheck disable=SC2046
          run "${rule}_replay" "$order" "$seed" $(rule_args "$rule") "${ONLINE[@]}" \
            --gate_variant replay \
            --gate_source "$ROOT/online_${rule}_${order}_seed${source_seed}"
        done
      done
    done
    ;;
  calibrate)
    for order in fwd rev; do
      run none "$order" 0
      for c in 0.25 0.5 0.75; do
        run "uniform_s$c" "$order" 0 --gate uniform --gate_scale "$c"
      done
    done
    ;;
  matched)
    for rule in protect surprise; do
      c=$(matched_scale "$rule")
      echo "$rule matched gate_scale=$c"
      for order in fwd rev; do
        for seed in $SEEDS; do
          run "matched_${rule}" "$order" "$seed" --gate uniform --gate_scale "$c"
        done
      done
    done
    ;;
  tier2)
    rule=${2:-}
    rule_args "$rule" >/dev/null
    for variant in type shuffle inverse; do
      for order in fwd rev; do
        for seed in $SEEDS; do
          extra=(--gate_variant "$variant")
          if [ "$variant" = shuffle ]; then
            extra+=(--gate_seed 0)
          fi
          # shellcheck disable=SC2046
          run "${rule}_${variant}" "$order" "$seed" $(rule_args "$rule") \
            "${ONLINE[@]}" "${extra[@]}"
        done
      done
    done
    ;;
  frontier)
    shopt -s nullglob
    python examples/domain_forgetting_analysis.py frontier \
      --run_dirs \
        "$ROOT"/online_protect*_seed? "$ROOT"/online_surprise*_seed? \
        "$ROOT"/online_matched_*_seed? \
        "$ROOT"/g3_bp_t5000000_lr1e-4 "$ROOT"/g4_bp_seed? "$ROOT"/g8_rev_bp_seed? \
        "$ROOT"/g9_*_seed? "$ROOT"/g8_joint_seed? \
      --output_dir "$ROOT/frontier_online"
    ;;
  *)
    sed -n 2,12p "$0"
    exit 1
    ;;
esac
