#!/usr/bin/env bash
# Domain-order commutator on BERT (news/wiki), BP mode, gate-run settings.
# Usage: scripts/run_commutator.sh {main|calibrate|matched|analyze}
#   main       none + map β=2, both orders, seeds 0-2 (12 runs)
#   calibrate  uniform c in {0.25,0.5,0.75}, both orders, seed 0 (6 runs)
#   matched    one uniform c* (mean stage-2 path over both orders matched to map),
#              both orders, seeds 0-2 (6 runs)
#   online     online_protect β=2 (refreshed every 50 updates), both orders, seeds 0-2 (6 runs)
#   online_matched  one uniform c* matched to the online gate's stage-2 path, seeds 0-2 (6 runs)
#   analyze    commutator.json over none, map, online and both matched runs
set -euo pipefail

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --mode backprop --tokens_per_domain 5000000 --lr 1e-4
)
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news wiki" ;;
    rev) echo "wiki news" ;;
  esac
}

run() {
  local name=$1 order=$2 seed=$3
  shift 3
  local out="$ROOT/commutator_${name}_${order}_seed${seed}"
  # shellcheck disable=SC2046
  python examples/mlm_domain_forgetting.py "${COMMON[@]}" \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee "$out.log"
}

matched_scale() {
  python - "$ROOT" "$1" <<'EOF'
import sys
import numpy as np
from fabricpc.experiments.domain_forgetting.commutator import read_paths

root, gated = sys.argv[1], sys.argv[2]
orders = ("fwd", "rev")


def stage2(run):
    return read_paths(f"{root}/{run}", 2)[1]["total"]


target = np.mean([stage2(f"commutator_{gated}_{o}_seed{s}") for o in orders for s in (0, 1, 2)])
scales = [0.25, 0.5, 0.75, 1.0]
runs = {c: f"commutator_uniform_s{c:g}" for c in scales[:-1]} | {1.0: "commutator_none"}
paths = [np.mean([stage2(f"{runs[c]}_{o}_seed0") for o in orders]) for c in scales]
if not paths[0] <= target <= paths[-1]:
    sys.exit(f"{gated} stage-2 path {target:.4g} outside calibration {paths}")
print(f"{float(np.interp(target, paths, scales)):.4f}")
EOF
}

case "${1:-}" in
  main)
    for order in fwd rev; do
      for seed in $SEEDS; do
        run none "$order" "$seed"
        run map_b2 "$order" "$seed" --gate map --gate_beta 2 --gate_map_windows 256
      done
    done
    ;;
  calibrate)
    for order in fwd rev; do
      for c in 0.25 0.5 0.75; do
        run "uniform_s$c" "$order" 0 --gate uniform --gate_scale "$c"
      done
    done
    ;;
  matched)
    c=$(matched_scale map_b2)
    echo "matched gate_scale=$c"
    for order in fwd rev; do
      for seed in $SEEDS; do
        run matched "$order" "$seed" --gate uniform --gate_scale "$c"
      done
    done
    ;;
  online)
    for order in fwd rev; do
      for seed in $SEEDS; do
        run online_protect "$order" "$seed" --gate online_protect --gate_beta 2 \
          --gate_interval 50 --gate_ema 0.3 --gate_map_windows 256
      done
    done
    ;;
  online_matched)
    c=$(matched_scale online_protect)
    echo "online matched gate_scale=$c"
    for order in fwd rev; do
      for seed in $SEEDS; do
        run matched_online "$order" "$seed" --gate uniform --gate_scale "$c"
      done
    done
    ;;
  analyze)
    shopt -s nullglob
    python examples/domain_forgetting_analysis.py commutator \
      --run_dirs "$ROOT"/commutator_none_*_seed? "$ROOT"/commutator_map_b2_*_seed? "$ROOT"/commutator_matched_*_seed? \
        "$ROOT"/commutator_online_protect_*_seed? "$ROOT"/commutator_matched_online_*_seed? \
      --output_dir "$ROOT/commutator_bert"
    ;;
  *)
    sed -n 2,10p "$0"
    exit 1
    ;;
esac
