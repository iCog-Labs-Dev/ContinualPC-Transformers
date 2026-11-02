#!/usr/bin/env bash
# Interventions on ePC's self-sharpening on BERT (news/reviews/wiki), from stage 1.
# Usage: scripts/run_epc_intervention.sh {calibrate|pilot|benchmark <setting>|bp <setting>|rollback <setting>}
#   calibrate  r1 of ePC and BP at the checkpoint (news, 4 batches of 64), to
#              $ROOT/epcint_calibration.json
#   pilot      one news stage, ePC, seed 0, settings penr0.05 penr0.2 scale0.5 scale0.25
#   benchmark  ePC with one setting, both orders, seeds 0-2 (6 runs)
#   bp         BP (lr 3e-5, the g9 flags, regime probe) with one setting, both
#              orders, seeds 0-2 (6 runs)
#   rollback   rollback.json for the 6 ePC benchmark runs of one setting
# Settings: penr<r> (--pc_error_norm_penalty r / r1 of the mode, from calibrate),
#   pen<gamma> (--pc_error_norm_penalty gamma), scale<kappa> (--update_scale
#   kappa on embeddings, L0_mha and L1_mha).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
PILOT_SETTINGS=(penr0.05 penr0.2 scale0.5 scale0.25)
CALIBRATION="$ROOT/epcint_calibration.json"
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000 --lr 3e-5
)
EPC=(
  --mode pc --pc_inference epc --pc_eta_infer 1e-4 --pc_infer_steps 5
  --pc_forward_updates none --pc_eta_control energy --pc_energy_target 0.17
  --pc_alignment_batches 2 --pc_regime_interval 100 --pc_regime_final
)
BP=(
  --mode backprop
  --pc_regime_interval 100 --pc_regime_eta 1e-4 --pc_regime_steps 5 --pc_regime_final
)
SCALED_MODULES=(embeddings L0_mha L1_mha)
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
    news) echo "news" ;;
  esac
}

penalty_for_ratio() {
  python - "$CALIBRATION" "$1" "$2" <<'EOF'
import json
import sys
from pathlib import Path

path, mode, ratio = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
if not path.is_file():
    sys.exit(f"{path} is missing; run 'scripts/run_epc_intervention.sh calibrate' first")
calibration = json.loads(path.read_text())
if mode not in calibration:
    sys.exit(f"{path} has no {mode} calibration")
print(f"{ratio / calibration[mode]['r1']:.6g}")
EOF
}

setting_args() {
  local setting=$1 mode=$2 gamma
  case "$setting" in
    penr*)
      gamma=$(penalty_for_ratio "$mode" "${setting#penr}") || exit 1
      echo "--pc_error_norm_penalty $gamma"
      ;;
    pen*) echo "--pc_error_norm_penalty ${setting#pen}" ;;
    scale*) echo "--update_scale ${setting#scale} --update_scale_modules ${SCALED_MODULES[*]}" ;;
    *)
      echo "unknown setting $setting (penr<r>|pen<gamma>|scale<kappa>)" >&2
      exit 1
      ;;
  esac
}

run() {
  local name=$1 order=$2 seed=$3
  shift 3
  local out="$ROOT/${name}_${order}_seed${seed}"
  # shellcheck disable=SC2046
  python examples/mlm_domain_forgetting.py "${COMMON[@]}" \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee -a "$out.log"
}

require_setting() {
  case "${1:-}" in
    penr* | pen* | scale*) ;;
    *)
      sed -n 2,13p "$0"
      exit 1
      ;;
  esac
}

calibrate() {
  local mode=$1
  shift
  python examples/mlm_domain_forgetting.py calibrate_penalty "${COMMON[@]}" \
    --domain news --batches 4 --batch_size 64 --seed 0 \
    --out "$ROOT/epcint_calibration_${mode}.json" "$@" 2>&1 |
    tee -a "$ROOT/epcint_calibration_${mode}.log"
}

case "${1:-}" in
  calibrate)
    mkdir -p "$ROOT"
    calibrate pc "${EPC[@]}"
    calibrate backprop --mode backprop
    python - "$ROOT" "$CALIBRATION" <<'EOF'
import json
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
merged = {
    mode: json.loads((root / f"epcint_calibration_{mode}.json").read_text())
    for mode in ("pc", "backprop")
}
out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
print({mode: entry["r1"] for mode, entry in merged.items()})
EOF
    ;;
  pilot)
    for setting in "${PILOT_SETTINGS[@]}"; do
      args=$(setting_args "$setting" pc)
      # shellcheck disable=SC2086
      run "epcint_pilot_${setting}" news 0 "${EPC[@]}" $args
    done
    ;;
  benchmark)
    require_setting "${2:-}"
    args=$(setting_args "$2" pc)
    for order in fwd rev; do
      for seed in $SEEDS; do
        # shellcheck disable=SC2086
        run "epcint_$2" "$order" "$seed" "${EPC[@]}" $args
      done
    done
    ;;
  bp)
    require_setting "${2:-}"
    args=$(setting_args "$2" backprop)
    for order in fwd rev; do
      for seed in $SEEDS; do
        # shellcheck disable=SC2086
        run "epcint_bp_$2" "$order" "$seed" "${BP[@]}" $args
      done
    done
    ;;
  rollback)
    require_setting "${2:-}"
    for order in fwd rev; do
      for seed in $SEEDS; do
        out="$ROOT/epcint_$2_${order}_seed${seed}"
        python examples/domain_forgetting_analysis.py rollback --run_dir "$out" \
          2>&1 | tee -a "$out.rollback.log"
      done
    done
    ;;
  *)
    sed -n 2,13p "$0"
    exit 1
    ;;
esac
