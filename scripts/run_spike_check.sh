#!/usr/bin/env bash
# Spike-only PC equivalence with BP on BERT (news/reviews/wiki), the g4b_pc flags.
# Usage: scripts/run_spike_check.sh {sequential|map|online|commutator|all|equivalence}
#   sequential   lr 1e-4, both orders, seeds 0-2 (5 runs; fwd seed 0 reuses
#                g4b_pc_seed0 when its identity matches, else 6)
#   map          static map gate β=2 on the BP map and on the learner (spike-only)
#                map, both orders, seed 0 (4 runs)
#   online       online_protect β=2 (refreshed every 50 updates), both orders, seed 0 (2 runs)
#   commutator   news/wiki 2-stage, ungated and map β=2, both orders, seed 0 (4 runs)
#   all          sequential, map, online and commutator (15-16 runs)
#   equivalence  per-domain |ΔCE| and |ΔBWT| of each spike run against its BP
#                counterpart (both map sources against the BP map run); PASS iff
#                all <= 1e-3
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
TOLERANCE=1e-3
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000 --lr 1e-4
)
SPIKE=(--mode pc --pc_inference spiking --pc_eta_infer 0 --pc_forward_updates initial)
MAP=(--gate map --gate_beta 2 --gate_map_windows 256)
PROTECT=(--gate online_protect --gate_beta 2 --gate_interval 50 --gate_ema 0.3 --gate_map_windows 256)
LEARNER=(--gate_map_source learner)
REUSED=g4b_pc_seed0
export PYTHONPATH=.

order_args() {
  case "$1" in
    fwd) echo "news reviews wiki" ;;
    rev) echo "wiki reviews news" ;;
    pair_fwd) echo "news wiki" ;;
    pair_rev) echo "wiki news" ;;
  esac
}

run() {
  local name=$1 order=$2 seed=$3
  shift 3
  local out="$ROOT/spike_${name}_${order}_seed${seed}"
  # shellcheck disable=SC2046
  "$PYTHON" examples/mlm_domain_forgetting.py "${COMMON[@]}" "${SPIKE[@]}" \
    --domains $(order_args "$order") --seed "$seed" \
    --output_dir "$out" --resume "$@" 2>&1 | tee -a "$out.log"
}

reuse_matches() {
  # shellcheck disable=SC2046
  "$PYTHON" - "$ROOT/$REUSED" "${COMMON[@]}" "${SPIKE[@]}" \
    --domains $(order_args fwd) --seed 0 --output_dir unused <<'EOF'
import json
import sys
from pathlib import Path

from examples.mlm_domain_forgetting import build_parser, config_from_args

path = Path(sys.argv[1]) / "run_config.json"
if not path.is_file():
    print("no")
    sys.exit(0)
config = config_from_args(build_parser().parse_args(sys.argv[2:]))
saved = json.loads(path.read_text())["config"]
print("yes" if json.loads(json.dumps(config.identity())) == saved else "no")
EOF
}

sequential() {
  for order in fwd rev; do
    for seed in $SEEDS; do
      if [ "$order" = fwd ] && [ "$seed" = 0 ] && [ "$(reuse_matches)" = yes ]; then
        echo "fwd seed 0: reusing $ROOT/$REUSED (identity matches)"
        continue
      fi
      run seq "$order" "$seed"
    done
  done
}

case "${1:-}" in
  sequential) sequential ;;
  map)
    for order in fwd rev; do
      run map_b2 "$order" 0 "${MAP[@]}"
      run map_b2_learner "$order" 0 "${MAP[@]}" "${LEARNER[@]}"
    done
    ;;
  online)
    for order in fwd rev; do
      run online_protect "$order" 0 "${PROTECT[@]}"
    done
    ;;
  commutator)
    for order in fwd rev; do
      run commutator_none "pair_$order" 0
      run commutator_map_b2 "pair_$order" 0 "${MAP[@]}"
    done
    ;;
  all)
    sequential
    for order in fwd rev; do
      run map_b2 "$order" 0 "${MAP[@]}"
      run map_b2_learner "$order" 0 "${MAP[@]}" "${LEARNER[@]}"
      run online_protect "$order" 0 "${PROTECT[@]}"
      run commutator_none "pair_$order" 0
      run commutator_map_b2 "pair_$order" 0 "${MAP[@]}"
    done
    ;;
  equivalence)
    reuse=$(reuse_matches)
    "$PYTHON" - "$ROOT" "$TOLERANCE" "$reuse" "$SEEDS" <<'EOF'
import json
import math
import sys
from pathlib import Path

from fabricpc.experiments.domain_forgetting.runner import check_comparable_runs

root, tolerance, reuse, seeds = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4].split()
bp_sequential = {
    ("fwd", "0"): "g3_bp_t5000000_lr1e-4",
    ("fwd", "1"): "g4_bp_seed1",
    ("fwd", "2"): "g4_bp_seed2",
    **{("rev", s): f"g8_rev_bp_seed{s}" for s in ("0", "1", "2")},
}
pairs = []
for (order, seed), bp in bp_sequential.items():
    if seed not in seeds:
        continue
    spike = f"spike_seq_{order}_seed{seed}"
    if (order, seed) == ("fwd", "0") and reuse == "yes":
        spike = "g4b_pc_seed0"
    pairs.append(("sequential", spike, bp))
for order in ("fwd", "rev"):
    pairs.append(("map", f"spike_map_b2_{order}_seed0", f"g10_{order}_map_b2_seed0"))
    pairs.append(
        ("map_learner", f"spike_map_b2_learner_{order}_seed0", f"g10_{order}_map_b2_seed0")
    )
    pairs.append(
        ("online", f"spike_online_protect_{order}_seed0", f"online_protect_{order}_seed0")
    )
    for gate in ("none", "map_b2"):
        pairs.append(
            (
                "commutator",
                f"spike_commutator_{gate}_pair_{order}_seed0",
                f"commutator_{gate}_{order}_seed0",
            )
        )


def load(run):
    path = root / run
    if not (path / "summary.json").is_file():
        return None
    return (
        json.loads((path / "run_config.json").read_text()),
        json.loads((path / "summary.json").read_text()),
    )


families: dict[str, list[bool]] = {}
print(f"{'family':<11} {'spike run':<40} {'bp run':<32} {'max|dCE|':>10} {'|dBWT|':>10}  verdict")
for family, spike, bp in pairs:
    left, right = load(spike), load(bp)
    if left is None or right is None:
        missing = spike if left is None else bp
        print(f"{family:<11} {spike:<40} {bp:<32} {'-':>10} {'-':>10}  MISSING {missing}")
        families.setdefault(family, []).append(False)
        continue
    spike_config = dict(left[0], config=dict(left[0]["config"]))
    source = spike_config["config"].pop("gate_map_source", "bp")
    wanted = "learner" if family == "map_learner" else "bp"
    problems = [] if source == wanted else [f"gate_map_source is {source!r}"]
    problems += check_comparable_runs(spike_config, right[0])
    if left[0]["config"]["mode"] != "pc" or right[0]["config"]["mode"] != "backprop":
        problems.append("modes are not (pc, backprop)")
    domains = right[1]["domains"]
    deltas = {d: abs(left[1]["final_ce"][d] - right[1]["final_ce"][d]) for d in domains}
    bwt = abs(left[1]["bwt_ce"] - right[1]["bwt_ce"])
    finite = all(math.isfinite(v) for v in (*deltas.values(), bwt))
    passed = not problems and finite and bwt <= tolerance and max(deltas.values()) <= tolerance
    families.setdefault(family, []).append(passed)
    verdict = "PASS" if passed else "FAIL" + (f" ({'; '.join(problems)})" if problems else "")
    print(
        f"{family:<11} {spike:<40} {bp:<32} {max(deltas.values()):>10.2e} {bwt:>10.2e}  {verdict}"
    )
    print("    " + "  ".join(f"{d}: {v:.2e}" for d, v in deltas.items()))
print()
for family, results in families.items():
    print(f"{family}: {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
everything = all(all(r) for r in families.values())
print(f"equivalence (all |dCE|, |dBWT| <= {tolerance:g}): {'PASS' if everything else 'FAIL'}")
sys.exit(0 if everything else 1)
EOF
    ;;
  *)
    sed -n 2,13p "$0"
    exit 1
    ;;
esac
