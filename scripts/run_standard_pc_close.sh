#!/usr/bin/env bash
# Standard PC / μPC close-out: one full news stage (616 updates, lr 3e-5, seed 0).
# Setting: η=1e-3, T=20, the best mean hidden-module cosine with BP (0.152) of the
# pretrained-checkpoint probe in results/spc_probe_dl/spc_probe. Probed modes: off and
# jacobian only; full (W / a load, μPC's optimizer) is run at this unprobed setting.
# Usage: scripts/run_standard_pc_close.sh {run|gonogo}
#   run      one news stage per μPC mode in $MUPC (default "off jacobian full"),
#            alignment at stage start and end (3 runs)
#   gonogo   per run: no divergence (finite losses, final news CE not above the
#            stage-0 CE, energy rose in at most half the updates), news CE gain
#            >= 0.42 nats, end mean hidden alignment >= 0.5; GO only if all hold
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT=${ROOT:-results/domain_forgetting}
MUPC=${MUPC:-"off jacobian full"}
PYTHON=${PYTHON:-python}
MIN_GAIN=0.42
MIN_ALIGNMENT=0.5
COMMON=(
  --checkpoint checkpoints/bert_mini_128.json
  --data_dir data/domain_forgetting
  --tokens_per_domain 5000000 --lr 3e-5 --domains news --seed 0
)
STANDARD=(
  --mode pc --pc_inference standard --pc_eta_infer 1e-3 --pc_infer_steps 20
  --pc_forward_updates none --pc_alignment_batches 2
)
export PYTHONPATH=.

case "${1:-}" in
  run)
    for mupc in $MUPC; do
      out="$ROOT/spcclose_${mupc}_news_seed0"
      "$PYTHON" examples/mlm_domain_forgetting.py "${COMMON[@]}" "${STANDARD[@]}" \
        --pc_mupc "$mupc" --output_dir "$out" --resume 2>&1 | tee -a "$out.log"
    done
    ;;
  gonogo)
    "$PYTHON" - "$ROOT" "$MIN_GAIN" "$MIN_ALIGNMENT" $MUPC <<'EOF'
import json
import math
import sys
from pathlib import Path

root, min_gain, min_alignment = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
verdicts = []
for mupc in sys.argv[4:]:
    path = root / f"spcclose_{mupc}_news_seed0"
    if not (path / "summary.json").is_file():
        print(f"{mupc}: MISSING {path}")
        verdicts.append(False)
        continue
    summary = json.loads((path / "summary.json").read_text())
    log = [
        json.loads(line)
        for line in (path / "training_log_stage_1.jsonl").read_text().splitlines()
        if line
    ]
    losses = [float(entry["masked_loss"]) for entry in log]
    final = float(summary["final_ce"]["news"])
    gain = float(summary["learning_gain_ce"]["news"])
    start_ce = float(
        json.loads((path / "masked_ce_matrix.json").read_text())["matrix"][0][0]
    )
    rises = [bool(entry["pc_energy_increased"]) for entry in log]
    rise_fraction = sum(rises) / len(rises) if rises else math.nan
    finite = all(math.isfinite(v) for v in (*losses, final, gain))
    stable = finite and final <= start_ce and rise_fraction <= 0.5
    alignment = [
        json.loads(line)
        for line in (path / "alignment.jsonl").read_text().splitlines()
        if line
    ]
    end = [r for r in alignment if r["stage"] == 1 and r["when"] == "end"]
    start = [r for r in alignment if r["stage"] == 1 and r["when"] == "start"]
    cosine = float(end[0]["mean_cosine_hidden"]) if end else math.nan
    checks = {
        f"no divergence (finite {finite}, final CE {final:.4f} <= stage-0 CE "
        f"{start_ce:.4f}, energy rose in {rise_fraction:.1%} <= 50% of updates)": stable,
        f"news CE gain {gain:.4f} >= {min_gain:g}": finite and gain >= min_gain,
        f"end mean hidden alignment {cosine:.3f} >= {min_alignment:g}": (
            math.isfinite(cosine) and cosine >= min_alignment
        ),
    }
    go = all(checks.values())
    verdicts.append(go)
    begin = float(start[0]["mean_cosine_hidden"]) if start else math.nan
    print(
        f"{mupc}: updates {len(log)}, final news CE {final:.4f}, gain {gain:.4f}, "
        f"hidden alignment start {begin:.3f} end {cosine:.3f}"
    )
    for name, ok in checks.items():
        print(f"    {'pass' if ok else 'FAIL'}  {name}")
    print(f"    {'GO' if go else 'NO-GO'}")
print(f"standard PC track: {'GO' if any(verdicts) else 'NO-GO (close)'}")
EOF
    ;;
  *)
    sed -n 2,11p "$0"
    exit 1
    ;;
esac
