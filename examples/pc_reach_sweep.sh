#!/usr/bin/env bash
# PC BERT inference-reach sweep: infer steps x inference rate x seed, plus a BP
# baseline per seed. Every run uses the same short schedule and a capped
# validation subset; results land in $OUT/summary.csv.
#
# Usage (from the repo root, on the GPU host):
#   DATA_ARGS="--data_source hf --dataset_name deven367/babylm-10M ..." \
#     bash examples/pc_reach_sweep.sh
set -euo pipefail

OUT=${OUT:-results/pc_reach_sweep}
STEPS_GRID=${STEPS_GRID:-"5 13 20"}
ETA_GRID=${ETA_GRID:-"0.01 0.05 0.1"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_STEPS=${MAX_STEPS:-2000}
MAX_VAL_SEQS=${MAX_VAL_SEQS:-4096}
DATA_ARGS=${DATA_ARGS:?set DATA_ARGS to the dataset/cache flags used for the 20k runs}
PYTHON=${PYTHON:-python}
EXTRA_ARGS=${EXTRA_ARGS:-}

COMMON_ARGS="--stage pretrain $DATA_ARGS \
  --depth 4 --embed_dim 256 --num_heads 4 --mlp_dim 1024 --seq_len 128 \
  --batch_size 8 --max_steps $MAX_STEPS --lr 3e-4 --warmup_steps 600 \
  --weight_decay 0.1 --mask_prob 0.15 --max_validation_sequences $MAX_VAL_SEQS \
  --validation_interval 0 --checkpoint_interval $MAX_STEPS"

mkdir -p "$OUT/logs"

run() {
  local name=$1
  shift
  local log="$OUT/logs/$name.log"
  if grep -q "^done stage=pretrain" "$log" 2>/dev/null; then
    echo "skip $name (done)"
    return
  fi
  echo "run  $name"
  local start end
  start=$(date +%s)
  PYTHONPATH=. $PYTHON examples/bert_demo.py $COMMON_ARGS --checkpoint "" "$@" \
    $EXTRA_ARGS >"$log" 2>&1 || echo "run failed: see $log"
  end=$(date +%s)
  echo "wall_seconds=$((end - start))" >>"$log"
}

for seed in $SEEDS; do
  run "bp_seed${seed}" --mode backprop --seed "$seed"
  for steps in $STEPS_GRID; do
    for eta in $ETA_GRID; do
      run "pc_steps${steps}_eta${eta}_seed${seed}" --mode pc --seed "$seed" \
        --pc_infer_steps "$steps" --pc_eta_infer "$eta"
    done
  done
done

summary="$OUT/summary.csv"
echo "run,mode,infer_steps,eta_infer,seed,val_masked_loss,val_masked_acc,val_masked_tokens,wall_seconds,zero_grad_nodes,token_embeddings_grad_norm" >"$summary"
for log in "$OUT"/logs/*.log; do
  name=$(basename "$log" .log)
  if [[ $name == bp_* ]]; then
    mode=backprop steps="" eta=""
    seed=${name#bp_seed}
  else
    mode=pc
    steps=$(sed -E 's/^pc_steps([0-9]+)_.*/\1/' <<<"$name")
    eta=$(sed -E 's/^.*_eta([0-9.]+)_seed.*/\1/' <<<"$name")
    seed=$(sed -E 's/^.*_seed([0-9]+)$/\1/' <<<"$name")
  fi
  final=$(grep "^final_validation_masked_loss=" "$log" | tail -1 || true)
  loss=$(sed -nE 's/.*final_validation_masked_loss=([^ ]+).*/\1/p' <<<"$final")
  acc=$(sed -nE 's/.*final_validation_masked_accuracy=([^ ]+).*/\1/p' <<<"$final")
  tokens=$(sed -nE 's/.*final_validation_masked_tokens=([^ ]+).*/\1/p' <<<"$final")
  wall=$(sed -nE 's/^wall_seconds=([0-9]+)/\1/p' "$log" | tail -1)
  norms=$(grep "^pc_grad_norms " "$log" | head -1 || true)
  zeros=""
  tok=""
  if [[ -n $norms ]]; then
    zeros=$(tr ' ' '\n' <<<"$norms" | grep -c '=0\.000e+00$' || true)
    tok=$(sed -nE 's/.*token_embeddings=([^ ]+).*/\1/p' <<<"$norms")
  fi
  echo "$name,$mode,$steps,$eta,$seed,$loss,$acc,$tokens,$wall,$zeros,$tok" >>"$summary"
done
echo "summary: $summary"
