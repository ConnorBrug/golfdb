#!/bin/bash
# 4-split cross-validation driver. Trains each split, evals EMA checkpoints with TTA,
# prints a final per-split table and 4-split average. Designed for ~8-9 hrs total.
#
# Usage:
#   tmux new -s train
#   cd ~/golfdb && source .venv/bin/activate
#   bash run_4split.sh
#   Ctrl+B then D to detach

set -u  # treat unset vars as errors, but allow failed commands so one split doesnt kill the sweep

ITERATIONS=${ITERATIONS:-8000}
BATCH=${BATCH:-22}
SEQ=${SEQ:-64}
RUN_TAG=${RUN_TAG:-max_$(date +%Y%m%d_%H%M%S)}

RESULTS_FILE="results_${RUN_TAG}.txt"
echo "==== GolfDB 4-split sweep ====" | tee "$RESULTS_FILE"
echo "tag: $RUN_TAG" | tee -a "$RESULTS_FILE"
echo "iterations per split: $ITERATIONS" | tee -a "$RESULTS_FILE"
echo "batch=$BATCH  seq=$SEQ" | tee -a "$RESULTS_FILE"
echo "start: $(date)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

declare -a PCES

for SPLIT in 1 2 3 4; do
    MODEL_DIR="models_${RUN_TAG}_s${SPLIT}"
    TRAIN_LOG="train_${RUN_TAG}_s${SPLIT}.log"
    EVAL_LOG="eval_${RUN_TAG}_s${SPLIT}.log"

    echo "---- split $SPLIT train  $(date) ----" | tee -a "$RESULTS_FILE"
    python train.py \
        --split $SPLIT \
        --model-dir "$MODEL_DIR" \
        --batch-size $BATCH \
        --seq-length $SEQ \
        --iterations $ITERATIONS \
        > "$TRAIN_LOG" 2>&1

    echo "---- split $SPLIT eval  $(date) ----" | tee -a "$RESULTS_FILE"
    # eval only EMA checkpoints with TTA, which is what we report
    python eval.py "$MODEL_DIR" \
        --split $SPLIT \
        --seq-length $SEQ \
        --mode ema \
        --tta \
        > "$EVAL_LOG" 2>&1

    # pull the best PCE out of the eval log for the summary, default 0.0 if missing
    BEST=$(grep -A 1 'Best checkpoint' "$EVAL_LOG" | grep 'PCE:' | awk '{print $2}')
    BEST=${BEST:-0.0000}
    echo "split $SPLIT best EMA+TTA PCE = $BEST" | tee -a "$RESULTS_FILE"
    PCES+=("$BEST")
    echo "" | tee -a "$RESULTS_FILE"
done

echo "==== summary ====" | tee -a "$RESULTS_FILE"
for i in "${!PCES[@]}"; do
    echo "split $((i+1)):  ${PCES[$i]}" | tee -a "$RESULTS_FILE"
done
# compute average in python in one shot with a list literal so empty/failed slots are explicit
AVG=$(python -c "vals=[${PCES[0]}, ${PCES[1]}, ${PCES[2]}, ${PCES[3]}]; print(sum(vals)/len(vals))")
echo "4-split average:  $AVG" | tee -a "$RESULTS_FILE"
echo "paper baseline:   0.7610" | tee -a "$RESULTS_FILE"
echo "end: $(date)" | tee -a "$RESULTS_FILE"
