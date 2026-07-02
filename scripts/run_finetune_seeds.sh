#!/bin/bash
set -e

cd "$(dirname "$0")/.."

SEEDS=(435576473 682937872 115102908 43848460 638996982)

for s in "${SEEDS[@]}"; do
  echo "=========================================="
  echo "Fine-tune seed $s"
  echo "=========================================="
  python train.py --config-name train_finetune_iedb data.seed=$s
done
