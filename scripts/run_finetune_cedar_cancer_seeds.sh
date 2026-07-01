#!/bin/bash
# Multi-seed sweep for CEDAR cancer fine-tune. Each iteration:
#   1) (Re)builds the train/val/test partition with that seed -> data/CEDAR_cancer/splits_seed{N}/
#   2) Runs train.py with data.seed=N (the YAML interpolates split paths from data.seed,
#      so the same seed governs both partition + DataLoader/init RNG).
#
# Override the seed list:  SEEDS="0 1 2" ./scripts/run_finetune_cedar_cancer_seeds.sh
# Override the strategy:   STRATEGY=peptide_group ./scripts/run_finetune_cedar_cancer_seeds.sh
#   (the YAML default points at splits_seed{N}/; for peptide_group you also need to
#    override data.split_csv_paths.{train,val,test} to splits_peptide_group_seed{N}/.)
set -e

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
STRATEGY="${STRATEGY:-random}"
EXTRA_ARGS=("$@")

for s in $SEEDS; do
  echo "=========================================="
  echo "CEDAR cancer fine-tune  seed=$s  strategy=$STRATEGY"
  echo "=========================================="
  python scripts/build_cedar_cancer_splits.py --strategy "$STRATEGY" --seed "$s"
  python train.py --config-name train_finetune_cedar_cancer data.seed="$s" "${EXTRA_ARGS[@]}"
done
