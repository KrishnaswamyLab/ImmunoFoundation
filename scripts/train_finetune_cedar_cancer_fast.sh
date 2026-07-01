#!/bin/bash

#SBATCH --job-name=cedar_cancer_finetune_fast
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=prio_sk2433
#SBATCH --partition=priority_gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --output=./logs/slurm/finetune/cedar_cancer/%x_%j.out
#SBATCH --error=./logs/slurm/finetune/cedar_cancer/%x_%j.err
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT

set -euo pipefail

cd $SLURM_SUBMIT_DIR
ml uv
source .venv/bin/activate

date
hostname
pwd

# Defaults — override via sbatch --export=ALL,VAR=val
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
INIT_CKPT="${INIT_CKPT:-/nfs/roberts/project/pi_sk2433/shared/ImmunoFoundationCkpt/phase2/epoch=15-step=5040.ckpt}"
TAG="${TAG:-phase2}"

echo "=========================================="
echo "Fast fine-tune on CEDAR cancer (frozen backbone, cached embeddings)"
echo "  SEED=$SEED  BATCH_SIZE=$BATCH_SIZE  MAX_EPOCHS=$MAX_EPOCHS"
echo "  INIT_CKPT=$INIT_CKPT"
echo "  TAG=$TAG"
echo "=========================================="

# 0) Build per-seed splits if missing (also auto-extracts the PDB zips on first run)
if [ ! -f "data/CEDAR_cancer/splits_seed${SEED}/train.csv" ]; then
  python scripts/build_cedar_cancer_splits.py --strategy random --seed "$SEED"
fi

# 1) Precompute backbone embeddings if missing for this checkpoint (one-time, ~5 min on GPU for ~2.7k structures)
python scripts/precompute_backbone.py \
  --config configs/train_finetune_cedar_cancer.yaml \
  --init-checkpoint "$INIT_CKPT"

# 2) Train fusion + head only on the cached embeddings.
# NOTE: init_checkpoint path contains '=' (e.g. epoch=15-step=5040.ckpt). Hydra splits CLI
# overrides on '=', so we must single-quote the value to make Hydra treat it as a string literal.
python train.py --config-name=train_finetune_cedar_cancer \
  "init_checkpoint='$INIT_CKPT'" \
  data.use_cached_embeddings=true \
  data.batch_size=$BATCH_SIZE \
  data.seed=$SEED \
  model.freeze_backbone=true \
  experiment.trainer.max_epochs=$MAX_EPOCHS \
  experiment.wandb.name="cedar_cancer_fast_${TAG}_seed${SEED}" \
  experiment.checkpointer.dirpath="ckpt/immunofoundation/cedar_cancer_fast_${TAG}_seed${SEED}"
