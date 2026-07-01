#!/bin/bash
# Full-path CEDAR cancer fine-tune (sequence + structure backbones unfrozen, CW contrastive on).
# Submit one per seed:
#   for s in 0 1 2 3 4; do sbatch --job-name=cedar_cancer_seed${s} --export=ALL,SEED=${s} scripts/train_finetune_cedar_cancer.sh; done
#
# Each job rebuilds its own data/CEDAR_cancer/splits_seed${SEED}/ partition (idempotent),
# then runs train.py with data.seed=${SEED}. The YAML interpolates split paths from data.seed,
# so seed governs both the partition and DataLoader/init RNG.

#SBATCH --job-name=cedar_cancer_finetune
#SBATCH --time=24:00:00
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

SEED="${SEED:-0}"
STRATEGY="${STRATEGY:-random}"

echo "=========================================="
echo "CEDAR cancer fine-tune (full path + CW contrastive)"
echo "  SEED=$SEED  STRATEGY=$STRATEGY"
echo "=========================================="

# Build per-seed splits if missing (idempotent; auto-extracts mut + wt zips on first run)
if [ ! -f "data/CEDAR_cancer/splits_seed${SEED}/train.csv" ]; then
  python scripts/build_cedar_cancer_splits.py --strategy "$STRATEGY" --seed "$SEED"
fi

python train.py --config-name=train_finetune_cedar_cancer data.seed="$SEED"
