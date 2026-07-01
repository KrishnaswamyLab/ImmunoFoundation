#!/bin/bash

#SBATCH --job-name=iedb_finetune
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=prio_sk2433
#SBATCH --partition=priority_gpu
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --output=./logs/slurm/finetune/iedb/%x_%j.out
#SBATCH --error=./logs/slurm/finetune/iedb/%x_%j.err
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT

cd $SLURM_SUBMIT_DIR
ml uv
source .venv/bin/activate

date
hostname
pwd

SEED="${SEED:-0}"
echo "Running fine-tune with seed=$SEED"

python train.py --config-name=train_finetune_iedb data.seed=$SEED
