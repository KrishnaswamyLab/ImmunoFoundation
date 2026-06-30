#!/bin/bash
#SBATCH --job-name=finetune_if
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /home/am3826/workspace/ImmunoFoundation/.venv/bin/activate
cd /home/am3826/workspace/ImmunoFoundation

python3 finetune_classifier.py
