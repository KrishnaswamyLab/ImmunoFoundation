#!/bin/bash
#SBATCH --job-name=finetune_af3_mlp
#SBATCH --output=outputs/finetune_af3_mlp_%j.out
#SBATCH --error=outputs/finetune_af3_mlp_%j.err
#SBATCH --partition=pi_sk2433_gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am3826@columbia.edu

# Activate environment
source ~/.bashrc
conda activate immuno
cd /home/am3826/workspace/ImmunoFoundation

# Run finetuning for MLP classifier on AF3 data
python train.py experiment=finetune_af3.yaml
