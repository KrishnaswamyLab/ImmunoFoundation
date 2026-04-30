#!/bin/bash

#SBATCH --job-name=stage_2
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h200
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --output=./logs/slurm/stage_2/dimer/%x_%j.out
#SBATCH --error=./logs/slurm/stage_2/dimer/%x_%j.err
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT


cd $SLURM_SUBMIT_DIR
ml uv
source .venv/bin/activate

date
hostname
pwd

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export NCCL_IB_DISABLE=1  
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python train.py --config-name train_stage2