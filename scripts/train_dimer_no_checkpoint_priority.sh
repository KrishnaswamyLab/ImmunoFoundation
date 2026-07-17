#!/bin/bash

#SBATCH --job-name=no_checkpoint_pretraining
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=prio_sk2433
#SBATCH --partition=priority_gpu
#SBATCH --gpus=h200:4
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --output=./logs/slurm/no_checkpoint_pretraining/afdb/%x_%j.out
#SBATCH --error=./logs/slurm/no_checkpoint_pretraining/afdb/%x_%j.err
#SBATCH --mail-user=joaofelipe.rocha@yale.edu
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


torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py --config-name train_stage2_no_checkpoint
