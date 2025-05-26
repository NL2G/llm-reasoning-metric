#!/usr/bin/env bash

#SBATCH --job-name=mt-eval-reasoning
#SBATCH --output=./logs/training-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#export HF_HOME="/ceph/dalarion/hf-cache"
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1


axolotl train --num-processes 4 ${CONFIG_PATH} --deepspeed ./deepspeed_configs/zero3_bf16.json