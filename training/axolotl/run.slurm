#!/usr/bin/env bash

#SBATCH --job-name=mt-eval-reasoning
#SBATCH --output=./logs/training-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES="3" axolotl vllm-serve ${CONFIG_PATH} > ./logs/vllm/vllm-${SLURM_JOB_ID}.log 2> ./logs/vllm/vllm-${SLURM_JOB_ID}.err &

CUDA_VISIBLE_DEVICES="0,1,2" axolotl train --num-processes 3 ${CONFIG_PATH} --deepspeed ./deepspeed_configs/zero3_bf16.json