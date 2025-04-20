#!/usr/bin/env bash

#SBATCH --job-name=mt-eval-reasoning
#SBATCH --output=./logs/training-%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export HF_HOME="/ceph/dalarion/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES="0" axolotl vllm-serve ./configs/default.yaml > /dev/null 2> /dev/null &

CUDA_VISIBLE_DEVICES="1,2,3" axolotl train --num-processes 3 ./configs/default.yaml --deepspeed ./deepspeed_configs/zero1.json