#!/usr/bin/env bash

#SBATCH --job-name=mt-serve
#SBATCH --output=./logs/serve-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export HF_HOME="/ceph/dalarion/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=1

echo "Starting server on $(hostname)"
vllm serve \
    --api-key="EMPTY" \
    --host=0.0.0.0 \
    --max-num-seqs=512 \
    --max-log-len=0 \
    --enable-chunked-prefill \
    --port=8081 \
    --enable-reasoning \
    --reasoning-parser="deepseek_r1" \
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"