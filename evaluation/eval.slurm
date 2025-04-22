#!/usr/bin/env bash

#SBATCH --job-name=run-eval-local
#SBATCH --output=./logs/eval-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-vram-48gb,gpu-vram-32gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export HF_HOME="/ceph/dalarion/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "<=============>"
echo "=> Running eval for ${MODEL_NAME} with ${PROMPT_TYPE} prompt type and ${LANGUAGE_PAIR} language pair, model id ${MODEL_ID};"
echo "<=============>"
nvitop -1
echo "<=============>"

python pairwise_eval-local.py \
    --model_name=${MODEL_NAME} \
    --prompt_type=${PROMPT_TYPE} \
    --use_system_prompt \
    --language_pairs=${LANGUAGE_PAIR} \
    --competition='wmt24' \
    --output="./${MODEL_ID}-${LANGUAGE_PAIR}.json" \
    --traces_sample_path="./${MODEL_ID}-${LANGUAGE_PAIR}-traces.csv" 
    --traces_sample_size=100