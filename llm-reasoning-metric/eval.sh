#!/usr/bin/env bash

#SBATCH --job-name=grpo-eval-4b
#SBATCH --output=./llm-reasoning-metric/logs/%j-eval/eval.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "=> Starting evaluation"
python llm-reasoning-metric/evaluate_vllm.py language_pairs=[${LANGUAGE_PAIR}] to_compute=[${TO_COMPUTE}]