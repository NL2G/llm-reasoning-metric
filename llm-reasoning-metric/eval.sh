#!/usr/bin/env bash

#SBATCH --job-name=grpo-eval-4b
#SBATCH --output=./llm-reasoning-metric/logs/%j-eval/eval.out
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "=> Starting evaluation"

# if competition is not set, set it to wmt24
if [ -z "${COMPETITION}" ]; then
    echo "=> Competition is not set, setting it to wmt24"   
    COMPETITION="wmt24"
else
    echo "=> Competition is set to ${COMPETITION}"
fi

PYTHONUNBUFFERED=1 python llm-reasoning-metric/evaluate_vllm.py language_pairs=[${LANGUAGE_PAIR}] to_compute=[${TO_COMPUTE}] competition=${COMPETITION}