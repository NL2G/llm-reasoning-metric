#!/usr/bin/env bash
#SBATCH --job-name=llm-reasoning-eval
#SBATCH --output=./llm-reasoning-metric/logs/%A-eval/%a/eval.out
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-44

MODEL_NAME="Rexhaif/Qwen3-4B-MT-Eval"
MODEL_ID="qwen3-4b-vX"

REASONING_EFFORTS=(
    "unbounded"
    "high"
    "medium"
    "low"
    "disabled"
)

LANGUAGE_PAIRS=(
    "en-de"
    "en-es"
    "ja-zh"
)

KINDS=(
    "gemba-da-like"
    "gemba-esa"
    "mt-ranking"
)

N_LANGUAGE_PAIRS=${#LANGUAGE_PAIRS[@]}
N_REASONING_EFFORTS=${#REASONING_EFFORTS[@]}
N_KINDS=${#KINDS[@]}

# Calculate indices from SLURM_ARRAY_TASK_ID
# We iterate: for each kind, for each reasoning effort, for each language pair
language_pair_idx=$((SLURM_ARRAY_TASK_ID % N_LANGUAGE_PAIRS))
reasoning_effort_idx=$(((SLURM_ARRAY_TASK_ID / N_LANGUAGE_PAIRS) % N_REASONING_EFFORTS))
kind_idx=$((SLURM_ARRAY_TASK_ID / (N_LANGUAGE_PAIRS * N_REASONING_EFFORTS)))

# Select the appropriate values
LANGUAGE_PAIR=${LANGUAGE_PAIRS[$language_pair_idx]}
REASONING_EFFORT=${REASONING_EFFORTS[$reasoning_effort_idx]}
KIND=${KINDS[$kind_idx]}

echo "--------------------------------"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Evaluating model: ${MODEL_NAME} with kind: ${KIND} and reasoning effort: ${REASONING_EFFORT} on language pair: ${LANGUAGE_PAIR}"
echo "--------------------------------"

PYTHONUNBUFFERED=1 python llm-reasoning-metric/evaluate_vllm.py \
    experiment.model=${MODEL_NAME} \
    experiment.model_id=${MODEL_ID} \
    experiment.kind=${KIND} \
    experiment.reasoning_effort=${REASONING_EFFORT} \
    competition=wmt24 \
    language_pairs=[${LANGUAGE_PAIR}]