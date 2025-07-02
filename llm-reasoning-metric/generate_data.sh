#!/usr/bin/env bash

#SBATCH --job-name=grpo-sft-data
#SBATCH --output=./llm-reasoning-metric/logs/%A-sft-data/%a/generate_data.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-9

datasets=(
    #"./data/mt-eval.parquet"
    #"./data/mt-da.parquet"
    #"./data/deepscaler.parquet"
    "./data/wmt24-train.parquet"
)

output_paths=(
    #"./data/sft/mt-eval-{shard_id}.parquet"
    #"./data/sft/mt-da-{shard_id}.parquet"
    #"./data/sft/deepscaler-{shard_id}.parquet"
    "./data/sft/wmt24-train-{shard_id}.parquet"
)

max_lengths=(
    #16384
    #16384
    #30720
    16384
)

N_OPTIONS=1
N_SHARDS=10

dataset_idx=$((SLURM_ARRAY_TASK_ID / N_SHARDS))
shard_id=$((SLURM_ARRAY_TASK_ID % N_SHARDS))

# Select the appropriate variables
dataset_path=${datasets[$dataset_idx]}
output_path=${output_paths[$dataset_idx]}
max_length=${max_lengths[$dataset_idx]}

output_path=$(echo $output_path | sed "s/{shard_id}/$shard_id/")

echo "Running for dataset: $dataset_path, output path: $output_path, shard_id: $shard_id, max_length: $max_length"

TORCHDYNAMO_VERBOSE=1 python llm-reasoning-metric/generate_data.py \
    dataset.path=$dataset_path \
    dataset.n_shards=$N_SHARDS \
    dataset.shard_id=$shard_id \
    dataset.output_path="$output_path" \
    llm.max_tokens=$max_length