#!/usr/bin/env bash

#SBATCH --job-name=sft-4b
#SBATCH --output=./llm-reasoning-metric/logs/%j-sft/training.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

set -e

GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=8
MODEL_NAME=Qwen/Qwen3-4B
MODEL_ID=qwen3_4b
LR=1e-5


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
for i in $(seq 0 $(($num_nodes-1))); do
    echo "====> Node $i: [${nodes_array[$i]}]"
done

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

echo "====> Head node IP: [$head_node_ip]"


srun torchrun \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --nnodes=$num_nodes \
    --nproc-per-node=4 \
    -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/sft-clean/train.parquet \
        data.val_files=./data/sft-clean/test.parquet \
        data.multiturn.enable=true \
        data.multiturn.messages_key=multiturn \
        data.max_length=40960 \
        data.truncation=error \
        data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        data.train_batch_size=$GLOBAL_BATCH_SIZE \
        +model.fsdp_config.model_dtype=bf16 \
        model.partial_pretrain=$MODEL_NAME \
        model.use_liger=True \
        model.enable_gradient_checkpointing=True \
        optim.lr=$LR \
        ulysses_sequence_parallel_size=4 \
        use_remove_padding=True \
        optim.lr_scheduler=cosine \
        optim.clip_grad=1.0 \
        optim.warmup_steps_ratio=0.1 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='llm-reasoning-mt-eval-sft' \
        trainer.experiment_name=$MODEL_ID \
        +trainer.nnodes=$num_nodes \
        +trainer.n_gpus_per_node=4 \
        trainer.default_local_dir="/hnvme/workspace/v106be28-outputs/verl-sft/$MODEL_ID" \
        trainer.total_epochs=1 $@