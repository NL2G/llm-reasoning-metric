#!/usr/bin/env bash

#SBATCH --job-name=grpo-multi-node-14b
#SBATCH --output=./llm-reasoning-metric/logs/%j-multi/training.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1

set -e

GLOBAL_BATCH_SIZE=1024
MINI_BATCH_SIZE=512
MICRO_BATCH_SIZE=2
MODEL_NAME=Qwen/Qwen3-14B
MODEL_ID=qwen3_14b
LR=1e-6

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

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "====> IP Head: [$ip_head]"

echo "====> Starting HEAD at [$head_node]"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

for i in $(seq 1 $(($num_nodes-1))); do
    node_i=${nodes_array[$i]}
    echo "====> Starting WORKER [$i] at [$node_i]"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
    sleep 5
done

echo "====> All nodes started"


PYTHONUNBUFFERED=1 ray job submit --address "$ip_head" -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files=[./data/mt-eval.parquet,./data/deepscaler.parquet] \
    data.val_files=./data/mt-eval-test.parquet \
    data.train_batch_size=$GLOBAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    +actor_rollout_ref.rollout.presence_penalty=1.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=./llm-reasoning-metric/verl_reward.py \
    custom_reward_function.name=reward_router \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='llm-reasoning-mt-eval' \
    trainer.experiment_name=$MODEL_ID \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE} \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.default_local_dir="/hnvme/workspace/v106be28-outputs/verl/$MODEL_ID" \
    trainer.total_epochs=1 $@