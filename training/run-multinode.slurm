#!/usr/bin/env bash

#SBATCH --job-name=mt-eval-reasoning
#SBATCH -D .
#SBATCH --output=./logs/training-multinode-%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=2

export HF_HOME="/ceph/dalarion/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="bond0"

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "================================================"
echo "Head node address: $head_node_ip, Current node address: $(hostname), Rank: $SLURM_PROCID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
ifconfig
echo "================================================"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

MAIN_NODE=${NODELIST[0]}
SECONDARY_NODE=(${NODELIST[1]})

# Launch vllm on main
srun --nodes=1 --ntasks=1 --nodelist="$MAIN_NODE" CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve ./configs/default.yaml > /dev/null 2> /dev/null &

AXOLOTL_COMMAND="-m axolotl.cli.train \
            ./configs/default.yaml \
            --vllm_host $head_node_ip \
            --vllm_port 8000 \
            --deepspeed ./deepspeed_configs/zero1.json"

ACCELERATE_MAIN="accelerate launch \
        --num_machines $SLURM_NNODES \
        --machine_rank $SLURM_PROCID \
        --main_process_ip $head_node_ip \
        --main_process_port 29500 \
        --gpu_ids 1,2,3 \
        --rdzv_backend c10d"

ACCELERATE_SECONDARY="accelerate launch \
        --gpu_ids 0,1,2,3 \
        --num_machines $SLURM_NNODES \
        --machine_rank $SLURM_PROCID \
        --main_process_ip $head_node_ip \
        --main_process_port 29500 \
        --rdzv_backend c10d"

# Launch training on secondary node (4 GPUs)
srun --nodes=1 --ntasks=1 --nodelist="$SECONDARY_NODE" $ACCELERATE_SECONDARY $AXOLOTL_COMMAND > /dev/null 2> /dev/null &

# Launch training on main node (3 remaining GPUs)
srun --nodes=1 --ntasks=1 --nodelist="$MAIN_NODE" $ACCELERATE_MAIN $AXOLOTL_COMMAND
        



