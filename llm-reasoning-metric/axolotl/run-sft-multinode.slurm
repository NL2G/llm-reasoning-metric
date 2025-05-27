#!/usr/bin/env bash

#SBATCH --job-name=sft-multinode
#SBATCH -D .
#SBATCH --output=./logs/sft-multinode-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=8

#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#export NCCL_DEBUG=INFO

echo "================================================"
echo "Current node address: $(hostname), Rank: $SLURM_PROCID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
ifconfig
echo "================================================"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

MAIN_NODE=${NODELIST[0]}

N_NODES=${#NODELIST[@]}
N_PROCESSES_PER_NODE=4
N_PROCESSES=$((N_NODES * N_PROCESSES_PER_NODE))

echo "N_NODES: $N_NODES"
echo "N_PROCESSES: $N_PROCESSES"

AXOLOTL_COMMAND="-m axolotl.cli.train $CONFIG_PATH --deepspeed ./deepspeed_configs/zero3_bf16.json"


ACCELERATE_MAIN="accelerate launch \
        --num_machines $N_NODES \
        --num_processes $N_PROCESSES \
        --machine_rank $SLURM_PROCID \
        --main_process_ip $MAIN_NODE \
        --main_process_port 29500 \
        --gpu_ids 0,1,2,3 \
        --rdzv_backend c10d"

echo "Launching axolotl on $N_NODES"
srun --nodes=$N_NODES --ntasks-per-node=1 --nodelist="$NODELIST" $ACCELERATE_MAIN $AXOLOTL_COMMAND
