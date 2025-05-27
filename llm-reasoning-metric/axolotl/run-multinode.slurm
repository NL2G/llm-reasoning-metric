#!/usr/bin/env bash

#SBATCH --job-name=mt-eval-reasoning
#SBATCH -D .
#SBATCH --output=./logs/training-multinode-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=5

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
LAST_NODE=${NODELIST[-1]}

# Create ALL_BUT_LAST_NODE array using a different approach
ALL_BUT_LAST_NODE=()
for ((i=0; i<${#NODELIST[@]}-1; i++)); do
    ALL_BUT_LAST_NODE+=("${NODELIST[i]}")
done
# Convert array to space-separated string
ALL_BUT_LAST_NODE="${ALL_BUT_LAST_NODE[*]}"
N_ALL_BUT_LAST_NODE=${#ALL_BUT_LAST_NODE[@]}
N_PROCESSES_PER_NODE=4
N_PROCESSES=$((N_ALL_BUT_LAST_NODE * N_PROCESSES_PER_NODE))

echo "ALL_BUT_LAST_NODE: $ALL_BUT_LAST_NODE"
echo "N_ALL_BUT_LAST_NODE: $N_ALL_BUT_LAST_NODE"
echo "N_PROCESSES: $N_PROCESSES"

# Launch vllm on last node
echo "Launching vllm on $LAST_NODE"
srun --nodes=1 --ntasks=1 --nodelist="$LAST_NODE" axolotl vllm-serve --tensor-parallel-size 4 ./configs/grpo/grpo-mid.yaml > ./logs/vllm/vllm-${SLURM_JOB_ID}.log 2> ./logs/vllm/vllm-${SLURM_JOB_ID}.err &

# copy yaml into tmp and override trl.vllm_server_host
yq --yaml-output ".trl.vllm_server_host = \"$LAST_NODE\"" ./configs/grpo/grpo-mid.yaml > ./configs/tmp/grpo-mid-${SLURM_JOB_ID}.yaml

AXOLOTL_COMMAND="-m axolotl.cli.train ./configs/tmp/grpo-mid-${SLURM_JOB_ID}.yaml"


ACCELERATE_MAIN="accelerate launch \
        --num_machines $N_ALL_BUT_LAST_NODE \
        --num_processes $N_PROCESSES \
        --machine_rank $SLURM_PROCID \
        --main_process_ip $MAIN_NODE \
        --main_process_port 29500 \
        --gpu_ids 0,1,2,3 \
        --rdzv_backend c10d"

echo "Launching axolotl on $ALL_BUT_LAST_NODE"
srun --nodes=$N_ALL_BUT_LAST_NODE --ntasks-per-node=1 --nodelist="$ALL_BUT_LAST_NODE" $ACCELERATE_MAIN $AXOLOTL_COMMAND
