#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64000M
#SBATCH --time=1-00:00
#SBATCH --account=rrg-lilimou
#SBATCH --output=slurm-logs/%j-llama-3-70B-distributed.out


nvidia-smi
nvidia-smi topo -m

export HEAD_NODE=$(hostname) # store head node's address
export HEAD_NODE_PORT=34568 # choose a port on the main node to start accelerate's main process

# export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

accelerate launch \
    --num_processes=2 \
    -m lm_eval \
        --model hf \
        --model_args pretrained="Qwen/Qwen3-0.6B",parallelize=False,dtype=bfloat16,trust_remote_code=true,add_bos_token=true \
        --tasks hellaswag\
        --batch_size 32 \
        --output_path runs/lm_eval \
        # --log_samples \