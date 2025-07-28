#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64000M
#SBATCH --time=2-00:00
#SBATCH --account=aip-lilimou
#SBATCH --output=slurm-logs/%j-Qwen3-8b-coqa-finetune.out


nvidia-smi 
nvidia-smi topo -m 



# export CUDA_VISIBLE_DEVICES=3
# export HF_HOME=~/large-file-storage
export HF_HOME=$SCRATCH

accelerate launch \
    --config-file=accl-config/fsdp-qwen3-conf.yaml \
    finetune.py \
        --task=coqa \
        --base_model=Qwen/Qwen3-8B \
        --lora_adapter="random_init" \
        --epochs=2 \
        --batch_size=2 \
        --lr=1e-5 \
        --eval_every_steps=100 \
        --gradient_accumulation_steps=2 \
        --output_dir=runs/Qwen3-8B-finetune-coqa--$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=coqa_local \
