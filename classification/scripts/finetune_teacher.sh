#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=0-03:00
#SBATCH --account=rrg-lilimou



export CUDA_VISIBLE_DEVICES=6,7
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# export MODEL="roberta-large"
export TASK_NAME=mnli
export EXP_NAME=$(date +%x--%T)--TEST
export OUTPUT=runs/$EXP_NAME

mkdir $OUTPUT


torchrun \
  --nproc_per_node=2 \
  finetune.py \
    --model_name_or_path $MODEL \
    --is_causal \
    --lora_adapter random_init \
    --task_name $TASK_NAME \
    --cache_dir=glue \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size=6 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps=3 \
    --num_train_epochs 20 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --output_dir $OUTPUT \
    --bf16 \
    --bf16_full_eval \
    --optim adamw_hf \
    --seed $((RANDOM % 100000)) \
  # --overwrite_cache