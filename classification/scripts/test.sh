#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00
#SBATCH --account=rrg-lilimou
#SBATCH --output=slurm-logs/slurm-%j-%n-causal-pretrained-prediction.out


# export CUDA_VISIBLE_DEVICES=6,7
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# export MODEL="roberta-large"
export TASK_NAME=mnli
export EXP_NAME=$(date +%x--%T)--predict
export OUTPUT=runs/$EXP_NAME


mkdir -p $OUTPUT


torchrun \
  --nproc_per_node=1 \
  finetune.py \
    --model_name_or_path $MODEL \
    --use_causal_lm \
    --task_name $TASK_NAME \
    --cache_dir=glue_pretrained \
    --do_eval \
    --do_predict \
    --max_seq_length 192 \
    --learning_rate 3e-4 \
    --per_device_eval_batch_size 4 \
    --output_dir $OUTPUT \
    --bf16 \
    --bf16_full_eval \
    --optim adamw_hf \
    --max_new_tokens=1024 \
    --seed $((RANDOM % 100000)) \
    --overwrite_cache \