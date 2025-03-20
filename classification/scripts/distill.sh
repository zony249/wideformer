#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --time=1-00:00
#SBATCH --account=rrg-lilimou
#SBATCH --output=slurm-logs/slurm-%j-%n-mnli-forward.log




sleep $((RANDOM % 100))

# export CUDA_VISIBLE_DEVICES=
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# export MODEL="roberta-large"
export TASK_NAME=mnli
export EXP_NAME=$(date +%y-%m-%d--%T)--$TASK_NAME-forward
export OUTPUT=$SCRATCH/wideformer/classification/runs/$EXP_NAME
export LORA_ADAPTER="models/$TASK_NAME-teacher/best_tfmr"
export NUM_GPUS=4


mkdir -p $OUTPUT

export START=$(date +%s)

torchrun \
  --nproc_per_node=$NUM_GPUS \
  distillation.py \
    --model_name_or_path=$MODEL \
    --lora_adapter=$LORA_ADAPTER \
    --task_name=$TASK_NAME \
    --num_student_layers=7 \
    --student_init_strategy=weight_copy \
    --cache_dir=glue_train \
    --do_train \
    --do_eval \
    --max_seq_length=128 \
    --per_device_train_batch_size=3 \
    --learning_rate=2e-4 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=20 \
    --eval_strategy=steps \
    --eval_steps=1000 \
    --ce_alpha=1. --kl_alpha=1. --hidden_alpha=3. \
    --output_dir=$OUTPUT \
    --bf16 \
    --bf16_full_eval \
    --optim=adamw_hf \
    --seed=$((RANDOM % 100000)) \
    # --overwrite_cache

export END=$(date +%s)

export RUNTIME=$(((END-START)/3600)) hrs
echo Total runtime: $RUNTIME
echo Approx. GPU Hours: $((RUNTIME * NUM_GPUS)) hrs