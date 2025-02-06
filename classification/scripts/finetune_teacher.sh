

export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# export MODEL="roberta-large"
export TASK_NAME=mrpc
export EXP_NAME=$(date +%x--%T)--TEST
export OUTPUT=runs/$EXP_NAME

mkdir $OUTPUT


python -m debugpy --listen 0.0.0.0:5678 finetune.py \
  --model_name_or_path $MODEL \
  --lora_adapter random_init \
  --is_causal \
  --task_name $TASK_NAME \
  --cache_dir=glue \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size=8 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs 20 \
  --eval_strategy steps \
  --eval_steps 20 \
  --output_dir $OUTPUT \
  --bf16 \
  --bf16_full_eval \
  --optim adamw_hf \
  --seed $((RANDOM % 100000)) \