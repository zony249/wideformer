

# export CUDA_VISIBLE_DEVICES=3
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# export MODEL="roberta-large"
export TASK_NAME=mnli
export EXP_NAME=$(date +%x--%T)--TEST
export OUTPUT=runs/$EXP_NAME

mkdir $OUTPUT


python -m debugpy --listen 0.0.0.0:5678 predict.py \
  --model_name_or_path $MODEL \
  --is_causal \
  --task_name $TASK_NAME \
  --cache_dir=glue \
  --do_predict \
  --max_seq_length 144 \
  --learning_rate 3e-4 \
  --per_device_eval_batch_size 1 \
  --output_dir $OUTPUT \
  --bf16 \
  --bf16_full_eval \
  --optim adamw_hf \
  --seed $((RANDOM % 100000)) \
  --overwrite_cache \