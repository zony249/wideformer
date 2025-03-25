# Add parent directory to python path to access lightning_base.py





export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR="wmt_en-ro_100k"
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export EXP_NAME=$(date +%x--%T)--deepseek-teacher
export OUTPUT_DIR=runs/$EXP_NAME
export NUM_GPUS=2


torchrun \
  --nproc-per-node=$NUM_GPUS \
  finetune.py \
    --model_name_or_path=$MODEL \
    --lora_adapter="random_init" \
    --data_dir=$DATA_DIR \
    --learning_rate 5e-4 \
    --train_batch_size 3 \
    --eval_batch_size 4 \
    --num_train_epochs 9 \
    --gradient_accumulation_steps 4 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length 256 \
    --max_target_length 128 \
    --n_val -1 \
    --eval_steps 1000 \
    --gpus 1 \
    --do_train --do_predict \
    --bf16 \
    --seed 122 \
    # --task "translation" \
    "$@"
