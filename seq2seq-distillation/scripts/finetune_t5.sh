# Add parent directory to python path to access lightning_base.py
export DATA_DIR="wmt_en-ro_100k"
export EXP_NAME=$(date +%x--%T)--t5-teacher
export OUTPUT_DIR=runs/$EXP_NAME

python -m debugpy --listen 0.0.0.0:5678 finetune.py \
    --model_name_or_path="t5-base" \
    --data_dir=$DATA_DIR \
    --learning_rate 5e-4 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_train_epochs 9 \
    --gradient_accumulation_steps 2 \
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
