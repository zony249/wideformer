# Add parent directory to python path to access lightning_base.py
export DATA_DIR="wmt_en-ro_100k"
export EXP_NAME=$(date +%x--%T)--TEST
export OUTPUT_DIR=runs/$EXP_NAME

python test.py \
    --model_name_or_path="runs/02/04/2025--09:39:10--t5-teacher/best_tfmr" \
    --data_dir=$DATA_DIR \
    --predict_batch_size 8 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length 256 \
    --max_target_length 128 \
    --bf16 \
    --seed 123 \
    # --task "translation" \
    "$@"
