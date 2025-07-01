




export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --num_processes=1 \
    finetune.py \
        --task=hellaswag \
