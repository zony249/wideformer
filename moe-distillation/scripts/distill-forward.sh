


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=1
# export HF_HOME=~/large-file-storage

    # --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
accelerate launch \
    --num_processes=4 \
    distillation.py \
        --task=coqa \
        --base_model=weight_copy \
        --teacher_model=runs/q3-0.6b-coqa--teacher \
        --lora_adapter=random_init \
        --epochs=20 \
        --batch_size=2 \
        --lr=2e-4 \
        --gradient_accumulation_steps=2 \
        --eval_every_steps=1000 \
        --ce_alpha=1.0 --kl_alpha=1.0 --hidden_alpha=3.0 \
        --matching_location=forward \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=coqa_local \
