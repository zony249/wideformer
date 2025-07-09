


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/large-file-storage

accelerate launch \
    --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
    distillation.py \
        --task=hellaswag \
        --base_model=Qwen/Qwen3-0.6B \
        --lora_adapter="random_init" \
        --teacher_model=runs/Qwen3-8B-finetuned-hellaswag \
        --epochs=1 \
        --batch_size=4 \
        --parallel_lanes=4 \
        --eval_every_steps=500 \
        --ce_alpha=1.0 --kl_alpha=1.0 --hidden_alpha=1.0 \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=hellaswag_local \
