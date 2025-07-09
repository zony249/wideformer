


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=3
export HF_HOME=~/large-file-storage

accelerate launch \
    --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
    finetune.py \
        --task=hellaswag \
        --base_model=Qwen/Qwen3-0.6B \
        --lora_adapter="random_init" \
        --epochs=1 \
        --batch_size=4 \
        --parallel_lanes=2 \
        --eval_every_steps=500 \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=hellaswag_local \
