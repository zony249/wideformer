


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=7
# export HF_HOME=~/large-file-storage

    # --config-file=accl-config/fsdp-qwen3-conf.yaml \
accelerate launch \
    --num_processes=1 \
    finetune.py \
        --task=coqa \
        --base_model=Qwen/Qwen3-0.6B \
        --lora_adapter="random_init" \
        --epochs=10 \
        --batch_size=2 \
        --gradient_accumulation_steps=2 \
        --lr=2e-5 \
        --eval_every_steps=25 \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=coqa_local \
