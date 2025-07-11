


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=6,7
export HF_HOME=~/large-file-storage

    # --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
accelerate launch \
    distillation.py \
        --task=hellaswag \
        --base_model=weight_copy \
        --teacher_model=Qwen/Qwen3-0.6B \
        --epochs=40 \
        --batch_size=2 \
        --lr=1e-4 \
        --gradient_accumulation_steps=2 \
        --eval_every_steps=200 \
        --ce_alpha=1.0 --kl_alpha=1.0 --hidden_alpha=3.0 \
        --matching_location=forward \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        # --force_load_local_dataset \
        # --local_dataset_dir=wikitext_local \
