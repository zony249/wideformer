


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=~/large-file-storage

accelerate launch \
    --config-file=accl-config/fsdp-conf.yaml \
    finetune.py \
        --task=hellaswag \
        --base_model=Qwen/Qwen3-8B \
        --epochs=1 \
        --batch_size=4 \
        --output_dir=runs/$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=hellaswag_local \
