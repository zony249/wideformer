#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00
#SBATCH --account=rrg-lilimou
#SBATCH --output=slurm-logs/slurm-%j-%n-causal-pretrained-prediction.out


export CUDA_VISIBLE_DEVICES=0,1
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export MODEL="models/deepseek-student-3l-mnli-base"
# export MODEL="roberta-large"
export TASK_NAME=mnli
export LORA_ADAPTER="models/mnli-shuffle-3l-3/best_tfmr"
export EXP_NAME=$(date +%y-%m-%d--%T)--finetune-predict
export OUTPUT=runs/$EXP_NAME


mkdir -p $OUTPUT


export START=$(date +%s)

torchrun \
  --nproc_per_node=2 \
  finetune.py \
    --model_name_or_path $MODEL \
    --lora_adapter=$LORA_ADAPTER \
    --task_name $TASK_NAME \
    --cache_dir=glue_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --learning_rate 3e-4 \
    --per_device_eval_batch_size 4 \
    --output_dir $OUTPUT \
    --bf16 \
    --bf16_full_eval \
    --optim adamw_hf \
    --seed $((RANDOM % 100000)) \
   #  --overwrite_cache \


if [[ $TASK_NAME == "mnli" ]]; then
   export TASK=MNLI-m
elif [[ $TASK_NAME == "qqp" ]]; then
   export TASK=QQP
elif [[ $TASK_NAME == "qnli" ]]; then
   export TASK=QNLI
elif [[ $TASK_NAME == "sst2" ]]; then
   export TASK=SST-2
elif [[ $TASK_NAME == "mrpc" ]]; then
   export TASK=MRPC
elif [[ $TASK_NAME == "rte" ]]; then
   export TASK=RTE
elif [[ $TASK_NAME == "stsb" ]]; then
   export TASK=STS-B
elif [[ $TASK_NAME == "cola" ]]; then
   export TASK=CoLA
else 
    echo "FAILURE"
fi


python convert_glue_preds.py --input_file $OUTPUT/predict_results_$TASK_NAME.txt --task $TASK
if [[ $TASK_NAME == "mnli" ]]; then 
   export TASK=MNLI-mm
   python convert_glue_preds.py --input_file $OUTPUT/predict_results_$TASK_NAME-mm.txt --task $TASK
fi


export END=$(date +%s)

export RUNTIME=$(((END-START)/3600)) hrs
echo Total runtime: $RUNTIME
echo Approx. GPU Hours: $((RUNTIME * NUM_GPUS)) hrs