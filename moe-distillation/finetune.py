import os 
from copy import deepcopy
import shutil 
from argparse import Namespace, ArgumentParser
from peft import get_peft_model, LoraConfig, TaskType 


import torch
from datasets import load_dataset
from sft_trainer import SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from sft_trainer import SFTConfig
from data_utils import get_dataset_and_task_processor


if __name__ == "__main__": 


    parser = ArgumentParser("finetune.py")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["hellaswag"])
    parser.add_argument("--lora_adapter", type=str, default=None, help="if specified, then the lora adapters would be loaded. Otherwise, randomly initialize")
    parser.add_argument("--eval_every_steps", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # os.makedirs(args.output_dir)

    datasets, formatting_func = get_dataset_and_task_processor(args.task, val_test_only=False)

    trainset = datasets["train"]
    eval_set = datasets["validation"]

    # dataset = load_dataset("Rowan/hellaswag", split="train")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, 
                                                torch_dtype=torch.bfloat16)

    if args.lora_adapter is not None: 
        pass
        # load lora adapter for model 
        raise NotImplementedError("need to implement loading finetuned LoRA adapter")
    else: 
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules = "all-linear"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer_cfg = SFTConfig(
        output_dir=args.output_dir, 
        num_train_epochs=args.epochs, 
        per_gpu_train_batch_size=args.batch_size,
        eval_strategy="steps", 
        eval_steps=args.eval_every_steps, 
        save_steps=args.eval_every_steps)

    trainer = SFTTrainer(model=model, 
                         args=trainer_cfg, 
                         train_dataset=trainset,
                         eval_dataset=eval_set,  
                         formatting_func=formatting_func)
    trainer.train()