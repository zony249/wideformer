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
from models import (
    ParallelModel,
    Qwen3ForCausalLM
)

from model_utils import load_model


if __name__ == "__main__": 


    parser = ArgumentParser("finetune.py")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["hellaswag", "wikitext", "coqa"])
    parser.add_argument("--lora_adapter", type=str, default=None, help="\"random_init\", name of adapter, or None")
    parser.add_argument("--eval_every_steps", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--force_load_local_dataset", action="store_true")
    parser.add_argument("--local_dataset_dir", type=str, default=None)
    parser.add_argument("--parallel_lanes", type=int, default=None)
    args = parser.parse_args()

    # os.makedirs(args.output_dir)

    model, tok = load_model(args.base_model, torch_dtype=torch.bfloat16)

    datasets, formatting_func = get_dataset_and_task_processor(args.task, 
                                                               tok=tok, 
                                                               val_test_only=False, 
                                                               load_from_disk=args.force_load_local_dataset, 
                                                               local_dataset_dir=args.local_dataset_dir)

    trainset = datasets["train"]
    eval_set = datasets["validation"]


    # Parallelize model if needed
    if args.parallel_lanes is not None: 
        assert isinstance(model, ParallelModel), f"{args.base_model} is not a ParallelModel"
        print(f"=== PARALLEL LANES: {args.parallel_lanes} ===")
        model.parallelize(args.parallel_lanes)

    # Initialize or load adapters
    if args.lora_adapter == "random_init": 
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules = "all-linear"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # load lora adapter for model 
    elif args.lora_adapter is not None: 
        raise NotImplementedError("TODO: Implement loading trained adapters")


    trainer_cfg = SFTConfig(
        output_dir=args.output_dir, 
        num_train_epochs=args.epochs, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, 
        learning_rate=args.lr, 
        eval_strategy="steps", 
        eval_steps=args.eval_every_steps, 
        save_steps=args.eval_every_steps, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        save_strategy="best", 
        metric_for_best_model="eval_loss")


    trainer = SFTTrainer(model=model, 
                         processing_class=tok,
                         args=trainer_cfg, 
                         train_dataset=trainset,
                         eval_dataset=eval_set,  
                         formatting_func=formatting_func)
    trainer.train()