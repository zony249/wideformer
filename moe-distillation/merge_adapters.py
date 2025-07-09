import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

peft_model = PeftModelForCausalLM.from_pretrained(base_model, "runs/Qwen3-0.6B-adapters-hellaswag")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("runs/Qwen3-0.6B-finetuned-hellaswag")
tokenizer.save_pretrained("runs/Qwen3-0.6B-finetuned-hellaswag")