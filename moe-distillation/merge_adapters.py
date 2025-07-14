import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM
from models.parallel_models.modeling_qwen3 import Qwen3ForCausalLMParallel

base_model = Qwen3ForCausalLMParallel.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
# base_model.parallelize(4)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

peft_model = PeftModelForCausalLM.from_pretrained(base_model, "runs/Qwen3-0.6b-hellaswag/best_tfmr")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("runs/Qwen3-0.6b-hellaswag/")
tokenizer.save_pretrained("runs/Qwen3-0.6b-hellaswag/")