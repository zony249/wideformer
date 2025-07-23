import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM
from models.parallel_models.modeling_qwen3 import Qwen3ForCausalLMParallel
from model_utils import create_student_from_teacher

base = "Qwen/Qwen3-8B"
adapter = "runs/transfers/q3-8b-hellaswag-all_one-distill/last_tfmr"
save_as = "runs/transfers/q3-8b-hellaswag-all_one-distill"

print(f"Saving to {save_as}")
# base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
base_model, tok = create_student_from_teacher(base, "weight_copy")
# base_model.parallelize(4)
tokenizer = AutoTokenizer.from_pretrained(base)

peft_model = PeftModelForCausalLM.from_pretrained(base_model, adapter)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(save_as)
tokenizer.save_pretrained(save_as)

num_params = 0
for p in merged_model.parameters():
    num_params += p.numel() 
print("====== NUMBER OF PARAMETERS ======")
print(f"{num_params:,}")