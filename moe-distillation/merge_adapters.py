import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM
from models.parallel_models.modeling_qwen3 import Qwen3ForCausalLMParallel
from model_utils import create_student_from_teacher

base = "Qwen/Qwen3-8B"
adapter = "runs/q3-8b-coqa-shuffle-distill/best_tfmr"
save_as = "runs/q3-8b-coqa-shuffle-distill"

# base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
base_model, tok = create_student_from_teacher(base, "weight_copy")
# base_model.parallelize(4)
tokenizer = AutoTokenizer.from_pretrained(base)

peft_model = PeftModelForCausalLM.from_pretrained(base_model, adapter)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(save_as)
tokenizer.save_pretrained(save_as)