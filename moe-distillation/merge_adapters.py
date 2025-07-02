import os 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

peft_model = PeftModelForCausalLM.from_pretrained(base_model, "runs/2025-07-02--11:47:57/best_tfmr")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("runs/Qwen3-finetuned-hellaswag")
tokenizer.save_pretrained("runs/Qwen3-finetuned-hellaswag")