import os
from transformers import AutoTokenizer, AutoConfig
from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

config = AutoConfig.from_pretrained("runs/Qwen3-0.6b-hellaswag")
model = Qwen3ForCausalLM.from_pretrained("runs/Qwen3-0.6b-hellaswag", device_map="auto")

# model.parallelize(4)
tok = AutoTokenizer.from_pretrained("runs/Qwen3-0.6b-hellaswag") 




prompt = "Tell me about LLMs please and thank"
inputs = tok([prompt], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True, max_new_tokens=100)
print("".join(tok.batch_decode(outputs[0])))