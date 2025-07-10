from transformers import AutoTokenizer, AutoConfig
from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel
import torch

device=5

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = Qwen3ForCausalLMParallel(config).to(f"cuda:{device}")

# model.parallelize(4)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B") 

model = model.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16)




prompt = "Tell me about LLMs please and thank you."
inputs = tok([prompt], return_tensors="pt").to(f"cuda:{device}")

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True, max_new_tokens=100)
print("".join(tok.batch_decode(outputs[0])))