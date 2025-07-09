from transformers import AutoTokenizer, AutoConfig
from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel

device=0

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLMParallel(config).to(f"cuda:{device}")

model.parallelize(4)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 

model.load_from_disk("runs/Qwen3-hellaswag-parallel")




prompt = "Tell me about LLMs please and thank you."
inputs = tok([prompt], return_tensors="pt").to(f"cuda:{device}")

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True, max_new_tokens=100)
print("".join(tok.batch_decode(outputs[0])))