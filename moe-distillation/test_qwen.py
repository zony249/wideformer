from transformers import AutoTokenizer 
from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel

device=0

model = Qwen3ForCausalLMParallel.from_pretrained("Qwen/Qwen3-0.6B").to(f"cuda:{device}")
model.parallelize(7)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 

prompt = "Tell me about LLMs please and thank you."
inputs = tok([prompt], return_tensors="pt").to(f"cuda:{device}")

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True, num_beams=5)
print(tok.batch_decode(outputs[0]))