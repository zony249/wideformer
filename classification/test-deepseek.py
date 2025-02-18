import os 
import numpy as np 

import torch 
from transformers import (
    AutoModelForCausalLM,  
    AutoModelForSequenceClassification,  
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments, 
)





if __name__ == "__main__": 

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        torch_dtype=torch.bfloat16).cuda()    
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "

    special_tokens = {
        "user": 151644, 
        "assistant": 151645
    }
    prefix = tok.decode(special_tokens["user"]) + " Explain in one word whether the following pair of sentences exhibit logical \"entailment\", \"neutral\", or \"contradiction\": "

    prompt = tok.bos_token + " Sam likes to eat icecream." + tok.bos_token + " Sam hates icecream." 

    suffix = tok.decode(special_tokens["assistant"]) 

    complete_prompt = system_prompt + prefix + prompt + suffix 

    inputs = tok([complete_prompt], return_tensors="pt")
    inputs = {k:v.cuda() for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=4096) 

    decoded = tok.batch_decode(outputs)
    print(decoded[0])

