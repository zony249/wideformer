from typing import Tuple, Dict, List, Union, Optional
import torch  
from torch import nn
from copy import deepcopy
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
)

from models import (
    Qwen3ForCausalLMParallel, Qwen3ForCausalLM, Qwen3DecoderLayer
)

LAYER_COPY_MAP = { 
    28: [3, 7, 11, 15, 19, 23, 27], 
    40: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
}


def load_model(hf_name_or_path: str, parallel=True,
               **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: 
    config = AutoConfig.from_pretrained(hf_name_or_path, **kwargs) 
    tok = AutoTokenizer.from_pretrained(hf_name_or_path) 
    
    if config.model_type == "qwen3" and parallel: 
        model = Qwen3ForCausalLMParallel.from_pretrained(hf_name_or_path, config=config, **kwargs)
    else: 
        model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, config=config, **kwargs)

    return model, tok


def create_student_from_teacher(hf_name_or_path: str, mode="weight_copy"): 

    config = AutoConfig.from_pretrained(hf_name_or_path)
    config.torch_dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(hf_name_or_path)
    if config.model_type == "qwen3": 
        template_model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, torch_dtype=torch.bfloat16) 
        config.num_hidden_layers = len(LAYER_COPY_MAP[template_model.config.num_hidden_layers])
        if mode == "weight_copy":
            layers = nn.ModuleList([deepcopy(template_model.model.layers[i]) for i in LAYER_COPY_MAP[template_model.config.num_hidden_layers]])
            template_model.model.layers = layers 
            template_model.config = config
        elif mode == "random_init": 
            template_model = Qwen3ForCausalLM(config)
        return template_model, tok
    else: 
        raise NotImplementedError("layer selection is not implemented for the current model")