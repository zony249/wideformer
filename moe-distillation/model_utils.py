from typing import Tuple, Dict, List, Union, Optional
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
)

from models import (
    Qwen3ForCausalLMParallel
)


def load_model(hf_name_or_path: str, 
               **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: 
    config = AutoConfig.from_pretrained(hf_name_or_path, **kwargs) 
    tok = AutoTokenizer.from_pretrained(hf_name_or_path) 
    
    if config.model_type == "qwen3": 
        model = Qwen3ForCausalLMParallel.from_pretrained(hf_name_or_path, config=config)
    else: 
        model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, config=config)

    return model, tok


