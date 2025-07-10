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


def load_model(hf_name_or_path: str, parallel=True,
               **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: 
    config = AutoConfig.from_pretrained(hf_name_or_path, **kwargs) 
    tok = AutoTokenizer.from_pretrained(hf_name_or_path) 
    
    if config.model_type == "qwen3" and parallel: 
        model = Qwen3ForCausalLMParallel.from_pretrained(hf_name_or_path, config=config, **kwargs)
    else: 
        model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, config=config, **kwargs)

    return model, tok


