import os 
from abc import ABC, abstractmethod 
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file, load_model

import torch 

class ParallelModel: 
    parallel:bool = False 
    
    @abstractmethod
    def parallelize(self, lanes): 
        raise NotImplementedError
    
    def load_from_disk(self, path, use_safetensors=True): 
        if use_safetensors: 
            checkpoint_file = "model.safetensors"
        else: 
            checkpoint_file = "pytorch_model.bin"

        index_file = os.path.join(path, checkpoint_file + ".index.json")
        if os.path.exists(index_file): 
            load_sharded_checkpoint(self, path, prefer_safe=use_safetensors)
        else: 
            load_model(self, os.path.join(path, checkpoint_file), strict=True)
