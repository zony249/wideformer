
from copy import deepcopy

import torch 
from torch import nn
from transformers import (
    PreTrainedModel,
)

from peft import (
    get_peft_model, 
    LoraConfig, 
    PeftModel, 
    TaskType
)

from modeling_qwen2 import Qwen2PreTrainedModel
from layer_mappings import LAYER_MAPPING, LAYER_COPYING



def init_student_from_teacher(teacher: PreTrainedModel, num_student_layers: int, model_args): 
    
    if teacher.config.model_type == "qwen2":
        t_layers = teacher.config.num_hidden_layers
        selected_layers = nn.ModuleList([deepcopy(teacher.model.layers[i]) for i in LAYER_COPYING[t_layers][num_student_layers]])
        student = deepcopy(teacher) 
        student.model.layers = selected_layers 

        student.config.num_hidden_layers = num_student_layers 
        student.config.max_window_layers = num_student_layers 
    else: 
        raise NotImplementedError 
    
    for _, param in student.named_parameters():
        param.requires_grad_(True)

    lora_config = LoraConfig(
        r=64, 
        init_lora_weights="gaussian", #"loftq", loftq_config=LoftQConfig(), 
        lora_dropout=0.1, 
        # target_modules=["query_proj", "key_proj"], 
        task_type=TaskType.CAUSAL_LM if model_args.use_causal_lm else TaskType.SEQ_CLS, 
        modules_to_save= ["lm_head.weight"] if model_args.use_causal_lm else ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight'],
    )
    student  = get_peft_model(student, lora_config)

    return student


def init_student_from_random(num_student_layers:int): 
    raise NotImplementedError
    return None

