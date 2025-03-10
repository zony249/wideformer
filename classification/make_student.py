
from copy import deepcopy

import torch 
from torch import nn
from transformers import (
    PreTrainedModel
)

from modeling_qwen2 import Qwen2PreTrainedModel
from layer_mappings import LAYER_MAPPING, LAYER_COPYING



def init_student_from_teacher(teacher: PreTrainedModel, num_student_layers: int): 
    
    if teacher.config.model_type == "qwen2":
        t_layers = teacher.config.num_hidden_layers
        selected_layers = nn.ModuleList([deepcopy(teacher.model.layers[i]) for i in LAYER_COPYING[t_layers][num_student_layers]])
        student = deepcopy(teacher) 
        student.model.layers = selected_layers 

        student.config.num_hidden_layers = num_student_layers 
        student.config.max_window_layers = num_student_layers 
    else: 
        raise NotImplementedError
    

    return student


def init_student_from_random(num_student_layers:int): 
    raise NotImplementedError
    return None

