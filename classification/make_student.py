
from copy import deepcopy

import torch 
from torch import nn
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    PreTrainedModel,
)

from peft import (
    get_peft_model, 
    LoraConfig, 
    PeftModel, 
    TaskType
)

from modeling_qwen2 import Qwen2PreTrainedModel, Qwen2ForSequenceClassification
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




if __name__ == "__main__": 

    model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    teacher_lora = "models/mnli-teacher/best_tfmr"
    num_labels = 3
    task_name = "mnli"
    cache_dir = "glue_train"
    model_load_dtype=torch.bfloat16
    student_init_strategy = "weight_copy"
    num_student_layers = 3


    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=cache_dir,
        # token=model_args.token,
        # trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        # use_fast=model_args.use_fast_tokenizer,
        # revision=model_args.model_revision,
        # token=model_args.token,
        # trust_remote_code=model_args.trust_remote_code,
    )

    # if model_args.use_causal_lm: 
    #     model = Qwen2ForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         config=config,
    #         revision=model_args.model_revision,
    #         token=model_args.token,
    #         trust_remote_code=model_args.trust_remote_code,
    #         ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    #         torch_dtype=model_load_dtype, 
    #     )
    # else:
    model = Qwen2ForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        # revision=model_args.model_revision,
        # token=model_args.token,
        # trust_remote_code=model_args.trust_remote_code,
        # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        torch_dtype=model_load_dtype, 
    )
    # print(model)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if teacher_lora is not None: 
        if teacher_lora == "random_init": 
            lora_config = LoraConfig(
                r=64, 
                init_lora_weights="gaussian", #"loftq", loftq_config=LoftQConfig(), 
                lora_dropout=0.1, 
                # target_modules=["query_proj", "key_proj"], 
                task_type=TaskType.SEQ_CLS, 
                modules_to_save= ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight'],
            )
            model = get_peft_model(model, lora_config)
        else: 
            model = PeftModel.from_pretrained(model, teacher_lora)
        # For distillation, we merge the teacher model first! 
        teacher = model.merge_and_unload() 
    else: 
        teacher = model 





    # TODO: create a way to specify a student and teacher model
    # If only teacher specified then student will be init from teacher
    # otherwise, load student from args.

    class Object(object): 
        pass 

    model_args = Object() 
    model_args.use_causal_lm = False
    if student_init_strategy == "weight_copy": 
        student = init_student_from_teacher(teacher, num_student_layers, model_args) 
    else: 
        student = init_student_from_random(model.args.num_student_layers)

    student = student.unload()    
    student.save_pretrained("deepseek-student-3l-mnli-base")
    # print_trainable_parameters(student)

