import os 
from typing import Dict, List, Union, Optional, Tuple 
from dataclasses import dataclass, field


import torch 
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    T5ForConditionalGeneration, 
)

from peft import PeftModel, get_peft_model
from utils import LegacySeq2SeqDataset


@dataclass 
class EvalArguments: 
    data_dir: str = field(
        metadata={"help": "Source data. must contain test.source and test.target"}
    )
    predict_batch_size: Optional[int] = field(
        default=8, metadata={"help": "batch size for prediction"}
    )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": "number of beams for beam search."}
    )
    output_dir: Optional[str] = field(
        default="", metadata={"help": "where to dump generations"}
    )
    max_source_length: Optional[int] = field(
        default=256, metadata={"help": "Maximum length of source text"}
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Maximum length of target text"}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Evaluate in FP16"}
    )
    bf16: bool = field(
        default=False, metadata={"help": "Evaluate in BF16"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random Seed. Defaults to 42"}
    )


@dataclass
class ModelArguments: 

    model_name_or_path: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    lora_adapter: Optional[str] = field(
        default=None, metadata={"help": "LoRA adapter for model, if any."}
    )





if __name__ == "__main__": 

    hfparser = HfArgumentParser((EvalArguments, ModelArguments)) 
    eval_args, model_args = hfparser.parse_args_into_dataclasses()


    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, use_safetensors=True)
    tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path)


    dataset_kwargs: dict = {
        "data_dir": eval_args.data_dir, 
        "prefix": model.config.prefix or "",
    }

    dataset = LegacySeq2SeqDataset(
        tok, 
        type_path="test",
        n_obs=-1,
        max_source_length=eval_args.max_source_length, 
        max_target_length=eval_args.max_target_length,
        **dataset_kwargs,
    )

    print(eval_args)
    print(model_args)


    test_dataloader = DataLoader(
                    dataset,
                    batch_size=eval_args.predict_batch_size,
                    collate_fn=dataset.collate_fn,
                    sampler=None,
                )

    for batch in test_dataloader: 
        

