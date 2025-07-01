import os 
from copy import deepcopy 
from typing import Tuple, List, Union, Dict, Any, Optional 
import re

import datasets
from torch.utils.data import Dataset
from datasets import load_dataset 

from .task_utils import AbstractTask

class Hellaswag(AbstractTask): 
    name: str = "hellaswag" 
    def __init__(self, split="train"): 
        super().__init__(split=split)

    def get_datasets(self, **dataset_kwargs) -> Dict[str, Dataset]: 
        """
        For Hellaswag, we only look for the "split" kwarg from dataset_kwargs 
        split: Union[List[str], str]
        """
        assert "split" in dataset_kwargs, f"dataset_kwargs missing argument 'split'"
        datasets = {}
        if isinstance(dataset_kwargs["split"], list):
            for spl in dataset_kwargs["split"]:
                assert spl in ["train", "validation", "test"]
                dataset = load_dataset("Rowan/hellaswag", split=spl)
                datasets[spl] = dataset
        else: 
            assert dataset_kwargs["split"] in ["train", "validation", "test"]
            dataset = load_dataset("Rowan/hellaswag", split=dataset_kwargs["split"])
            datasets[dataset_kwargs["split"]] = dataset
        return datasets


    def pre_process_fn(self, examples: List[Dict]) -> Any:
        if isinstance(examples, list):
            outputs = []
            for example in examples:
                question = f"{example['activity_label']}: {example['ctx_a']} {example['ctx_b'].capitalize()}" + example["endings"][int(example["label"])]
                # options = "\n".join([f"{opt}: " + e for opt, e in zip(["A", "B", "C", "D"], example["endings"])]) + "\n\n"
                # answer = f'Answer: {["A", "B", "C", "D"][int(example["label"])]}'
                outputs.append(question)
            return outputs
        else: 
            question = f"{examples['activity_label']}: {examples['ctx_a']} {examples['ctx_b'].capitalize()}" + examples["endings"][int(examples["label"])]
            return question


# This is from LM_Eval: 

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)




if __name__ == "__main__": 
    hellaswag = Hellaswag("train") 
    print(hellaswag.dataset[0]) 
    print(hellaswag.pre_process_fn([hellaswag.dataset[0]]))

    dataset = load_dataset("Rowan/hellaswag", split="train")
    print(process_docs(dataset)[0])
