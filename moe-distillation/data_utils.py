import os 
import sys 
from copy import deepcopy 
from typing import Tuple, List, Union, Optional

import torch 
from torch.utils.data import Dataset
from datasets import load_dataset 
from tasks import Hellaswag

TASK_MAP = { 
    "hellaswag": Hellaswag
}

def get_dataset_and_task_processor(task_name: str, 
                                   val_test_only=False, 
                                   load_from_disk=False,
                                   local_dataset_dir=None) -> Tuple[Dataset, callable]: 
    # NOTE: in the sft trainer, if the dataset is already pre-processed, i.e. if it has input_ids, then 
    # sft will skip a bunch of processing.
    assert task_name in TASK_MAP, f"'{task_name}' not recognized as a registered task name"
    if task_name == "hellaswag": 
        split = ["validation", "test"] 
        split = split + ["train"] if not val_test_only else split
        task = TASK_MAP[task_name](split=split, 
                                   load_local=load_from_disk, 
                                   local_dir=local_dataset_dir) 
    else: 
        raise NotImplementedError
    return task.datasets, task.pre_process_fn