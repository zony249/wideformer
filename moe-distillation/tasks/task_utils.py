import os 
import sys 
from copy import deepcopy 
from typing import Tuple, List, Union, Dict, Any, Optional
from abc import ABC, abstractmethod

import torch 
from torch.utils.data import Dataset
from datasets import load_dataset 

class AbstractTask: 
    name: str 
    def __init__(self, **dataset_kwargs): 
        self.datasets = self.get_datasets(**dataset_kwargs) # {"split": dataset}
    @abstractmethod
    def get_dataset(self, **dataset_kwargs): 
        raise NotImplementedError 
    @abstractmethod
    def pre_process_fn(self, example: Dict) -> Any:
        raise NotImplementedError