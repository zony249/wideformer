import os 
from abc import ABC, abstractmethod 

import torch 

class ParallelModel: 
    parallel:bool = False 
    
    @abstractmethod
    def parallelize(self, lanes): 
        raise NotImplementedError
    
