import json
import os
import torch.backends.cudnn as cudnn
import numpy as np

import torch
import random
from dataclasses import dataclass, field
from typing import List, Dict


def create_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng

class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TrainingInfo:
    epochs: List[int] = field(default_factory=list)  # List of epoch indices
    losses: List[List[float]] = field(default_factory=list)  # List of loss lists
    errors: List[List[float]] = field(default_factory=list)  # List of error lists
    batch_info: List[Dict[str, float]] = field(default_factory=list)  # List of batch info dictionaries

    def add_epoch_data(self, epoch: int, loss: List[float], error: List[float], batch: Dict[str, float]):
        """
        Add data for a single epoch.
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.errors.append(error)
        self.batch_info.append(batch)
