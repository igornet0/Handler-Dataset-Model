import torch
from torch.utils.data import Dataset as _Dataset
import numpy as np

from . import * 
from .Model import Model

class HandlerModel:

    def __init__(self, models: list[Model] = []):

        self.models = models
    
    def load_model(self, path_model: str):

        model = Model().load(path_model)
        self.models.append(model)

        return model