from os.path import join, isdir
from os import listdir, walk
from types import FunctionType as function
from typing import Callable, List, Union, Any

from .Data import *
from .Label import *

class Labels:

    def __init__(self, labels, output_shape: int = None, filter: Callable = None, map: Callable = None) -> None:
        
        self.labels = labels
        
        self.output_shape = output_shape

        self.filter = filter
        self.map = map

        self.buffer = {}

    def clear_buffer(self):
        self.buffer = {}

    def get(self, key: Any) -> Any:
        if isinstance(self.labels, dict):
            return self.labels[key]
        elif isinstance(self.labels, list):
            return self.labels[key]
        elif isinstance(self.labels, function):
            return self.labels(key)
        else:
            raise ValueError(f"Labels type {type(self.labels)} not supported {key=}")

    def __getitem__(self, key: Any) -> Any:
        if key in self.buffer.keys():
            label = self.buffer[key]
        else:
            label = self.get(key)
        
        label = self.process_label(label)

        self.buffer[key] = label

        return label
    
    def set_label(self, key: Any, label: Label):
        if not key in self.buffer.keys():
            raise ValueError(f"Key {key} not found in buffer")
        
        self.buffer[key] = label
    
    def to_type(self, label) -> Label:
        if isinstance(label, Label):
            return label
        else:
            return Label(label)
        
    def process_label(self, label):
        label: Label = self.to_type(label)

        if self.filter is not None:
            if not self.filter(label):
                return self.to_type(None)

        if self.map is not None:
            label = self.map(label)

        return self.to_type(label)
    
    def get_labels(self):
        return self.labels
        
    @property
    def shape(self):
        return self.output_shape


class LabelsFile(Labels):

    def __init__(self, labels: str = "labels", extension: str = ".json",
                 output_shape: int = None, filter: Callable = None, map: Callable = None) -> None:
        
        if not os.path.exists(labels):
            raise ValueError("Path labels not found")

        super().__init__(labels, output_shape, filter, map)

        self.set_extension(extension)

    def get_extension(self):
        return self.extension

    def set_extension(self, extension):

        self.extension = extension

    def get(self, data: Data): 

        if isinstance(data, Image) or isinstance(data, File):
            if f"{data.name_file}{self.extension}" in listdir(data.path_data):
                return self.to_type(join(data.path_data, f"{data.name_file}{self.extension}"))
        else:
            return super().get(data)
        
    def to_type(self, label):
        if isinstance(label, Label):
            return label
        else:
            return LabelF(label)

    def get_files(self, path: str = None):
        if path is not None:
            if not isdir(path):
                raise ValueError(f"Path {path} is not a directory")
        else:
            path = self.labels

        for root, _, files in walk(path):
            for file in files:
                if not file.endswith(self.extension):
                    continue

                yield self.to_type(join(root, file))

    def __len__(self):
        return len(list(self.get_files()))


class LabelsPolygon(LabelsFile):

    def __init__(self, labels: Union[str, Callable, List, Label] = "labels", extension: str = ".json",
                 output_shape: tuple = None, filter: Callable = None, map: Callable = None, round: bool = True) -> None:
        
        super().__init__(labels, extension, output_shape, filter, map)

        self.round = round

    def to_type(self, label):
        if isinstance(label, LabelP):
            return label
        else:
            return LabelP(label, round=self.round)
        
    def __getitem__(self, key: Any) -> Any:
        label: LabelP = super().__getitem__(key)

        if label.get() and isinstance(key, Image):
            if not label.resize_flag:
                label.resize(key.get_image().shape, key.shape)
            if key.rotate_flag:
                label.rotate(-90, key.shape[0]//2, key.shape[1]//2)

        label = self.process_label(label)

        self.buffer[key] = label

        return label