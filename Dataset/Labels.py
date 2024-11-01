from os.path import join, isdir
from os import listdir
from types import FunctionType as function
from typing import Callable, List, Union

from .Data import *
from .Label import *

class Labels:

    def __init__(self, labels, output_shape: int = None, filter: Callable = None, map: Callable = None) -> None:

        self.labels = labels
        self.args = None

        self.buffer = {}

        self.output_shape = output_shape

        self.filter = filter
        self.map = map
            

    def clear_buffer(self):
        self.buffer.clear()


    def __getitem__(self, item):
        self.args = item        
        return self
    

    def __iter__(self):
        if isinstance(self.labels, list):
            for label in self.labels:
                yield self.process_label(label)
        else:
            yield from self.get_label()

    def to_type(self, label) -> Label:
        if isinstance(label, Label):
            return label
        else:
            return Label(label)
        
    def process_label(self, label):
        label = self.to_type(label)

        if self.filter is not None:
            label = self.filter(label.get())

        if self.map is not None:
            label = self.map(label.get())

        return self.to_type(label)
    
    def get_labels(self):
        return self.labels
        
    def get_label(self) -> Label:
        if isinstance(self.labels, function):
            return Label(self.labels(self.args))

        label = self.labels
        label = self.process_label(label)

        if isinstance(label, Label):
            return label
        else:
            return Label(label)

    @property
    def shape(self):
        return self.output_shape

    def __len__(self):
        if self.labels is None:
            return 0
        
        return len(self.labels)


class LabelsFile(Labels):

    def __init__(self, labels: Union[str, Callable, List, Label] = "labels", extension: str = ".json",
                 output_shape: int = None, filter: Callable = None, map: Callable = None) -> None:
        
        super().__init__(labels, output_shape, filter, map)

        self.extension = extension


    def to_type(self, label):
        if isinstance(label, LabelF):
            return label
        else:
            return LabelF(label)

    def searh_label_path(self, name_file_search: str):
        if not f"{name_file_search.split('.')[0]}{self.extension}" in listdir(self.labels):
            for dir in listdir(self.labels):
                if not isdir(join(self.labels, dir)):
                    continue
                if f"{name_file_search.split('.')[0]}{self.extension}" in listdir(join(self.labels, dir)):
                    return join(self.labels, dir, f"{name_file_search.split('.')[0]}{self.extension}")
            return None
        return join(self.labels, f"{name_file_search.split('.')[0]}{self.extension}")


    def get_files(self, path: str, files_only: bool = True):
        
        files = listdir(path)

        return filter(lambda x: x.endswith(self.extension), files) if files_only else files


    def gen_buffer(self, buffer_gen: dict):
        for path_data, gen in buffer_gen.items():
            while True:
                try:
                    yield join(path_data, next(gen))
                except StopIteration:
                    break


    def get_path(self, getbuffer: bool = False):
        files = self.get_files(self.labels, False)
        buffer_gen = {}
        for image_file in files:
            path_data = join(self.labels, image_file)
            if isdir(path_data):
                buffer_gen[path_data] = self.get_files(path_data)
            else:
                yield path_data

        if buffer_gen and not getbuffer:
            yield from self.gen_buffer(buffer_gen)
        else:
            return buffer_gen
        

    def get_label_from_path(self):
        for path_file in self.get_path():
            yield LabelF(path_file)


    def __iter__(self):
        for label in self.get_label_from_path():
            label = self.process_label(label)
            if label is None:
                continue

            yield self.to_type(label)


    def get_label(self) -> LabelF:

        if self.args is not None:
            item = self.args
            if isinstance(item, Image):
                label: LabelF = self.buffer.setdefault(item.path_data, 
                                                            LabelF(self.searh_label_path(item.image_file)))

                return label
        else:
            return next(self.get_label_from_path())


    def __len__(self):
        return len(list(self.get_label_from_path()))


class LabelsPolygon(LabelsFile):

    def __init__(self, labels: Union[str, Callable, List, Label] = "labels", extension: str = ".json",
                 output_shape: tuple = None, filter: Callable = None, map: Callable = None) -> None:
        
        super().__init__(labels, extension, output_shape, filter, map)

        self.extension = extension


    def get_label_from_path(self):
        for path_file in self.get_path():
            yield LabelP(path_file)


    def to_type(self, label):
        if isinstance(label, LabelP):
            return label
        else:
            return LabelP(label)
        

    def __iter__(self):
        if isinstance(self.labels, str):
            yield from super().__iter__()
        else:
            for label in self.labels:
                label = self.process_label(label)
                if label is None:
                    continue

                yield self.to_type(label)
        

    def get_label(self) -> LabelP:
        
        if self.args is not None:
            item = self.args
            if isinstance(item, Image):
                label: LabelP = self.buffer.setdefault(item.path_data, 
                                                    LabelP(self.searh_label_path(item.image_file)))
                if item.resize:
                    label.resize(label.get(), item.size, item.desired_size)
            
                if item.rotate:
                    label.rotate(-90, item.desired_size[0]/2, item.desired_size[1]/2)

            return label
        else:
            return next(self.get_label_from_path())