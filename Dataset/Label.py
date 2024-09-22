import json, random, math
from os.path import join, isdir, isfile
from os import listdir
from types import FunctionType as function
from typing import Callable, List, Union

import numpy as np

from .Data import *

class Label:

    def __init__(self, label):
        self.label = label
                 
    def get(self) -> list:
        if isinstance(self.label, list):
            return self.label
        
        if self.label is None:
            return self.label
        
        return [self.label]

    def __len__(self):
        if not isinstance(self.label, list):
            return 0
        return len(self.label)


class LabelFile(Label):

    def __init__(self, path_files: str) -> None:
        self.path = path_files
        self.extension = path_files.split(".")[-1]
        
        super().__init__(self.get_file_data())


    def get_file_data(self) -> list:

        if not self.path.endswith(self.extension):
            return None
        
        if self.extension == "json":
            with open(self.path) as file:
                data = json.load(file)
            return data.get("label", [])
        
        if self.extension == "txt":
            with open(self.path) as file:
                data = file.read().splitlines()
            return data
        
        return None



class LabelPolygon(Label):

    def __init__(self, label = None) -> None:
        if isinstance(label, str):
            label = self.get_file_data(label)

        super().__init__(label)


    def get_file_data(self, path_file: str):
        if path_file is None:
            return None
        
        with open(path_file) as file:
            data = json.load(file)
            
        shapes = data.get("shapes", [{}])
        labels = []
        for shape in shapes:
            points = shape.get("points", [])
            if not points:
                continue
            label = [coord for point in points for coord in self.round(point)]
            labels.append(label)

        self.label = labels

        return self.label
    

    def round(self, label):
        if isinstance(label[0], list):
            label = np.array(label).reshape((-1, 2))
            return [self.round(point) for point in label]
        
        return [round(x) for x in label]


    def resize(self, label, shape, shape_new):
        new_points = []
        if isinstance(label[0], list):
            labels = np.array(label)
            return labels.map(lambda x: self.resize(x, shape, shape_new))

        points = np.array(label)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = int(point[0] * (shape_new[0] / shape[0])) 
            new_y = int(point[1] * (shape_new[1] / shape[1])) 
            new_points.extend([new_x, new_y])

        self.label = new_points
        return new_points


    def back(self, width, height):
        new_points = []
        
        points = np.array(self.label)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = point[0] * width
            new_y = point[1] * height
            new_points.extend([new_x, new_y])

        self.label = new_points
        return self.label


    def rotate(self, label, angle_degrees, center_x, center_y):
        if isinstance(label[0], list):
            labels = np.array(label)
            return labels.map(lambda x: self.rotate(x, angle_degrees, center_x, center_y))
        
        points = np.array(label)
        points = points.reshape((-1, 2))

        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)

        rotated_points = []

        for x, y in points:
            # Смещение к началу координат
            x -= center_x
            y -= center_y
            
            # Поворот
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            
            new_x += center_x 
            new_y += center_y
        
            new_x = round(new_x, 2)
            new_y = round(new_y, 2)
            rotated_points.extend([new_x, new_y])
        
        self.label = rotated_points
        return rotated_points    
    
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


    def to_type(self, label):
        if isinstance(label, Label):
            return label
        else:
            return Label(label)


    def __getitem__(self, item):
        self.args = item        
        return self
    

    def __iter__(self):
        if isinstance(self.labels, list):
            for label in self.labels:
                yield self.to_type(label)

        elif isinstance(label, function):
            return Label(label(self.args))
        else:
            return self.get_label()

    def process_label(self, label):
        if self.filter is not None:
            label = self.filter(label)

        if self.map is not None:
            label = self.map(label)

        return label
        
    def get_label(self) -> Label:
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
        return len(self.labels)


class LabelsFile(Labels):

    def __init__(self, labels: Union[str, Callable, List, Label] = "labels", extension: str = ".json",
                 output_shape: int = None, filter: Callable = None, map: Callable = None) -> None:
        
        super().__init__(labels, output_shape, filter, map)

        self.extension = extension


    def to_type(self, label):
        if isinstance(label, LabelFile):
            return label
        else:
            return LabelFile(label)

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
            yield LabelFile(path_file)


    def __iter__(self):
        for label in self.get_label_from_path():
            label = self.process_label(label)
            if label is None:
                continue

            yield self.to_type(label)


    def get_label(self) -> LabelFile:

        if self.args is not None:
            item = self.args
            if isinstance(item, Image):
                label: LabelFile = self.buffer.setdefault(item.path_data, 
                                                            LabelFile(self.searh_label_path(item.image_file)))

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
            yield LabelPolygon(path_file)


    def to_type(self, label):
        if isinstance(label, LabelPolygon):
            return label
        else:
            return LabelPolygon(label)
        

    def __iter__(self):
        if isinstance(self.labels, str):
            yield from super().__iter__()
        else:
            for label in self.labels:
                label = self.process_label(label)
                if label is None:
                    continue

                yield self.to_type(label)
        

    def get_label(self) -> LabelPolygon:
        
        if self.args is not None:
            item = self.args
            if isinstance(item, Image):
                label: LabelPolygon = self.buffer.setdefault(item.path_data, 
                                                    LabelPolygon(self.searh_label_path(item.image_file)))
                if item.resize:
                    label.resize(label.get(), item.size, item.desired_size)
            
                if item.rotate:
                    label.rotate(label.get(), i-90, item.desired_size[0]/2, item.desired_size[1]/2)

            return label
        else:
            return next(self.get_label_from_path())