import json, random, math
from os.path import join
from os import listdir
from types import FunctionType as function

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

class LabelFile:

    def __init__(self, path_files: str):
        self.path = path_files
        self.extension = path_files.split(".")[-1]
        
        self.label = self.get_file_data()

    def round_point(self, point):
        return [round(p, 2) for p in point]
    
    def get_file_data(self) -> list:

        if not self.path.endswith(self.extension):
            return None
        
        with open(self.path) as file:
            data = json.load(file)
            
        points = data.get("shapes", [{}])[0].get("points", data.get("points", []))
        if not points:
            return None
        
        label = [coord for point in points for coord in self.round_point(point)]

        return label

    def get_label(self) -> list:
        return self.label

    def resize_points(self, shape, shape_new):
        new_points = []
        points = np.array(self.label)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = int(point[0] * (shape_new[0] / shape[0])) 
            new_y = int(point[1] * (shape_new[1] / shape[1])) 
            new_points.extend([new_x, new_y])

        self.label = new_points
        return self.label

    def back_coordinates(self, width, height):
        new_points = []
        points = np.array(self.label)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = point[0] * width
            new_y = point[1] * height
            new_points.extend([new_x, new_y])

        self.label = new_points
        return self.label

    def rotate_polygon(self, angle_degrees, center_x, center_y):

        points = np.array(self.label)
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
        return self.label
    

class Labels:

    def __init__(self, labels: (str | function | list | Label) = "labels", path: bool = False, output_shape: int = None):
        
        self.labels = labels
        self.path = path
        self.args = None

        self.buffer = {}

        self.output_shape = output_shape

    def get_labels_from_path(self):
        for labels_file in listdir(self.labels):
            yield LabelFile(labels_file, self.path, extension=labels_file.split('.')[-1])
    
    
    def searh_label_path(self, name_file: str):
        for labels_file in listdir(self.labels):
            if labels_file.split('.')[0] == name_file.split('.')[0]:
                return join(self.labels, labels_file)
            
    def clear_buffer(self):
        self.buffer.clear()

    def __getitem__(self, item):
        self.args = item        
        return self

    def __iter__(self):
        if self.path:
            yield from self.get_labels_from_path()
        else:
            if isinstance(self.labels, list):
                for label in self.labels:
                    if isinstance(label, Label) or isinstance(label, LabelFile):
                        yield label
                    else:
                        yield Label(label)
            else:
                return self.get_label()

    def get_label(self) -> Label:
        if not self.args is None:
            item = self.args
            if isinstance(item, Image) and self.path:
                label: LabelFile = self.buffer.setdefault(item.path_data, 
                                                        LabelFile(self.searh_label_path(item.image_file)))
                
                if item.resize:
                    label.resize_points(item.size, item.desired_size)
            
                if item.rotate:
                    label.rotate_polygon(-90, item.desired_size[0]/2, item.desired_size[1]/2)

                return label
            
        if isinstance(self.labels, function):
            return Label(self.labels(self.args))


    @property
    def shape(self):
        return self.output_shape
