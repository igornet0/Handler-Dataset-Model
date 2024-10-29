import json, math

import numpy as np

from .Data import *
import os

class Label:

    def __init__(self, label):
        if label is None:
            raise ValueError("Label is None")

        self.label = label
                 
    def get(self) -> list:
        if isinstance(self.label, list):
            return self.label
        
        return [self.label]
    
    @property
    def shape(self):
        return np.array(self.get()).shape

    def __len__(self):
        if not isinstance(self.label, list):
            return 0
        
        return len(self.label)


class LabelF(Label):

    def __init__(self, path_files: str) -> None:
        if os.path.exists(path_files):
            path_files = os.path.abspath(path_files)
        else:
            raise ValueError(f"Path {path_files} does not exist")

        self.path = path_files
        self.extension = path_files.split(".")[-1]
        
        super().__init__(self.get_file_data())


    def get_file_data(self) -> list:

        if self.extension == "json":
            with open(self.path) as file:
                data = json.load(file)
            return data.get("label", [])
        
        if self.extension == "txt":
            with open(self.path) as file:
                data = file.read().splitlines()
            return data
        
        return None



class LabelP(Label):

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
    