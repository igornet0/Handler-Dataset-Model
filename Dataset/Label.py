import json, random, math
from os.path import join
from os import listdir
from types import FunctionType as function

import numpy as np

from .Data import *

class Label:

    def __init__(self, label: (str | function | list)):
        self.label = label
                 
    def get_label(self, *args, **kwargs) -> any:
        if isinstance(self.label, function):
            result = self.label(*args, **kwargs)
        elif isinstance(self.label, list):
            return self.label
        
        if isinstance(result, list):
            return result

        return [result]

    def get_labels(self) -> iter:
        for label in self.get_label():
            yield label

    def __iter__(self):
        return self.get_labels()

class LabelFile(Label):

    def __init__(self, label: str, path_files: str = "train", extension: str = ".json"):
        super().__init__(label)
        
        self.path = path_files
        self.extension = extension

    def round_point(self, point):
        return [round(p, 2) for p in point]
    
    def get_file_data(self, label_file: str) -> list:

        if not label_file.endswith(self.extension) or not label_file in listdir(self.path):
            return None
        
        with open(join(self.path, label_file)) as file:
            data = json.load(file)
            
        points = data.get("shapes", [{}])[0].get("points", data.get("points", []))
        if not points:
            return None
        
        label = [coord for point in points for coord in self.round_point(point)]

        return label

    def get_label(self, data: (str | Image), col_rotate: int = 1) -> list:
        image = None

        if isinstance(data, Image):
            image = data
            label_file = f"{image.name_file.split('.')[0]}{self.extension}"
        else:
            label_file = data
        
        if self.label is None:
            self.label = self.get_file_data(label_file)

        label = self.label

        if isinstance(image, Image):
            if image.resize:
                self.label = self.resize_points(self.label, image.size, image.desired_size)
            
            if image.rotate:
                for _ in range(col_rotate):
                    label = self.rotate_polygon(label, -90, image.desired_size[0]/2, image.desired_size[1]/2)
                col_rotate += 1

        return label
    
    def get_labels(self, shuffle: bool = False) -> iter:
        files = listdir(self.path)

        if shuffle:
            random.shuffle(files)

        for label_file in files:
            yield self.get_label(label_file)
            self.label = None

    def resize_points(self, points, shape, shape_new):
        new_points = []
        points = np.array(points)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = int(point[0] * (shape_new[0] / shape[0])) 
            new_y = int(point[1] * (shape_new[1] / shape[1])) 
            new_points.extend([new_x, new_y])

        return new_points

    def back_coordinates(self, points, width, height):
        new_points = []
        points = np.array(points)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = point[0] * width
            new_y = point[1] * height
            new_points.extend([new_x, new_y])

        return new_points

    def rotate_polygon(self, points, angle_degrees, center_x, center_y):
        points = np.array(points)
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
        
        return rotated_points