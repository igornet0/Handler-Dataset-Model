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
        if isinstance(self.label, list) or isinstance(self.label, dict):
            return self.label
        elif self.label is None:
            return []
        
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
            path_data = Path(path_files)
        else:
            raise ValueError(f"Path {path_files} does not exist")

        self.path_data = path_data.parent
        self.name_file = path_data.stem
        self.extension = path_data.suffix
        
        super().__init__(self.get_file_data())
    
    def get_file_path(self):
        return os.path.join(self.path_data, f"{self.name_file}{self.extension}")

    def get_file_data(self) -> list:

        file_path = self.get_file_path()

        if self.extension.endswith(".json"):
            with open(file_path) as file:
                data = json.load(file)

            return data.get("label", [])
        
        if self.extension.endswith(".txt"):
            with open(file_path, "r") as file:

                data = file.read().splitlines()
            return data
        
        return None


class LabelP(LabelF):

    def __init__(self, label:dict = None, round: bool = True) -> None:

        self.flag_round = round
        self.resize_flag = False

        if isinstance(label, str):
            super().__init__(label)
        else:
            self.label = label

    def __getitem__(self, key):
        return self.label[key]
    
    def set_polygon(self, label, polygon):
        self.label[label] = polygon

    def get(self) -> dict:
        return super().get()

    def get_file_data(self):

        file_path = self.get_file_path()
        
        with open(file_path) as file:
            data = json.load(file)
            
        shapes = data.get("shapes", [{}])
        labels = {}
        for i, shape in enumerate(shapes):
            points = shape.get("points", [])
            if not points:
                continue
            
            if self.flag_round:
                points = self.round(points)

            labels.setdefault(shape.get("label", i+1), points)

        self.label = labels

        return self.label

    def round(self, points=None):
        if points is None:
            if isinstance(self.label, dict):
                points = self.label.values()
            else:
                points = self.label
        
        return np.round(np.array(points)).flatten().tolist()

    def resize(self, shape, shape_new):
        """
        Resize a label to a new shape.

        Parameters
        ----------
        shape : tuple
            The current shape of the label.
        shape_new : tuple
            The new shape of the label.

        Returns
        -------
        new_points : list
            The resized label.
        """

        for label, points in self.label.items():

            points = np.array(points)
            points = points.reshape((-1, 2))
            new_points = [] 
            for point in points:
                new_x = int(point[0] * (shape_new[0] / shape[0])) 
                new_y = int(point[1] * (shape_new[1] / shape[1])) 
                new_points.extend([new_x, new_y])

            self.label[label] = new_points

        self.flag_resize = True
            

    def back(self, width, height, points=None):
        if points is None:
            points = self.label

        new_points = []
        
        points = np.array(points)
        points = points.reshape((-1, 2))

        for point in points:
            new_x = point[0] * width
            new_y = point[1] * height
            new_points.extend([new_x, new_y])

        return new_points

    def rotate(self, angle_degrees, center_x, center_y):    
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)

        for label, points in self.label.items():

            points = np.array(points)
            points = points.reshape((-1, 2))
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

            self.label[label] = rotated_points
        