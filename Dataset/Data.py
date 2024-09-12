import cv2
import numpy as np
import tensorflow as tf
import os
from types import FunctionType as function

class Data:

    def __init__(self, data: (str | list)):
        self.data = data

    def get_data(self):
        if isinstance(self.data, list):
            yield from self.data

        else:
            return self.data

    @property
    def shape(self):
        return np.array(list(self.get_data())).shape
    
    

class Image(Data):

    def __init__(self, data: np.ndarray, path_data: str, 
                 desired_size: tuple = (500, 500, 3),
                 rotate: bool = False):
        
        super().__init__(data)

        self.path_data = path_data
        self.image_file = path_data.split(os.path.sep)[-1]
        self.size = data.shape
        self.desired_size = desired_size
        self.rotate = rotate
        self.extension = f".{self.path_data.split('.')[-1]}"

        self.resize = False


    def resize_image(self, image):
        image = cv2.resize(image, self.desired_size[:2])
        return image
    
    def normalize_image(self, image):
        return image / 255
    
    def rotate_image(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def get_data(self):
        if not self.desired_size is None and self.data.shape != self.desired_size:
            image = self.resize_image(self.data)
            self.resize = True
        else:
            self.resize = False
            image = self.data

        if self.rotate:
            image = self.rotate_image(image)
            self.data = image
        
        return self.normalize_image(image)
    
    @property
    def shape(self):
        return self.desired_size
    
    def __len__(self):
        return self.shape