import cv2
import numpy as np

from types import FunctionType as function

class Data:

    def __init__(self, data: (str | function | list )):
        self.data = data

    def get_data(self):
        if isinstance(self.data, function):
            yield from self.data()

        elif isinstance(self.data, list):
            yield from self.data

        else:
            return self.data
        
    def __iter__(self):
        return self.get_data()

class Image(Data):

    def __init__(self, data: np.ndarray, name_file: str = None, 
                 path_data: str = ".", desired_size: tuple = (500, 500, 3),
                 rotate: bool = False):
        
        super().__init__(data)

        self.name_file = name_file
        self.path_data = path_data

        self.size = data.shape
        self.desired_size = desired_size
        self.resize = False
        self.rotate = rotate
        self.extension = f".{self.name_file.split('.')[-1]}"


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