import cv2
import numpy as np
import os

from typing import Union, Generator

class Data:

    def __init__(self, data: Union[str, Generator], args: tuple = None):
        self.data = data
        self.args = args


    def loop_generator(self, generator: Generator) -> list:
        return [i for i in generator(*self.args if self.args else [])]


    def get(self) -> np.ndarray:
        if isinstance(self.data, Generator):
            self.data = np.array(self.loop_generator(self.data))
        
        if isinstance(self.data, list):
            return np.array(self.data)
        
        if not isinstance(self.data, np.ndarray):
            return self.data
        

    @property
    def shape(self) -> tuple:
        return self.get().shape
    

    def __len__(self):
        return len(self.get())
    
    

class Image(Data):

    def __init__(self, data: Union[str, Generator], path_data: str = None, 
                 desired_size: tuple = None,
                 rotate: bool = False):
        
        """
        Args:
            data (Union[str, Generator]): If str, reads the image from the path.
            path_data (str): Path to the image file.
            desired_size (tuple): Desired shape of the image.
            rotate (bool): Whether to rotate the image by 90 degrees.
        """
        
        if isinstance(data, str):
            path_data = data
            data = cv2.imread(data)
        
        super().__init__(data)

        if desired_size is None:
            desired_size = data.shape
        elif len(desired_size) != len(data.shape):
            raise Exception(f"Desired shape {desired_size} does not match data shape {data.shape}")

        if not path_data is None:
            self.path_data = path_data
            self.image_file = path_data.split(os.path.sep)[-1]
            self.extension = f".{self.path_data.split('.')[-1]}"
        else:
            self.image_file = "image.png"
            self.extension = ".png"
            self.path_data = self.image_file
            
        self.desired_size = desired_size
        self.rotate_flag = rotate
        

        self.resize_flag = False


    def get_image(self) -> np.ndarray:
        return self.data


    def resize(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, self.desired_size[:2])
        return image
    
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image / 255
    

    def rotate(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    
    def get(self) -> np.ndarray:
        if self.data.shape != self.desired_size:
            image = self.resize(self.data)
            self.resize_flag = True
        else:
            self.resize_flag = False
            image = self.data

        if self.rotate_flag:
            image = self.rotate(image)
        
        return self.normalize(image)
    
    @property
    def shape(self):
        return self.desired_size
    

    def flatten(self):
        return self.data.flat()


    def __len__(self):
        if isinstance(self.data, np.ndarray):
            return len(self.data.flatten())
        
        return len(self.data)
    

class File(Data):

    def __init__(self, data:str):
        
        if os.path.exists(data):
            data = os.path.abspath(data)
        else:
            raise ValueError(f"Path {data} does not exist")

        super().__init__(data)
        

    def get_file_data(self, path: str) -> Generator:
        with open(path) as file:
            for line in file.read().splitlines():
                yield line


    def get(self) -> list:
        return list(self.get_file_data(self.data))
