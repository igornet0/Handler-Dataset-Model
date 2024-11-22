import cv2
import numpy as np
import os
from pathlib import Path
import torch

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
    
    def __iter__(self):
        yield from self.get()
    
    

class Image(Data):

    def __init__(self, data: Union[str, Generator], args: tuple = None, 
                 desired_size: tuple = ()):
        
        """
        Args:
            data (Union[str, Generator]): If str, reads the image from the path.
            path_data (str): Path to the image file.
            desired_size (tuple): Desired shape of the image.
            rotate (bool): Whether to rotate the image by 90 degrees.
        """

        filter_extension = lambda x: x.endswith(".jpg") or x.endswith(".png") or x.endswith(".jpeg")

        if isinstance(data, str):
            if not filter_extension(data):
                raise ValueError(f"File {data} does not have the extension .jpg, .png or .jpeg")
            
            if not os.path.exists(data):
                raise ValueError(f"Path {data} does not exist")
            
            path_data = Path(data)
            data = cv2.imread(data)
        
        super().__init__(data, args)

        if not desired_size:
            desired_size = data.shape

        elif len(desired_size) != len(data.shape):
            raise Exception(f"Desired shape {desired_size} does not match data shape {data.shape}")

        self.path_data = path_data.parent
        self.name_file = path_data.stem
        self.extension = path_data.suffix
            
        self.desired_size = desired_size
        
        self.resize_flag = False
        self.rotate_flag = False


    def get_image(self) -> np.ndarray:
        return self.data


    def resize(self) -> np.ndarray:
        self.data = cv2.resize(self.data, self.desired_size[:2])
        self.resize_flag = True
        return self.data
    
    
    def normalize(self) -> np.ndarray:
        if max(self.flatten()) > 1:
            self.data = self.data / 255
        
        return self.data
    
    def rotate(self):
        self.data = cv2.rotate(self.data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.rotate_flag = True
        return self.data
    
    def get(self) -> torch.Tensor:
        if self.data.shape != self.desired_size:
            self.resize()
        
        return self.normalize()
    
    @property
    def shape(self):
        return self.desired_size
    

    def flatten(self):
        return self.data.flat


    def __len__(self):
        if isinstance(self.data, np.ndarray):
            return len(self.data.flatten())
        
        return len(self.data)
    

class File(Data):

    def __init__(self, data:str):
        
        if not os.path.exists(data):
            raise ValueError(f"Path {data} does not exist")
            
        path_data = Path(data)

        super().__init__(path_data)
        

    def get_file_data(self, path: str) -> Generator:
        with open(path) as file:
            for line in file.read().splitlines():
                yield line


    def get(self) -> list:
        return list(self.get_file_data(self.data))
