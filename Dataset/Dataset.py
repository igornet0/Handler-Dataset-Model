import random
from os import listdir, walk
from os.path import join, isdir, exists
from typing import Union, Iterable
from torch.utils.data import Dataset as DataLoader
import matplotlib.pyplot as plt

from .Label import *
from .Labels import *
from .Data import *

class Dataset:

    def __init__(self, data: Union[str, Iterable], labels: Labels = None, output_shape: int = None, 
                 test_size: float = 0.2):

        """
        :param data: Path to the folder containing the data or a list of data
        :param labels: Path to the folder containing the labels or a list of labels
        :param output_shape: Shape of the output data
        :param test_size: float indicating the proportion of the data to be used for testing
        """
        
        if not isinstance(labels, Labels):
            labels = Labels(labels, output_shape=output_shape)

        self.data = Data(data)
        self.labels = labels
        self.test_size = test_size

    
    def get_data(self) -> iter:
        return self.data


    def get_label(self, data: Data) -> Label:
        return self.labels[data].get_label()


    def get_data_label(self) -> iter:
        if self.labels.get_labels() is None:
            raise ValueError("Labels not found")

        for data in self.get_data():

            label = self.get_label(data)

            if label is None:
                continue

            yield data, label


    def get_bath(self, batch_size: int = 32, shuffle: bool = True) -> iter:
        bath = []
        
        for data in self:
            bath.append(data)
            if len(bath) == batch_size:
                if shuffle:
                    random.shuffle(bath)
                yield bath
                bath = []

        return bath if not shuffle else random.shuffle(bath)
    
    def get_loader(self):
        if self.labels.get_labels() is not None:
            dataset = self.get_data_label()
        else:
            dataset = self.get_data()
        
        for data in dataset:
            yield data

    def __iter__(self) -> iter:
        yield from self.get_loader()

    def __len__(self):
        return len(self.data)
    

    @property
    def shape(self):
        return self.data.shape


class DatasetImage(Dataset):

    def __init__(self, data_path: str, labels: Labels = None, 
                 extension: Union[str, set] = ".png", 
                 desired_size: tuple = None, rotate: bool = False,
                 shuffle_path: bool = False, test_size: float = 0.2):

        if not os.path.exists(data_path):
            raise ValueError("Path data not found")
        
        super().__init__(data=None, labels=labels, output_shape=desired_size, test_size=test_size)

        self.data_path = data_path

        self.set_desired_size(desired_size)
        self.set_extension(extension)
        self.set_rotate(rotate)
        self.set_shuffle_path(shuffle_path)

    def set_desired_size(self, desired_size: tuple):
        if not isinstance(desired_size, tuple):
            raise ValueError("desired_size must be a tuple")
        
        self.desired_size = desired_size
    
    def set_extension(self, extension: Union[str, set]):
        if not isinstance(extension, str) and not isinstance(extension, set):
            raise ValueError("extension must be a string or a set")
        
        self.extension = set(extension) if isinstance(extension, str) else extension

    def set_rotate(self, rotate: bool):
        if not isinstance(rotate, bool):
            raise ValueError("rotate must be a boolean")
        
        self.rotate = rotate
    
    def set_shuffle_path(self, shuffle_path: bool):
        if not isinstance(shuffle_path, bool):
            raise ValueError("shuffle_path must be a boolean")
        
        self.shuffle_path = shuffle_path

    def get_shuffle_path(self):
        return self.shuffle_path
    
    def get_images(self, path: str = None) -> iter:

        if path is not None:
            if not isdir(path):
                raise ValueError(f"Path {path} is not a directory")

            path = path
        else:
            path = self.data_path

        for root, _, files in walk(path):
            for file in files:
                if not any(file.endswith(ext) for ext in self.extension):
                    continue
                yield Image(join(root, file), desired_size=self.desired_size)

    def get_path_images(self):
        for path_data in listdir(self.data_path):
            yield join(self.data_path, path_data)

    def get_data_from_path_shuffle(self) -> iter:
        buffer_path = {path_file: self.get_images(path_file) for path_file in self.get_path_images()}
        while buffer_path:
            for path_file, images in buffer_path.items():
                try:
                    image = next(images)
                except StopIteration:
                    buffer_path.pop(path_file)
                    break

                yield image

    def get_data(self):
        if self.get_shuffle_path():
            dataset = self.get_data_from_path_shuffle()
        else:
            dataset = self.get_images()

        for data in dataset:
            if self.rotate:
                for _ in range(4):
                    data.rotate()
                    yield data

            else:
                yield data

    def get_output_shape(self):
        return self.desired_size
    
    def get_col_files(self):
        return sum([len(files) for _, _, files in walk(self.data_path)])
        
    def __len__(self):
        col = self.get_col_files()
        if self.rotate:
            col *= 4
        
        return col
    
    @staticmethod
    def show_img(img, label=None, polygon = False):
        if polygon:
            x, y = [], []
            for i in range(1, 9):
                if i % 2 == 1:
                    x.append(label[i-1])
                else:
                    y.append(label[i-1])
            for i in range(0, len(x), 2):
                plt.plot(x[i:i+2], y[i:i+2], 'ro-')
            plt.plot([x[0], x[3]], [y[0], y[3]], 'ro-')
            plt.plot([x[1], x[2]], [y[1], y[2]], 'ro-')
        
        if label:
            plt.title(label)

        plt.imshow(img)
        plt.show()
        
    @property
    def shape(self):
        return self.desired_size