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
        for data in self:
            if not isinstance(data, Data):
                continue

            yield data


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


    def get_bath(self, batch_size: int = 32, shuffle: bool = True, test: bool = False) -> iter:
        bath = []

        if self.labels.get_labels() is not None:
            dataset = self.get_data_label()
        else:
            dataset = self.get_data()
        
        for data in dataset:
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
        yield from self.data


    def __len__(self):
        return len(self.data)
    

    @property
    def shape(self):
        return self.data.shape


class DatasetImage(Dataset):

    def __init__(self, data_path: str, labels: Labels = None, 
                 extension: str = ".png", 
                 desired_size: tuple = None, rotate: bool = False, test_size: float = 0.2):
        
        super().__init__(data=None, labels=labels, output_shape=desired_size, test_size=test_size)

        self.data_path = data_path

        self.desired_size = desired_size
        self.extension = extension
        self.rotate = rotate

    def set_rotate(self, rotate: bool):
        if not isinstance(rotate, bool):
            raise ValueError("rotate must be a boolean")
        
        self.rotate = rotate

    def get_images(self, path_files: str) -> iter:
        for file in listdir(path_files):
            if not file.endswith(self.extension):
                continue
            yield join(path_files, file)
    

    def gen_buffer(self, buffer_gen: dict):
        for _, gen in buffer_gen.items():
            while True:
                try:
                    yield next(gen)
                except StopIteration:
                    break

    def get_path_images(self):
        for path_data in listdir(self.data_path):
            yield path_data


    def get_data_from_path(self) -> iter:
        for path_file in self.get_path_images():
            for image_file in self.get_images(join(self.data_path, path_file)):
                
                yield Image(image_file, desired_size=self.desired_size, path_data=path_file)


    def get_data(self):
        for data in super().get_data():
            if self.rotate:
                for _ in range(4):
                    data.rotate()
                    yield data

            else:
                yield data

    
    def get_data_label(self) -> iter:
        if self.labels.get_labels() is None:
            raise ValueError("Labels not found")

        for data in self.get_data():

            label = self.get_label(data)

            if label is None:
                continue

            yield data, label


    def __iter__(self):
        yield from self.get_data_from_path()


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