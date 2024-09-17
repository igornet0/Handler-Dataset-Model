import cv2
from os import listdir
from os.path import join, isdir, exists
from types import FunctionType as function
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .Label import *
from .Data import *

class Dataset:

    def __init__(self, data: (str | list), labels = None, output_shape: int = None, shuffle: bool = False, 
                 split: bool = True, test_size: float = 0.2):

        if not isinstance(labels, Labels):
            path = False
            if "/" in labels:
                if not exists(labels):
                    raise Exception(f"Labels path not found: {labels}")
                
                path = True

            labels = Labels(labels, path=path, output_shape=output_shape)

        self.data = Data(data)
        self.labels = labels

        self.shuffle = shuffle
        self.split = split
        self.test_size = test_size


    def generator_data(self) -> iter:
        if self.labels:
            yield from self.get_data_label()
        else:
            for data in self:
                yield data

    def get_label(self, data: Data) -> Label:
        return self.labels[data].get_label()

    def get_data_label(self):

        for data, label in zip(self, self.labels):
            label = label.get()
            if label is None:
                continue

            yield data, label


    def create_data(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
                self.generator_data,
                output_signature=(
                    tf.TensorSpec(shape=self.shape),
                    tf.TensorSpec(shape=self.labels.shape),
                ),
            )
        
        return ds
    
    def __iter__(self):
        yield from self.data.get_data()
    
    def get_ds(self):
        if self.split:
            ds = self.create_data()
            train_size = int(len(self) * (1 - self.test_size))

            train_ds = ds.take(train_size)
            test_ds = ds.skip(train_size)
            return train_ds, test_ds
        
        return self.create_data()

    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self.data.shape


class DatasetImage(Dataset):

    def __init__(self, data_path: str, labels: (str | function | list | Label) = "labels/", 
                 extension: str = ".png", 
                 desired_size = (500, 500, 3), rotate: bool = False,
                 shuffle: bool = False, split: bool = True, test_size: float = 0.2):
        
        super().__init__(data=None, labels=labels, output_shape=desired_size, shuffle=shuffle, split=split, test_size=test_size)

        self.data_path = data_path

        self.desired_size = desired_size
        self.extension = extension
        self.rot = rotate

    
    def get_files(self, path: str, files_only: bool = True):
        files = listdir(path)

        return filter(lambda x: x.endswith(self.extension), files) if files_only else files
    
    def gen_image(self, path_file: str):
        for file in self.get_files(path_file):
            yield join(path_file, file)
    
    def get_path_images(self):
        files = self.get_files(self.data_path, False)
        buffer_gen = {}
        for image_file in files:
            path_data = join(self.data_path, image_file)
            if isdir(path_data):
                buffer_gen[path_data] = self.gen_image(path_data)
            else:
                yield path_data

        if buffer_gen:
            buffer_size = 5
            while True:
                for path_file, gen in buffer_gen.items():
                    for _ in range(buffer_size):
                        try:
                            yield next(gen)
                        except StopIteration:
                            buffer_gen[path_data] = self.gen_image(path_file)

    def get_data_from_path(self):
        for path_file in self.get_path_images():
            image = cv2.imread(path_file)

            if image is None:
                continue
            
            yield Image(image, path_file, self.desired_size, self.rot)

    def gen_rotate(self, data: Image):
        for _ in range(4):
            image = data.get_data()
            label = self.get_label(data).get()
            if label is None:
                continue

            yield image, label
        
        self.labels.clear_buffer()

    def get_data_label(self):
        for data in self:
            if self.rot:
                try:
                    yield from self.gen_rotate(data)
                except StopIteration:
                    pass
                continue

            image = data.get_data()
            label = self.get_label(data).get()
            if label is None:
                continue

            yield image, label

    def __iter__(self):
        yield from self.get_data_from_path()


    def get_output_shape(self):
        return self.desired_size
    

    def get_col_files(self):
        files_col = 0
        for file in self.get_files(self.data_path, False):
            if isdir(join(self.data_path, file)):
                files_col += len(list(self.get_files(join(self.data_path, file))))
            else:
                if not file.endswith(self.extension):
                    continue
                files_col += 1

        return files_col


    def __len__(self):
        files_col = self.get_col_files()
        if self.rot:
            files_col *= 4

        return files_col

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