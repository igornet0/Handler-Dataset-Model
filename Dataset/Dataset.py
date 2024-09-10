import cv2
from os import listdir
from os.path import join, isdir
from types import FunctionType as function
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .Label import *
from .Data import *

class Dataset:

    def __init__(self, data: (str | function | list), labels: (Label | None) = None, shuffle: bool = False):
        if isinstance(labels, str):
            if not labels in listdir(data):
                raise Exception("Labels not found")
    
            labels = LabelFile(label=None, path_files=join(data, labels))

        elif not isinstance(labels, Label):
            labels = Label(labels)

        self.data = data

        self.shuffle = shuffle
        self.labels = labels

    def generator_data(self, args = {}) -> iter:
        if self.labels is None:
            return self.get_data()
        
        for data in self.get_data():
            
            if isinstance(data, Image):
                image = data
                if args.get("rotate", False):
                    image.rotate = True
                    for _ in range(4):
                        image_np = image.get_data()
                        label = self.labels.get_label(image)
                        yield image_np, label
                    continue

                data = image.get_data()
                label = self.labels.get_label(image)
            else:
                label = self.labels.get_label(data)

            if not label or not data:
                continue
            
            yield data, label


    def create_data(self, shape_input = None, shape_output = None, args = {}):
        ds = tf.data.Dataset.from_generator(
                lambda: self.generator_data(args),
                output_signature=(
                    tf.TensorSpec(shape=shape_input if shape_input else (None,),),
                    tf.TensorSpec(shape=shape_output if shape_output else (None,),),
                )
            )
        
        return ds
    
    def get_ds(self, shape_input = None, shape_output = None, args = {}):
        return self.create_data(shape_input=shape_input, shape_output=shape_output, args=args)

    def get_data_label(self, data_gen, label_gen):
        for data, label in zip(data_gen, label_gen):

            if not label or not data:
                continue

            yield data, label

    def get_data(self):
        if isinstance(self.data, str):
            yield from self.data

        elif isinstance(self.data, function):
            yield from self.data()

        elif isinstance(self.data, list):
            yield from self.data


class DatasetImage(Dataset):

    def __init__(self, data_path: str, labels: (str | function | list | Label) = "labels", 
                 extension: str = ".png", desired_size = (500, 500, 3), 
                 shuffle: bool = False):

        super().__init__(data_path, labels, shuffle)

        self.data_path = data_path

        self.desired_size = desired_size
        self.extension = extension

    def get_ds(self, shape_output = None, rotate: bool = False):
        return super().create_data(shape_input=self.desired_size, shape_output=shape_output, args={"rotate": rotate})
    
    def get_files(self, path: str):
        files = listdir(path)

        if self.shuffle:
            random.shuffle(files)

        return files
    
    def get_path_images(self):
        files = self.get_files(self.data_path)

        for image_file in files:

            self.file_name = image_file
            path_data = join(self.data_path, image_file)
            if isdir(path_data):
                
                for image_file in self.get_files(path_data):
                    yield join(path_data, image_file)

            elif not image_file.endswith(self.extension):
                continue

            else:
                yield path_data
    
    def get_data(self):
        for path_file in self.get_path_images():
            image = cv2.imread(path_file)

            if image is None:
                continue
            
            yield Image(image, self.file_name, path_file, self.desired_size)

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
        

