import pytest

import sys

sys.path.append('../')

from Dataset import Data, Image, File

sys.path.clear()

import os

class TestData_list:

    data = Data([1, 2, 3])

    def test_get_data(self):
        assert list(self.data.get()) == [1, 2, 3]
    
    def test_shape(self):
        assert self.data.shape == (3, )

class TestData_list_Generator:

    def generator(data):
        for i in data:
            yield i

    data = Data(generator, [1, 2, 3])

    def test_get_data(self):
        assert self.data.get().tolist() == [1, 2, 3]
    
    def test_shape(self):
        assert self.data.shape == (3, )

class TestData_str:

    data = Data("Hello world")

    def test_get_data(self):
        assert self.data.get().tolist() == ["Hello world"]
    
    def test_shape(self):
        assert self.data.shape == (1, )

class TestData_List_str:

    data = Data(["Hello world", "My name is John"])

    def test_get_data(self):
        assert self.data.get().tolist() == ["Hello world", "My name is John"]
    
    def test_shape(self):
        assert self.data.shape == (2, )

class TestImage_str:

    image_file = "Test_data/test.jpg"
    data = Image(image_file, desired_size=(50, 50, 3))
    image = data.get_image()

    def test_get_data(self):
        assert (self.data.get() == self.data.normalize(self.data.resize(self.image))).all()

    def test_get_image(self):
        assert (self.data.get_image() == self.image).all()
    
    def test_resize_image(self):
        assert self.data.resize(self.image).shape == (50, 50, 3)
    
    def test_shape(self):
        assert self.data.shape == (50, 50, 3)

class TestImage_list:
    image_file = "Test_data/test.jpg"
    data = Image(image_file, desired_size=(50, 50, 3))
    image = data.get_image()
    data = Image(image)

    def test_get_data(self):
        assert (self.data.get() == self.data.normalize(self.image)).all()

    def test_get_image(self):
        assert (self.data.get_image() == self.image).all()
    
    def test_resize_image(self):
        assert self.data.resize(self.image).shape == (256, 256, 3)
    
    def test_shape(self):
        assert self.data.shape == (256, 256, 3)

    def test_len(self):
        assert len(self.data) == 256 * 256 * 3


class TestFile_str:

    data = File("Test_data/test.jpg")

    def test_get_data(self):
        assert self.data.get().tolist() == ['Hello world my name is {user.get_name()}, welcome !']
    
    def test_get_path(self):
        assert str(self.data) == "Test_data/test.jpg"