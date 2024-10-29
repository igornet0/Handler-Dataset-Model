import pytest

import sys

sys.path.append('../')

from Dataset import Data, Image

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

    data = Data(generator([1, 2, 3]))

    def test_get_data(self):
        assert list(self.data.get()) == [1, 2, 3]
    
    def test_shape(self):
        assert self.data.shape == (3, )

class TestData_str:

    data = Data("Hello world")

    def test_get_data(self):
        assert list(self.data.get()) == ["Hello world"]
    
    def test_shape(self):
        assert self.data.shape == (1, )

class TestData_List_str:

    data = Data(["Hello world", "My name is John"])

    def test_get_data(self):
        assert list(self.data.get()) == ["Hello world", "My name is John"]
    
    def test_shape(self):
        assert self.data.shape == (2, )

class TestImage_str:

    image_file = "test.jpg"
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

    image_file = "test.jpg"
    data = Image(image_file)
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