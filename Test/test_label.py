import pytest

import sys

sys.path.append('../')

from Dataset import Label, LabelF, LabelP

sys.path.clear()

import os

class TestLabel_list:

    label = Label([1, 2, 3])

    def test_get_label(self):
        assert list(self.label.get()) == [1, 2, 3]
    
    def test_shape(self):
        assert self.label.shape == (3, )

class TestLabel_str:

    label = Label("Hello world")

    def test_get_label(self):
        assert list(self.label.get()) == ["Hello world"]
    
    def test_shape(self):
        assert self.label.shape == (1, )

["test.txt", "test.json", "test.csv"]

class TesLabelF_TXT:

    label = LabelF("test.txt")

    def test_get_label(self):
        assert list(self.label.get()) ==["Hello world my name is {user.get_name()}, welcome !"]
    
    def test_shape(self):
        assert self.label.shape == (1, )

class TesLabelF_JSON:

    label = LabelF("test.json")

    def test_get_label(self):
        print(list(self.label.get()))
        assert list(self.label.get()) ==["Hello world my name is {user.get_name()}, welcome !"]
    
    def test_shape(self):
        assert self.label.shape == (1, )