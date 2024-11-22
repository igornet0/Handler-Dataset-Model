import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestLabels_list:
    labels = {
        '0': [1, 0, 0],
        '1': [0, 1, 0],
        '2': [0, 0, 1]
    }
    
    labels = Labels(labels, output_shape=3, filter=lambda x: x, map=lambda x: x)

    def test_init(self):
        assert self.labels.output_shape == 3

        assert self.labels.labels == {
                                        '0': [1, 0, 0],
                                        '1': [0, 1, 0],
                                        '2': [0, 0, 1]
                                    }
        
        for data in range(3):
            assert self.labels[str(data)].get() == self.labels.labels[str(data)]


class TestLabels_files:

    path_labels = "test_Polygons"

    labels = LabelsFile(path_labels)
    dataset = DatasetImage(path_labels, labels=labels)

    def test_init(self):
        assert self.labels.labels == self.path_labels

        assert self.labels.extension == ".json"
    
    def test_get_files(self):
        assert len(list(self.labels.get_files())) == 1500

    def test_get(self):

        for data in self.dataset:
            assert len(data) == 2
            d, l = data

            assert isinstance(d, Image)
            assert isinstance(l, LabelF)

class TestLabels_polygons:

    path_labels = "test_Polygons"

    labels = LabelsPolygon(path_labels)
    dataset = DatasetImage(path_labels, labels=labels)

    def test_init(self):
        assert self.labels.labels == self.path_labels

        assert self.labels.extension == ".json"
    
    def test_get_files(self):
        assert len(list(self.labels.get_files())) == 1500

    def test_get(self):

        for data in self.dataset:
            assert len(data) == 2
            d, l = data

            assert isinstance(d, Image)
            assert isinstance(l, LabelP)

            points = l.get()
            assert len(points) == 8
            