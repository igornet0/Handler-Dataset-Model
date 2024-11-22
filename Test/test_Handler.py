import pytest

import sys

sys.path.append('../')

from Model import *
from Dataset import *

import os
from torch import tensor
from torchvision import transforms

class TestHandler_Dataset:

    path_labels = "test_Polygons"

    transform = transforms.Compose([
            transforms.ToTensor()
        ])

    labels = LabelsPolygon(path_labels)
    dataset = DatasetImage(path_labels, labels=labels)
    handler = HandlerDataset(dataset, transform=transform)

    def test_init(self):
        assert len(self.dataset) == 1500
        assert len(self.handler) == 1500
        assert self.labels.labels == self.path_labels

        assert self.labels.extension == ".json"

    def test_get(self):

        for i in range(self.handler):
            data, label = self.handler[i]

            assert isinstance(data, tensor)
            assert isinstance(label, tensor)

    def test_show(self):
        for i in range(self.handler):
            data, label = self.handler[i]
            DatasetImage.show_img(data, label, polygon=True)



if __name__ == "__main__":

    TestHandler_Dataset().test_show()