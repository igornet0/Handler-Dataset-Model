import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestDataset_list:
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    labels = ["a", "b", "c"]
    labels = Labels(labels)
    
    ds = Dataset(data, labels)

    def test_init(self):

        assert isinstance(self.ds.data, Data)
        assert isinstance(self.ds.labels, Labels)
        assert self.ds.data.data == self.data
        assert self.ds.test_size == 0.2

    def test_get_data(self):

        data = self.ds.get_data()
        assert len(list(data)) == 3

        for d, data in zip(data, self.data):
            assert d == data
            assert isinstance(d, Data)

    @pytest.mark.skipif(labels is None, reason="Labels not found")
    def test_get_label(self):

        label = self.ds.get_label(self.data[0])
        assert isinstance(label, Label)
        assert label.get() == self.labels[0]

    @pytest.mark.skipif(labels is None, reason="Labels not found")
    def test_get_data_label(self):

        data = self.ds.get_data_label()
        assert len(list(data)) == 3
        for d, l in data:
            assert d == data
            assert isinstance(d, Data)
            assert isinstance(l, Label)

    def test_get_bath(self):

        data = self.ds.get_bath()
        assert len(list(data)) == 3
        for d, l in data:
            assert d == data
            assert isinstance(d, Data)
            assert isinstance(l, Label)