import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestDataset_list:
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    ds = Dataset(data)

    def test_init(self):

        assert isinstance(self.ds.data, Data)
        assert isinstance(self.ds.labels, Labels)
        assert self.ds.data.data == self.data

    def test_get_files(self):
        files = self.ds.get_files(self.dataset_path, files_only=False)
        assert len(files) == 3
        assert files[0] == '1'
        assert files[1] == '2'
        assert files[2] == '3'


    def test_get_path_images(self):
        paths = list(self.ds.get_path_images())
        assert len(paths) == 3
        assert paths[0] == f'{self.dataset_path}/1/0.jpg'
        assert paths[1] == 'tests/data/1.jpg'
        assert paths[2] == 'tests/data/2.jpg'


    def test_get_data_from_path(self):
        data = self.ds.get_data_from_path()
        assert len(list(data)) == 500
        for d in data:
            assert isinstance(d, Image)


    def test_gen_rotate(self):
        data = self.ds.get_data_from_path()
        assert len(list(data)) == 12
        for d in data:
            assert isinstance(d, Image)


    def test_get_data_label(self):
        data = self.ds.get_data_label()
        assert len(list(data)) == 500
        for d in data:
            assert isinstance(d[0], Image)
            assert isinstance(d[1], Label)


    def test_get_output_shape(self):
        assert self.ds.get_output_shape() == (256, 256, 3)


    def test_get_col_files(self):
        assert self.ds.get_col_files() == 500


    def test_len(self):
        assert len(self.ds) == 500

