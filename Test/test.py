import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestDatasetImage_Classification:

    dataset_path = "test_datasets/test_Classification"
    labels_class = {'0': [1, 0, 0], '1': [0, 1, 0], '2': [0, 0, 1]}
    labels = Labels(lambda x: {'0': [1, 0, 0], '1': [0, 1, 0], '2': [0, 0, 1]}[x.path_data.split(os.path.sep)[-2]], output_shape=3)
    ds = DatasetImage(dataset_path, labels, extension = '.jpg', rotate=False, split=True, test_size=0.2, desired_size=(256, 256, 3))

    def test_init(self):
        
        assert self.ds.data_path == self.dataset_path
        assert self.ds.desired_size == (256, 256, 3)
        assert self.ds.extension == '.jpg'
        assert self.ds.rot == False
        assert self.ds.split == True
        assert self.ds.test_size == 0.2

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

