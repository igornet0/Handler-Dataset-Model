import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestDatasetImage:
    path_dataset = "test_Classification"

    ds = DatasetImage(path_dataset, extension=".jpg", desired_size=(256, 256, 3), 
                      rotate=False, test_size=0.2)

    def test_init(self):

        assert self.ds.data_path == self.path_dataset


    def test_get_path_images(self):
        paths = list(self.ds.get_path_images())
        
        assert len(paths) == 3
        assert paths[0] == '1'
        assert paths[1] == '2'
        assert paths[2] == '3'


    def test_get_data_from_path(self):
        data = list(self.ds.get_data_from_path())
        assert len(data) == 1500
        for d in data:
            assert isinstance(d, Image)


    def test_gen_rotate(self):
        self.ds.set_rotate(True)
        data = list(self.ds.get_data())
        assert len(data) == 1500 * 4
        for d in data:
            assert isinstance(d, Image)

        self.ds.set_rotate(False)


    def test_get_data_label(self):
        classes = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1]}

        labels = Labels(lambda x: classes[x.path_data], 
                        output_shape=3)
        
        self.ds.labels = labels
        data = list(self.ds.get_data_label())

        assert len(data) == 1500
        for d in data:
            assert classes[d[0].path_data] == d[1].get()
            assert isinstance(d[0], Image)
            assert isinstance(d[1], Label)



    def test_get_output_shape(self):
        assert self.ds.get_output_shape() == (256, 256, 3)


    def test_get_col_files(self):
        assert self.ds.get_col_files() == 1500


    def test_len(self):
        assert len(self.ds) == 1500

    
    def test_get_bath(self):
        data = list(self.ds.get_bath())
        assert len(data[0]) == 32
        for bath in data:
            for d, l in bath:
                assert isinstance(d, Image)
                assert isinstance(l, Label)

