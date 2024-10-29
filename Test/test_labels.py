import pytest

import sys

sys.path.append('../')

from Dataset import *

import os

class TestLabels_list:
    labels = ["a", "b", "c"]
    
    labels = Labels(labels)