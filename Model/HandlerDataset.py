import torch
from torch.utils.data import Dataset as _Dataset
import numpy as np

from Dataset import Dataset

class HandlerDataset(_Dataset):

    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.data = []
        self.labels = []

        self.generater_data = self.start_loader(dataset.get_loader())

    def start_loader(self, loader: iter):
        for data in loader:
            if len(data) == 2:
                data, label = [x.get() for x in data]
                yield data, label
            else:
                data = data.get()
                yield data
    
    def retart_loader(self):
        self.generater_data = self.start_loader(self.dataset.get_loader())

    def update_data(self, data):
        self.data.append(data)
        return self.data[-1]

    def update_label(self, label):
        self.labels.append(label)
        return self.labels[-1]


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        if len(self.data) <= idx:
            try:
                data = next(self.generater_data)
            except StopIteration:
                self.retart_loader()
                data = next(self.generater_data)

            if len(data) == 2:
                label = self.update_label(torch.tensor(data[1]).float())
                data = self.update_data(data[0])
            else:
                data = self.update_data(data)
        else:
            data = self.data[idx]
            label = self.labels[idx]

        
        if self.transform:
            data = self.transform(data).float()
        
        return data, label





    