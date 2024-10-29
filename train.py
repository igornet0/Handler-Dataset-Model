from Dataset import *
from Model import *
from main import get_classes

from test import *

import os

def train_model_detection(path_dataset_train, path_model=None, save_checkpoints=True, batch_size=32, epochs=10) -> Model:

    dataset_train = DatasetImage(path_dataset_train, 
                                 labels="train/labels", output_shape=8,
                                 split=True, rotate=True)

    ds, ds_test = dataset_train.get_ds()

    model = Model(save=save_checkpoints, name_model="ModelDetection.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, ds_test, batch_size=batch_size, epochs=epochs)

    return model

def train_model_polygon(path_dataset_train, path_model=None, save_checkpoints=True, batch_size=32, epochs=10) -> Model:

    labels = LabelsPolygon(path_dataset_train)

    dataset_train = DatasetImage(path_dataset_train, 
                                 labels=labels,
                                 split=True, rotate=True)

    ds, ds_test = dataset_train.get_ds()
    for _, data in zip(range(5), ds):
        print(data)

    if isinstance(path_model, Model):
        model = path_model
        model.set_save(save_checkpoints)
    else:
        model = ModelPolygons(save=save_checkpoints, name_model="ModelPolygon.keras")
        if path_model is not None:
            model.load_model(path_model)

    model.train(dataset_train, batch_size=batch_size, epochs=epochs)

    return model

def train_model_classification(path_dataset_train, path_model=None, 
                               save_checkpoints=True, batch_size=32, epochs=10, 
                               test: bool=False) -> Model:

    label = get_classes(path_dataset_train)
    print(label)

    def get_label(x):
        path = x.path_data.split(os.path.sep)[-2]
        return label[path]

    labels = Labels(lambda x: get_label(x), 
                    output_shape=len(label))
    
    dataset_train = DatasetImage(path_dataset_train, labels=labels, desired_size=(250, 250, 3),
                                 rotate=False, split=True)

    model = ModelClassification(save=save_checkpoints, name_model="ModelClassification250.keras")
    if path_model is not None:
        model.load_model(path_model)
    else:
        model.create_model(dataset_train.desired_size, labels.output_shape)

    model.train(dataset_train, batch_size=batch_size, epochs=epochs)

    if test: 
        test_model_classification(model, path_dataset_train)

    return model