from Dataset import *
from Model import *
from main import get_classes

import os
import shutil

def test_model_classification(model: (Model | str), path_dataset_test="test"):

    label_dict = get_classes(path_dataset_test)
    print(label_dict)
    labels = Labels(lambda x: label_dict[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(label_dict))
    
    if not isinstance(model, Model):
        model_new = ModelClassification(name_model="ModelClassification250.keras")
        model_new.load_model(model)
        model = model_new

    dataset_train = DatasetImage(path_dataset_test, labels=labels, desired_size=(250, 250, 3))
    labels = get_classes(path_dataset_test, one_hot=False)

    batch_size = len(dataset_train)
    for i, image in zip(range(batch_size), dataset_train):
        label = dataset_train.get_label(image).get()

        map = lambda x: x.index(max(x))
        image_p = image.get_data()
        pred = model.predict(image_p, map)

        if pred[0] != map(label):
            if not labels[map(label)] in os.listdir("cash"):
                os.mkdir(os.path.join("cash", labels[map(label)]))
            shutil.move(image.path_data, os.path.join("cash", labels[map(label)], image.image_file))
            # dataset_train.show_img(image.get_image(), label)
            # print(image.path_data)
            # print(model.predict(image_p), label)
            # print(labels[pred[0]], labels[map(label)])

        if i % (batch_size // 10) == 0:
            print(f"{i}/{batch_size}")
