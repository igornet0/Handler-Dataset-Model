import os, shutil
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from Dataset import DatasetImage, Labels
from Model import Model

def get_classes(path, one_hot: bool=True):
        classes = {'Патент': [0, 0, 0, 0, 1], 'Миграционная карта': [1, 0, 0, 0, 0], 
                   'Паспорт': [0, 1, 0, 0, 0], 'Страница КПП': [0, 0, 1, 0, 0], 'Регистрация': [0, 0, 0, 1, 0]}
        
        if not one_hot:
            classes = {x.index(max(x)): path for path, x in classes.items()}
        return classes

def train_model_detection(path_dataset_train="train", path_model=None, save_checkpoints=True, batch_size=32, epochs=10):

    dataset_train = DatasetImage(path_dataset_train, 
                                 labels="train/labels", output_shape=8,
                                 shuffle=True, rotate=True)

    ds, ds_test = dataset_train.get_ds()

    model = Model(save=save_checkpoints, name_model="ModelDetection.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, ds_test, batch_size=batch_size, epochs=epochs)

    return model

def train_model_classification(path_dataset_train="train", path_model=None, 
                               save_checkpoints=True, batch_size=32, epochs=10, 
                               test: bool=False):

    label = get_classes(path_dataset_train)
    print(label)

    def get_label(x):
        path = x.path_data.split(os.path.sep)[-2]
        # if path != "Патент":
        #     return None
        # else:
        #     return label[path]

        return label[path]

    labels = Labels(lambda x: get_label(x), 
                    output_shape=len(label))
    
    dataset_train = DatasetImage(path_dataset_train, labels=labels, desired_size=(250, 250, 3),
                                 shuffle=True, rotate=False, split=True)

    model = Model(save=save_checkpoints, name_model="ModelClassification250.keras", classification=True)
    if path_model is not None:
        model.load_model(path_model)
    else:
        model.create_model(dataset_train.desired_size, labels.output_shape)

    model.train(dataset_train, batch_size=batch_size, epochs=epochs)
    
    if test: 
        test_model_classification(model, path_dataset_train)

    return model

def test_model_classification(model: (Model | str), path_dataset_test="test"):

    label_dict = get_classes(path_dataset_test)
    print(label_dict)
    labels = Labels(lambda x: label_dict[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(label_dict))
    
    if not isinstance(model, Model):
        model_new = Model(name_model="ModelClassification250.keras")
        model_new.load_model(model)
        model = model_new

    dataset_train = DatasetImage(path_dataset_test, labels=labels, shuffle=True, desired_size=(250, 250, 3))
    labels = get_classes(path_dataset_test, one_hot=False)
    stat = {"error": {}}
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
    print(stat)            
        

def main():
    path_model = "checkpoints/ModelDetection.keras"
    path_model_class = "ModelClassification250_1.keras"

    # train_model_detection(path_dataset_train="dataset_new", path_model=path_model_class, save_checkpoints=True)
    train_model_classification(path_dataset_train="cash", path_model=path_model_class)
    train_model_classification(path_dataset_train="dataset_new", path_model=path_model_class)
    test_model_classification(path_model_class, path_dataset_test="dataset_new")


if __name__ == "__main__":
    main()