import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from Dataset import DatasetImage, Labels
from Model import Model

def get_classes(path):
        def to_one_hot(labels, num_classes):
            one_hot = {}
            for key, value in labels.items():
                one_hot[key] = [0] * num_classes
                one_hot[key][value - 1] = 1  # Уменьшаем на 1 для индексации
            return one_hot
        
        classes = {x: i for i, x in enumerate(os.listdir(path))}
        classes = to_one_hot(classes, len(classes))
        return classes

def train_model_detection(path_dataset_train="train", path_model=None, save_checkpoints=True):
    batch_size = 32
    epochs = 10

    dataset_train = DatasetImage(path_dataset_train, 
                                 labels="train/labels", output_shape=8,
                                 shuffle=True, rotate=True)

    # dataset_test = Dataset(path_dataset_test)

    ds, ds_test = dataset_train.get_ds()
    # ds_test = dataset_test.get_data()

    model = Model(save=save_checkpoints, name_model="ModelDetection.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, ds_test, batch_size=batch_size, epochs=epochs)

def train_model_classification(path_dataset_train="train", path_model=None, save_checkpoints=True):
    batch_size = 32
    epochs = 10
    
    labels = Labels(lambda x: get_classes(path_dataset_train)[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(get_classes(path_dataset_train)))
    
    dataset_train = DatasetImage(path_dataset_train, labels=labels, 
                                 shuffle=True, rotate=True)
    # dataset_test = Dataset(path_dataset_test)

    ds, ds_test = dataset_train.get_ds()
    # ds_test = dataset_test.get_data()

    model = Model(save=save_checkpoints, name_model="ModelClassification.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, ds_test, batch_size=batch_size, epochs=epochs)

def test_model_classification(path_dataset_test="test", path_model=None):
    
    labels = Labels(lambda x: get_classes(path_dataset_test)[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(get_classes(path_dataset_test)))
    
    model = Model(name_model="ModelClassification.keras")
    if path_model is not None:
        model.load_model(path_model)

    dataset_train = DatasetImage(path_dataset_test, labels=labels, shuffle=True, extension=".jpg")
    # dataset_test = Dataset(path_dataset_test)
    ds = dataset_train.get_ds(rotate=True, shape_output=len(get_classes(path_dataset_test)))
    # ds_test = dataset_test.get_data()

    for d, l in ds.as_numpy_iterator():
        pred = model.predict(d)
        print(pred, l)
        dataset_train.show_img(d, l)
        

def main():
    path_model = "checkpoints/ModelDetection.keras"
    path_model = None

    train_model_detection(path_dataset_train="train", path_model=path_model, save_checkpoints=True)
    train_model_classification(path_dataset_train="train_g",save_checkpoints=True)


if __name__ == "__main__":
    main()