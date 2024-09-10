import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from Dataset import DatasetImage, Label
from Model import Model

def train_model_detection(path_dataset_train="train", path_dataset_test="test", path_model=None, save_checkpoints=True):
    batch_size = 32
    epochs = 10

    dataset_train = DatasetImage(path_dataset_train, labels="labels", shuffle=True)
    # dataset_test = Dataset(path_dataset_test)
    for image, l in dataset_train.generator_data({"rotate": True}):
        dataset_train.show_img(image, l, polygon=True)
    exit(0)
    ds = dataset_train.get_ds()
    # ds_test = dataset_test.get_data()

    model = Model(save=save_checkpoints, name_model="ModelDetection.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, batch_size=batch_size, epochs=epochs)

def train_model_classification(path_dataset_train="train", path_dataset_test="test", path_model=None, save_checkpoints=True):
    batch_size = 32
    epochs = 10

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
    
    labels = Label(lambda x: get_classes(path_dataset_train)[x.name_file])
    dataset_train = DatasetImage(path_dataset_train, labels=labels, shuffle=True, extension=".jpg")
    # dataset_test = Dataset(path_dataset_test)

    ds = dataset_train.get_ds(rotate=True, shape_output=len(get_classes(path_dataset_train)))
    # ds_test = dataset_test.get_data()

    model = Model(save=save_checkpoints, name_model="ModelClassification.keras")
    if path_model is not None:
        model.load_model(path_model)

    model.train(ds, batch_size=batch_size, epochs=epochs)


def main():
    path_model = "checkpoints/ModelDetection.keras"

    # train_model_detection(path_dataset_train="train", path_dataset_test="test", path_model=path_model, save_checkpoints=True)
    train_model_classification(path_dataset_train="dataset_new",save_checkpoints=True)


if __name__ == "__main__":
    main()