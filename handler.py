from Dataset import DatasetImage, Labels, Image
from Model import Model

import os, shutil
from sys import argv


def choise_doit(data: Image, pred_path: str, cash_path: None):
    print("1 - next\n2 - move to pred\n3 - move to cash\n4 - delete")
    match int(input(">>>")):
        case 1:
            return True
        case 2:
            shutil.move(data.path_data, pred_path)
        case 3:
            if not cash_path is None:
                shutil.move(data.path_data, cash_path)
        case 4:
            os.remove(data.path_data)

def handler_dataset_label_image(dataset: DatasetImage, model: Model, dop_map = lambda x: x):

    for _, data in zip(range(len(dataset)), dataset):
        map = lambda x: x.index(max(x))
        image_p = data.get_data()
        pred = model.predict(image_p, map)
        
        if not dataset.labels is None:
            label = dataset.get_label(data).get()

            if pred[0] != map(label):
                print(model.predict(image_p), label)
                dataset.show_img(data.get_image(), label)
                dop_map(data, pred[0], map(label))
        else:
            dataset.show_img(data.get_image(), pred[0])

def handler_dataset_image(dataset: DatasetImage, model: Model, dop_map = lambda x: x):
    for _, data in zip(range(len(dataset)), dataset):
        map = lambda x: x.index(max(x))
        image_p = data.get_data()
        pred = model.predict(image_p, map)
        dop_map(data, pred[0])


def main(path_dataset_test):
    from main import get_classes

    labels_dict = get_classes(path_dataset_test)

    labels = Labels(lambda x: labels_dict[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(labels_dict))

    dataset_train = DatasetImage(path_dataset_test, labels=labels, shuffle=True, desired_size=(250, 250, 3), extension=".jpg")
    labels = get_classes(path_dataset_test, one_hot=False)

    model = Model()
    model.load_model("ModelClassification250.keras")

    def pred_label(x, p, l):
        print(f"[INFO] pred={labels[p]} labels={labels[l]}")
        cash_path = "cash.1"

        if not labels[p] in os.listdir(cash_path):
            os.mkdir(os.path.join(cash_path, labels[p]))
        if not labels[l] in os.listdir(cash_path):
            os.mkdir(os.path.join(cash_path, labels[l]))

        choise_doit(x, os.path.join(cash_path, labels[p], x.image_file), 
                    cash_path=os.path.join(cash_path, labels[l], x.image_file))


    handler_dataset_image(dataset_train, model, pred_label)

if __name__ == "__main__":
    main(argv[1])