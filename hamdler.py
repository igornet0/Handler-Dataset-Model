from Dataset import DatasetImage, Labels, Image
from Model import Model

import os, shutil

def choise_doit(data: Image, pred_path: str):
    print("1 - next\n2 - move to pred\n3 - delete")
    match int(input(">>>")):
        case 1:
            return True
        case 2:
            shutil.move(data.path_data, pred_path)
        case 3:
            os.remove(data.path_data)

def handler_dataset_image(dataset: DatasetImage, model: Model, dop_map = lambda x: x):

    for i, data in zip(range(len(dataset)), dataset):
        label = dataset.get_label(data).get()

        map = lambda x: x.index(max(x))
        image_p = data.get_data()
        pred = model.predict(image_p, map)

        if pred[0] != map(label):
            dataset.show_img(data.get_image(), label)
            print(model.predict(image_p), label)
            dop_map(data, pred[0])


def main(path_dataset_test):
    from main import get_classes

    labels_dict = get_classes(path_dataset_test)

    labels = Labels(lambda x: labels_dict[x.path_data.split(os.path.sep)[-2]], 
                    output_shape=len(labels_dict))

    dataset_train = DatasetImage(path_dataset_test, labels=labels, shuffle=True, desired_size=(250, 250, 3))
    labels = get_classes(path_dataset_test, one_hot=False)

    model = Model()
    model.load_model("ModelClassification250.keras")

    handler_dataset_image(dataset_train, model, lambda x, p: choise_doit(x, os.path.join(path_dataset_test, labels[p], x.image_file)))

if __name__ == "__main__":
    main()