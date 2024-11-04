import os

def get_classes(one_hot: bool=True):
        classes = {'Патент': [0, 0, 0, 0, 1], 'Миграционная карта': [1, 0, 0, 0, 0], 
                   'Паспорт': [0, 1, 0, 0, 0], 'Страница КПП': [0, 0, 1, 0, 0], 'Регистрация': [0, 0, 0, 1, 0]}
        
        if not one_hot:
            classes = {x.index(max(x)): path for path, x in classes.items()}

        return classes
        

def main():

    # from train import train_model_classification, train_model_detection, test_model_classification, train_model_polygon
    # from test import test_model_classification
    from train import train_model_classification
    # path_model = "checkpoints/ModelDetection.keras"
    path_model_class = "checkpoints/last_11_ModelClassification.pth"

    # train_model_detection(path_dataset_train="dataset_new", path_model=path_model_class, save_checkpoints=True)
    # train_model_classification(path_dataset_train="cash", path_model=path_model_class)
    # train_model_classification(path_dataset_train="dataset_new", path_model=path_model_class)
    # test_model_classification(path_model_class, path_dataset_test="dataset_new")

    train_model_classification(path_dataset="dataset_new", 
                               path_model=None, 
                               epochs=20,
                                save_checkpoints=True, DEBUG=True)


if __name__ == "__main__":
    main()