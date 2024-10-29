import tensorflow as tf
import pandas as pd
import json
import random

import os

test_dataset_structure = {
    "test_Classification/":{
        "1":500,
        "2":500,
        "3":500
    },

    "test_Polygons/":{
        "1":(500, 500, 8),
        "2":(500, 500, 8),
        "3":(500, 500, 8)
    },

    "test_datasets_custem": {
        "1": ["date", "open", "close", "low", "high", "volume"],
        "2": ["date", "open", "close", "low", "high", "volume"],
        "3": ["date", "open", "close", "low", "high", "volume"]
    }
}

def generate_test_dataset(path_dataset_test="test_datasets/", default_structure: dict = None):

    if not os.path.exists(path_dataset_test):
        os.mkdir(path_dataset_test)

    if default_structure is None:
        default_structure = test_dataset_structure

    for path, files in default_structure.items():
        if not os.path.exists(os.path.join(path_dataset_test, path)):
            os.mkdir(os.path.join(path_dataset_test, path))

        for path_file, size in files.items():
            if not os.path.exists(os.path.join(path_dataset_test, path, path_file)):
                os.mkdir(os.path.join(path_dataset_test, path, path_file))
            
            if isinstance(size, tuple):
                for i in range(max(size)):
                    if i < size[0]:
                        data = tf.random.normal((256, 256, 3))
                        tf.keras.preprocessing.image.save_img(os.path.join(path_dataset_test, path, path_file, f"{i}.jpg"), data)
                    if i < size[1]:
                        with open(os.path.join(path_dataset_test, path, path_file, f"{i}.json"), "w") as file:
                            data = {
                                "shapes":[{
                                        "label": f"{x}",
                                        "points": tf.random.normal([size[2]]).numpy().tolist()
                                        } for x in range(1, random.randint(5, 50))]
                                    }
                            json.dump(data, file)

            elif isinstance(size, list):
                data = tf.random.normal((500, len(size)))
                dataframe = pd.DataFrame(data, columns=size)
                dataframe.to_csv(os.path.join(path_dataset_test, path, path_file, f"{path_file}.csv"))  
                
            else:
                for i in range(size):
                    data = tf.random.normal((256, 256, 3))
                    tf.keras.preprocessing.image.save_img(os.path.join(path_dataset_test, path, path_file, f"{i}.jpg"), data)
    
    
if __name__ == "__main__":
    generate_test_dataset()