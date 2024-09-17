import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import numpy as np
from Dataset import Dataset

class Model:

    def __init__(self, name_model: str = "Model.keras", classification: bool = False, save: bool = False):
        self.name_model = name_model
        self.save = save

        self.model = None
        self.classification = classification

    def set_save(self, save: bool):
        self.save = save

    def predict(self, data: np.ndarray, map_f = None):
        if self.model is None:
            raise Exception("Model is not loaded")

        predict = self.model.predict(np.array([data]), verbose=0)  
        if not map_f is None:
            return np.array([map_f(x.tolist()) for x in predict])
        return np.array(predict)

    def load_model(self, path: str = "Model.keras"):
        self.name_model = path
        self.model = tf.keras.models.load_model(path)
    
    def save_model(self):
        print(f"[TRAIN INFO] Model safe in file: {self.name_model}")
        self.model.save(self.name_model)

    def add_callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ]
        return callbacks
    
    def create_model(self, input_shape: tuple = (500, 500, 3), output_shape: int = 1):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(output_shape, activation='softmax' if self.classification else 'linear')
        ])

        self.compile()

        return self.model

    def compile(self):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        if self.classification:
            self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            return 
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def test_model(self, ds_test):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        # оцениваем точность на тестовой выборке
        test_loss, test_mae = self.model.evaluate(ds_test)
        
        print('[TEST INFO] Test loss:', test_loss)
        print('[TEST INFO] Test mae:', test_mae)
    
        return test_loss, test_mae

    def train(self, ds, ds_test = None, epochs=200, batch_size=32):

        steps_per_epoch = 10

        if isinstance(ds, Dataset):
            steps_per_epoch = len(ds)
            if ds.split:
                steps_per_epoch = int(len(ds) * (1 - ds.test_size))
                validation_steps = int(len(ds) * ds.test_size) // batch_size
                ds, ds_test = ds.get_ds()

            else:
                ds = ds.get_ds()
            
            steps_per_epoch //= batch_size

        if self.model is None:
            input_shape = ds.element_spec[0].shape
            output_shape = ds.element_spec[1].shape[0]
            print(f"Input shape: {input_shape}, output shape: {output_shape}")
            self.model = self.create_model(input_shape, output_shape)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        if ds_test is not None:
            callbacks = self.add_callbacks()

            ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            history = self.model.fit(ds, epochs=epochs, 
                                    verbose=1, 
                                    callbacks=callbacks,
                                    validation_data=ds_test,
                                    validation_steps=validation_steps,  
                                    steps_per_epoch=steps_per_epoch)
        else:
            history = self.model.fit(ds, epochs=epochs,  
                                    steps_per_epoch=steps_per_epoch, 
                                    verbose=1)
        if self.save:
            self.save_model() 

        return history
