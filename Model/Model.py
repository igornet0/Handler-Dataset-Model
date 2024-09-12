import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import numpy as np

class Model:

    def __init__(self, name_model: str = "Model.keras", save: bool = False):
        self.name_model = name_model
        self.save = save

        self.model = None

    def predict(self, data: np.ndarray):
        if self.model is None:
            raise Exception("Model is not loaded")

        return self.model.predict(data)    

    def load_model(self, path: str = "Model.keras"):
        self.model = tf.keras.models.load_model(path)
    
    def save_model(self):
        print(f"[TRAIN INFO] Model safe in file: {self.name_model}")
        self.model.save(self.name_model)

    def add_checkpoint(self):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/" + self.name_model,  # Имя файла для сохранения
            monitor='val_mae',  # Отслеживаемая метрика
            save_best_only=True,  # Сохраняем только лучшую модель
            mode='min',  # Сохраняем при минимальном значении
            verbose=1  # Выводим информацию о сохранении
        )

        return checkpoint
    
    def create_model(self, input_shape: tuple = (500, 500, 3), output_shape: int = 1):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(output_shape, activation='linear')
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return self.model
    
    def test_model(self, ds_test):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        # оцениваем точность на тестовой выборке
        test_loss, test_mae = self.model.evaluate(ds_test)
        
        print('[TEST INFO] Test loss:', test_loss)
        print('[TEST INFO] Test mae:', test_mae)
    
        return test_loss, test_mae

    def train(self, ds, ds_test = None, epochs=200, batch_size=32):

        if self.model is None:
            input_shape = ds.element_spec[0].shape
            output_shape = ds.element_spec[1].shape[0]
            print(f"Input shape: {input_shape}, output shape: {output_shape}")
            self.model = self.create_model(input_shape, output_shape)

        chekpoint = self.add_checkpoint()

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        if ds_test is not None:
            ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            history = self.model.fit(ds, epochs=epochs, 
                                    steps_per_epoch=10,
                                    verbose=1, 
                                    batch_size=batch_size, callbacks=[chekpoint],
                                    validation_data=ds_test)
        else:
            history = self.model.fit(ds, epochs=epochs, 
                                    steps_per_epoch=10,
                                    verbose=1, 
                                    batch_size=batch_size)
        if self.save:
            self.save_model() 

        return history
