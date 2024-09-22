import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

from math import ceil

import numpy as np
from Dataset import Dataset

class Model:

    def __init__(self, name_model: str = "Model.keras", save: bool = False):
        self.name_model = name_model
        self.save = save

        self.model = None

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
    
    def create_model(self, input_shape: tuple = (500, 500, 3)):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten()
        ])

        self.compile()

        return self.model

    def compile(self):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, ds, ds_test = None, epochs=200, batch_size=32):

        steps_per_epoch = 10

        if isinstance(ds, Dataset):
            steps_per_epoch = len(ds)
            if ds.split:
                steps_per_epoch = len(ds) * (1 - ds.test_size)
                validation_steps = ceil((len(ds) * ds.test_size) / batch_size) - 1
                ds, ds_test = ds.get_ds()

            else:
                ds = ds.get_ds()
            
            steps_per_epoch = ceil(steps_per_epoch / batch_size) - 1

        if self.model is None:
            input_shape = ds.element_spec[0].shape
            output_shape = ds.element_spec[1].shape[0]
            print(f"Input shape: {input_shape}, output shape: {output_shape}")
            self.model = self.create_model(input_shape, output_shape)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print("[TRAIN INFO] Start training")
        print(f"[TRAIN INFO] Steps per epoch: {steps_per_epoch}")

        if ds_test is not None:
            print(f"[TRAIN INFO] Validation steps: {validation_steps}")
            callbacks = self.add_callbacks()

            ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            history = self.model.fit(ds, epochs=epochs, 
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

class ModelClassification(Model):

    def __init__(self, name_model: str = "ModelClassification.keras", save: bool = False):
        super().__init__(name_model, save)

    def create_model(self, input_shape: tuple = (500, 500, 3), output_shape: int = 1):
        super().create_model(input_shape, output_shape)

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation='softmax'))

        return self.model

    def compile(self):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

class ModelPolygons(Model):

    def __init__(self, name_model: str = "ModelPolygons.keras", save: bool = False):
        super().__init__(name_model, save)

    def create_model(self, input_shape=(500, 500, 3), num_anchors=9, num_classes=1):
        inputs = Input(shape=input_shape)

        # Слой свертки 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Слой свертки 2
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Слой свертки 3
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Слой свертки 4
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Выходной слой
        # num_anchors * (num_classes + 5) включает координаты (x, y, w, h), объектность и классы
        num_outputs = num_anchors * (num_classes + 5)
        x = Conv2D(num_outputs, (1, 1), strides=(1, 1), padding='same')(x)

        # Преобразование выходов
        output_shape = (-1, (input_shape[0] // 8) * (input_shape[1] // 8) * num_anchors, num_classes + 5)
        outputs = Reshape(output_shape)(x)

        self.model = Model(inputs, outputs)
        return self.model

    def compile(self):
        if self.model is None:
            raise Exception("Model is not loaded")
        
        optimizer = Adam(learning_rate=1e-4)
    
        # Функция потерь — простая для классификации и детекции
        # Можно сделать кастомную функцию потерь для YOLO (включающую ошибки координат)
        loss = BinaryCrossentropy(from_logits=True)  # Используем BinaryCrossentropy для объектности

        # Компиляция
        self.model.compile(optimizer=optimizer, loss=loss)

        return self.model
    
    def add_callbacks(self):
        # Коллбэк для сохранения лучших весов
        checkpoint = ModelCheckpoint(
            f'BestModel{self.name_model}',      # Имя файла для сохранения
            monitor='val_loss',   # Мониторим потерю на валидационных данных
            save_best_only=True,  # Сохраняем только лучшие веса
            mode='min',           # Сохраняем веса при минимальном значении val_loss
            verbose=1
        )
        
        # Коллбэк для ранней остановки
        early_stopping = EarlyStopping(
            monitor='val_loss',   # Мониторим потерю на валидационных данных
            patience=10,          # Количество эпох для ожидания улучшений
            mode='min',           # Останавливаем при минимальном значении val_loss
            verbose=1
        )
        
        return [checkpoint, early_stopping]