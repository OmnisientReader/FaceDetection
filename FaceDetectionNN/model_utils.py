from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.api.regularizers import l2
import tensorflow as tf
import keras
import numpy as np


class ModelUtils:

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(128, 128, 1), kernel_regularizer=l2(0.1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (7, 7), activation='relu', kernel_regularizer=l2(0.1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.8))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod    
    @keras.saving.register_keras_serializable()
    def downsample(image):
        return tf.image.resize(image, (32, 32))

    @staticmethod
    def predict_face(img_part, model):
        prediction = model.predict(np.expand_dims(img_part, axis=0))
        return prediction[0][0] > 0.6