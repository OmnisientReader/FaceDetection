from image_utils import*
from data_utils import*
from model_utils import*
from face_detection import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import zipfile
import multiprocessing
import os
import keras.api.models
from PIL import Image
from io import BytesIO
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.api.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

class Visualization:

    @staticmethod
    def img_show(img_path, coords):

        image = mpimg.imread(img_path)
        image = cv2.resize(image, (400, 400))# Constant values(not checked with others)
        img_height, img_width = image.shape[:2]

        fig, ax = plt.subplots()
        im = ax.imshow(image)

        for coord in coords:
            if coord:
                rect = plt.Rectangle((coord[1], coord[0]), coord[2], coord[2], linewidth=2, edgecolor='r',
                                     facecolor='none')
                ax.add_patch(rect)
                print('ok', coord)

        plt.savefig("my_plot.png")



def train_model(zip_file_path, zip_file_path1, target_dir1, target_dir2):

    images, labels = DataLoading.load_images_from_zip(zip_file_path, target_dir1)
    images1, labels1 = DataLoading.load_images_from_zip(zip_file_path1, target_dir2, label=0)

    imgs = np.array([[[0] * 128] * 128])
    lbls = np.array([])

    for i in range(len(images)):
        imgs = np.append(imgs, images[i:i + 1], axis=0)
        imgs = np.append(imgs, images1[i:i + 1], axis=0)
        lbls = np.append(lbls, labels[i])
        lbls = np.append(lbls, labels1[i])

    imgs = np.delete(imgs, 0, axis=0)

    train_images, test_images, train_labels, test_labels = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    batch_size = 32
    train_dataset = DataLoading.create_tf_dataset(train_images, train_labels, batch_size)

    model = ModelUtils.create_model()
    model.fit(train_dataset, epochs=18)

    model.save("my_model.h5")

    test_generator = DataLoading.data_generator(test_images, test_labels, batch_size)
    loss, accuracy = model.evaluate(test_generator, steps=len(test_images) // batch_size)
    print("Точность на тестовой выборке:", accuracy)


if __name__ == '__main__':
    zip_file_path = '/Images/archive (2).zip'
    zip_file_path1 = '/Images/archive.zip'
    target_dir1 = 'Human Faces Dataset/Real Images'
    target_dir2 = 'data'
    model_path = os.path.join(os.path.dirname(__file__), 'CNNmodels/my_model (1).h5')
    model_path1 = os.path.join(os.path.dirname(__file__), 'CNNmodels/my_model_fastest.h5')
    new_image_path = "NewData/neuro.jpg"

    # Train the model (if needed) uncomment the following line
    # train_model(zip_file_path, zip_file_path1, target_dir1, target_dir2)

    detector = FaceDetector(model_path, model_path1)
    detector.load_models()
    face_coordinates = detector.detect_faces(new_image_path)
    Visualization.img_show(new_image_path, face_coordinates) 


