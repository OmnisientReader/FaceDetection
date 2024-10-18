import numpy as np
import zipfile
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
import tensorflow as tf
from image_utils import ImageUtils

class DataLoading:

    @staticmethod
    def load_images_from_zip(zip_path, target_directory, valid_extensions=['.jpg', '.png'], label=1,
                              number_of_img=4000):
        image_paths = []
        labels = []
        counter = 0

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(tuple(valid_extensions)) and file_info.filename.startswith(
                        target_directory) and not (file_info.filename.startswith('data/natural_images/person')):

                    if counter == number_of_img:
                        break

                    counter += 1

                    with zip_ref.open(file_info) as file:
                        image = ImageUtils.load_img_pillow(BytesIO(file.read()), target_size=(128, 128), grayscale=True)
                        image_array = ImageUtils.img_to_array(image) / 255.0
                        image_paths.append(image_array)
                        labels.append(label)

        return np.array(image_paths), np.array(labels)

    @staticmethod
    def create_tf_dataset(images, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def data_generator(images, labels, batch_size):
        while True:
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                yield np.array(batch_images), np.array(batch_labels)

