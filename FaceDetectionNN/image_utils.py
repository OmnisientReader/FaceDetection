import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from io import BytesIO

class ImageUtils:

    @staticmethod
    def load_img_pillow(path_or_bytes, target_size=None, grayscale=False):
        try:
            if isinstance(path_or_bytes, str):
                img = Image.open(path_or_bytes)
            elif isinstance(path_or_bytes, BytesIO):
                img = Image.open(path_or_bytes)
            else:
                raise TypeError("path_or_bytes must be a string (file path) or BytesIO object.")

            if grayscale:
                img = img.convert('L')
            if target_size:
                img = img.resize(target_size)
            return img
        except FileNotFoundError:
            print(f"Error: Image file not found at {path_or_bytes}")
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def load_img(image_path, size=(128, 128)):
        try:
            img_pil = Image.open(image_path)
            img_resized_pil = img_pil.resize(size)
            img_resized = np.array(img_resized_pil)
            img_grayscale = np.mean(img_resized, axis=2) / 255.0 if len(img_resized.shape) == 3 else img_resized
            return img_grayscale
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def img_to_array(img):
        if img is None:
            return None
        return np.array(img)

    @staticmethod
    def split_image_opencv(img, square_size, step=15):
        height, width = img.shape[:2]
        image_parts = []

        for y in range(0, height - square_size + 1, step):
            for x in range(0, width - square_size + 1, step):
                cropped = img[y:y + square_size, x:x + square_size].copy()
                downsampled = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
                image_parts.append({
                    'image_part': downsampled,
                    'y': y,
                    'x': x,
                    'size': square_size
                })

        return image_parts
