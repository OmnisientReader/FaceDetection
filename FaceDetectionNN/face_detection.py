from image_utils import*
from model_utils import*
from data_utils import*

class FaceDetector:

    def __init__(self, model_path, model_path1):
        self.model_path = model_path
        self.model_path1 = model_path1
        self.model = None
        self.model1 = None

    def load_models(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = keras.models.load_model(self.model_path)
            self.model1 = keras.models.load_model(self.model_path1)
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def detect_faces(self, image_path):
        n_img = Image.open(image_path)
        n_img = n_img.resize((400, 400))
        new_image_array = ImageUtils.load_img(image_path, size=n_img.size)

        part_sizes = [_ for _ in range(50, 300, 20)]
        img_parts = []
        for part_size in part_sizes:
            parts = ImageUtils.split_image_opencv(new_image_array, part_size)
            img_parts.extend(parts)

        face_coords = []

        for img_part in img_parts:
            img_tensor = tf.convert_to_tensor(img_part['image_part'], dtype=tf.float32)

            if ModelUtils.predict_face(img_tensor, self.model1) and ModelUtils.predict_face(img_tensor , self.model):
                face_coords.append([img_part['y'], img_part['x'], img_part['size']])
        return face_coords
