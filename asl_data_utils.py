import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array

# Remove 'nothing' from CLASS_NAMES
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['delete', 'space']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMG_SIZE = 224

class ASLDataGenerator(Sequence):
    def __init__(self, image_label_list, batch_size=32, augment=True, class_names=None):
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        # Filter image_label_list to only include selected classes
        self.image_label_list = [(img, self.class_map[CLASS_NAMES[label]] if isinstance(label, int) else self.class_map[label])
                                for img, label in image_label_list if (CLASS_NAMES[label] if isinstance(label, int) else label) in self.class_names]
        self.batch_size = batch_size
        self.augment = augment
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_label_list) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.image_label_list[idx * self.batch_size:(idx + 1) * self.batch_size] 
        images, landmarks, labels = [], [], []
        for img_path, label in batch:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Use MediaPipe to detect hand and crop
            results = self.hands.process(img)
            landmark_vec = np.zeros(21 * 3, dtype=np.float32)  # 21 landmarks, (x, y, z)
            if results.multi_hand_landmarks:
                # Get bounding box around hand landmarks
                h, w, _ = img.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        x, y, z = lm.x, lm.y, lm.z
                        coords.extend([x, y, z])
                        xi, yi = int(x * w), int(y * h)
                        if xi < x_min: x_min = xi
                        if yi < y_min: y_min = yi
                        if xi > x_max: x_max = xi
                        if yi > y_max: y_max = yi
                    landmark_vec = np.array(coords, dtype=np.float32)
                # Add some padding
                pad = 10
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                img = img[y_min:y_max, x_min:x_max]
            # Resize and normalize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            if self.augment:
                img = self.augment_image(img)
                landmark_vec = self.augment_landmarks(landmark_vec)
            images.append(img)
            landmarks.append(landmark_vec)
            labels.append(label)

        images = np.array(images)
        landmarks = np.array(landmarks)
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.class_names))
        return {'image': images, 'landmark': landmarks}, labels

    def on_epoch_end(self):
        random.shuffle(self.image_label_list)

    def augment_image(self, img):
        # Rotation, scale, shift, brightness, horizontal flip
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        # Random rotation +/-10 deg
        angle = random.uniform(-10, 10)
        # Random scale 0.9-1.1
        scale = random.uniform(0.9, 1.1)
        # Random shift
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        return img

    def augment_landmarks(self, landmark_vec):
        # Add small Gaussian noise
        if landmark_vec.shape[0] == 63:
            landmark_vec = landmark_vec + np.random.normal(0, 0.01, size=landmark_vec.shape).astype(np.float32)
            # Random horizontal flip (mirror x)
            if random.random() > 0.5:
                for i in range(21):
                    landmark_vec[i*3] = 1.0 - landmark_vec[i*3]  # x = 1-x
        return landmark_vec

def collect_balanced_image_list(data_dir, n_per_class=700, webcam_data_dir=None, webcam_n_per_class=200, class_names=None):
    class_names = class_names if class_names is not None else CLASS_NAMES
    class_map = {name: idx for idx, name in enumerate(class_names)}
    image_label_list = []
    for class_name in class_names:
        # Collect images from main dataset
        class_dir = os.path.join(data_dir, class_name)
        main_imgs = []
        if os.path.isdir(class_dir):
            main_imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        # Collect images from webcam_data if provided
        webcam_imgs = []
        if webcam_data_dir:
            webcam_class_dir = os.path.join(webcam_data_dir, class_name)
            if os.path.isdir(webcam_class_dir):
                webcam_imgs = [os.path.join(webcam_class_dir, f) for f in os.listdir(webcam_class_dir) if f.lower().endswith('.jpg')]
        # Sample from each source
        selected_main = random.sample(main_imgs, min(n_per_class, len(main_imgs))) if main_imgs else []
        selected_webcam = random.sample(webcam_imgs, min(webcam_n_per_class, len(webcam_imgs))) if webcam_imgs else []
        for img_path in selected_main + selected_webcam:
            label = class_name
            image_label_list.append((img_path, class_map[label]))
    random.shuffle(image_label_list)
    return image_label_list

class TripletDataGenerator(Sequence):
    """
    Keras Sequence for generating triplets (anchor, positive, negative) for triplet loss training.
    Robust to missing/corrupt images and supports balanced sampling from main and webcam data.
    Returns dicts with 'image' and 'landmark' for each branch.
    """
    # Remove 'nothing' from SIMILAR_GROUPS
    SIMILAR_GROUPS = [
        ['A', 'E', 'S', 'X'],
        ['C', 'O', 'Q', 'D', 'P'],
        ['F', 'W'],
        ['G', 'H', 'U', 'Z'],
        ['I', 'J', 'Z'],
        ['K', 'V', 'W', 'Y'],
        ['L', 'T'],
        ['M', 'N'],
        ['R', 'U', 'X'],
    ]
    def __init__(self, data_dir, webcam_data_dir=None, batch_size=4, img_size=112, class_names=None, max_retries=10):
        self.data_dir = data_dir
        self.webcam_data_dir = webcam_data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.max_retries = max_retries
        self.class_to_imgs = self._collect_image_paths()
        self.classes = [c for c in self.class_names if len(self.class_to_imgs[c]) > 1]
        self.n = 10000  # Arbitrary large number for __len__
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        self.similar_negatives = self._build_similar_negatives()

    def _collect_image_paths(self):
        class_to_imgs = {c: [] for c in self.class_names}
        for c in self.class_names:
            dirs = [os.path.join(self.data_dir, c)]
            if self.webcam_data_dir:
                dirs.append(os.path.join(self.webcam_data_dir, c))
            for d in dirs:
                if os.path.isdir(d):
                    class_to_imgs[c] += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.jpg')]
        return class_to_imgs

    def _build_similar_negatives(self):
        # Map class name to set of similar classes (excluding itself)
        sim_map = {c: set() for c in self.class_names}
        for group in self.SIMILAR_GROUPS:
            for c in group:
                if c in sim_map:
                    sim_map[c].update([x for x in group if x != c and x in sim_map])
        return sim_map

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):
        anchors_img, positives_img, negatives_img = [], [], []
        anchors_lm, positives_lm, negatives_lm = [], [], []
        for _ in range(self.batch_size):
            for _ in range(self.max_retries):
                anchor_class = random.choice(self.classes)
                # Use similar negatives if available, else any other class
                sim_neg_pool = list(self.similar_negatives.get(anchor_class, []))
                sim_neg_pool = [c for c in sim_neg_pool if c in self.classes and c != anchor_class]
                if sim_neg_pool:
                    neg_class = random.choice(sim_neg_pool)
                else:
                    neg_class = random.choice([c for c in self.classes if c != anchor_class])
                anchor_imgs = self.class_to_imgs[anchor_class]
                neg_imgs = self.class_to_imgs[neg_class]
                if len(anchor_imgs) < 2 or len(neg_imgs) < 1:
                    continue
                a_path, p_path = random.sample(anchor_imgs, 2)
                n_path = random.choice(neg_imgs)
                a_img, a_lm = self._load_img_and_landmarks(a_path)
                p_img, p_lm = self._load_img_and_landmarks(p_path)
                n_img, n_lm = self._load_img_and_landmarks(n_path)
                if (a_img is not None and p_img is not None and n_img is not None and
                    a_lm is not None and p_lm is not None and n_lm is not None):
                    anchors_img.append(a_img)
                    positives_img.append(p_img)
                    negatives_img.append(n_img)
                    anchors_lm.append(a_lm)
                    positives_lm.append(p_lm)
                    negatives_lm.append(n_lm)
                    break
        # Return as dicts for Keras multi-input
        return (
            {
                'image': np.array(anchors_img),
                'landmark': np.array(anchors_lm)
            },
            {
                'image': np.array(positives_img),
                'landmark': np.array(positives_lm)
            },
            {
                'image': np.array(negatives_img),
                'landmark': np.array(negatives_lm)
            }
        ), np.zeros((self.batch_size, 1))

    def _load_img_and_landmarks(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                return None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img)
            landmark_vec = np.zeros(21 * 3, dtype=np.float32)
            if results.multi_hand_landmarks:
                h, w, _ = img.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    landmark_vec = np.array(coords, dtype=np.float32)
                    break
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0
            return img, landmark_vec
        except Exception:
            return None, None
