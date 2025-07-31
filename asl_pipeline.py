# ASL Hand Gesture Recognizer Pipeline
# Author: GitHub Copilot
# Requirements: tensorflow, opencv-python, mediapipe, numpy, matplotlib
#
# 1. Data Preprocessing & Augmentation
# 2. Model Definition (MobileNetV2 Transfer Learning)
# 3. Training Loop
# 4. Real-Time Webcam Inference with MediaPipe

import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from asl_data_utils import ASLDataGenerator, collect_balanced_image_list, CLASS_NAMES, TripletDataGenerator
import random
import argparse

# --- CONFIG ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 28
EPOCHS = 20
DATA_DIR = r'C:\Users\kenjo\ASLModel\asl_alphabet_train\asl_alphabet_train'  # Change to your dataset path
MODEL_PATH = 'asl_mobilenetv2.h5'
EMBEDDING_MODEL_PATH = 'asl_triplet_embedding_model.keras'

# Remove 'nothing' from CLASS_NAMES
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['delete', 'space']  # No 'nothing'

# --- 1. DATA PREPROCESSING & AUGMENTATION ---
def get_data_generators(data_dir, img_size, batch_size, webcam_data_dir='webcam_data', class_names=None):
    image_label_list = collect_balanced_image_list(data_dir, n_per_class=700, webcam_data_dir=webcam_data_dir, class_names=class_names)
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(image_label_list))
    train_list = image_label_list[:split_idx]
    val_list = image_label_list[split_idx:]
    train_gen = ASLDataGenerator(train_list, batch_size=batch_size, augment=True, class_names=class_names)
    val_gen = ASLDataGenerator(val_list, batch_size=batch_size, augment=False, class_names=class_names)
    return train_gen, val_gen

# --- 2. MODEL DEFINITION ---
def build_model(img_size, num_classes):
    # Image input branch
    image_input = Input(shape=(img_size, img_size, 3), name='image')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)

    # Landmark input branch
    landmark_input = Input(shape=(21*3,), name='landmark')
    l = Dense(128, activation='relu')(landmark_input)
    l = Dropout(0.3)(l)

    # Concatenate features
    combined = Concatenate()([x, l])
    combined = Dense(128, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[image_input, landmark_input], outputs=output)
    # Use a lower learning rate for Adam optimizer
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@tf.keras.utils.register_keras_serializable()
def l2_normalize(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis=1)

def build_embedding_model(img_size, embedding_dim=128):
    # Image input branch
    image_input = Input(shape=(img_size, img_size, 3), name='image')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers[:-1]:
        layer.trainable = False
    for layer in base_model.layers[-1:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)

    # Landmark input branch
    landmark_input = Input(shape=(21*3,), name='landmark')
    l = Dense(128, activation='relu')(landmark_input)
    l = Dropout(0.3)(l)

    # Concatenate features
    combined = Concatenate()([x, l])
    combined = Dense(embedding_dim)(combined)
    embedding = Lambda(l2_normalize, output_shape=(embedding_dim,), name='embedding')(combined)
    model = Model(inputs=[image_input, landmark_input], outputs=embedding)
    return model

# Triplet loss function
def triplet_loss(a, p, n, alpha=0.2):
    pos = tf.reduce_sum(tf.square(a - p), axis=1)
    neg = tf.reduce_sum(tf.square(a - n), axis=1)
    return tf.reduce_mean(tf.maximum(pos - neg + alpha, 0.0))

class ValidationDropEarlyStopping(Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.prev_val_acc = None
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if self.prev_val_acc is not None:
            if val_acc < self.prev_val_acc:
                self.wait += 1
            else:
                self.wait = 0
            if self.wait >= self.patience:
                print(f"\nValidation accuracy dropped for {self.patience} consecutive epochs. Stopping training.")
                self.model.stop_training = True
        self.prev_val_acc = val_acc

# --- 3. TRAINING LOOP ---
def train_model(class_names=None):
    train_gen, val_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE, class_names=class_names)
    model = build_model(IMG_SIZE, len(class_names) if class_names else NUM_CLASSES)
    model.summary()
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max')
    val_drop_stop = ValidationDropEarlyStopping(patience=3)
    try:
        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[early_stop, val_drop_stop]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        raise
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def train_triplet_model(data_dir, webcam_data_dir=None, img_size=224, embedding_dim=128, batch_size=4, epochs=5):
    import gc
    import traceback
    triplet_gen = TripletDataGenerator(data_dir, webcam_data_dir, batch_size=batch_size, img_size=img_size)
    embedding_model = build_embedding_model(img_size, embedding_dim)
    # Multi-modal input: each input is a dict with 'image' and 'landmark'
    input_a_img = tf.keras.Input((img_size, img_size, 3), name='image_a')
    input_a_lm = tf.keras.Input((21*3,), name='landmark_a')
    input_p_img = tf.keras.Input((img_size, img_size, 3), name='image_p')
    input_p_lm = tf.keras.Input((21*3,), name='landmark_p')
    input_n_img = tf.keras.Input((img_size, img_size, 3), name='image_n')
    input_n_lm = tf.keras.Input((21*3,), name='landmark_n')
    emb_a = embedding_model({'image': input_a_img, 'landmark': input_a_lm})
    emb_p = embedding_model({'image': input_p_img, 'landmark': input_p_lm})
    emb_n = embedding_model({'image': input_n_img, 'landmark': input_n_lm})
    stacked = Lambda(lambda x: tf.stack(x, axis=1))([emb_a, emb_p, emb_n])
    triplet_model = Model(
        [input_a_img, input_a_lm, input_p_img, input_p_lm, input_n_img, input_n_lm],
        stacked
    )

    def _triplet_loss(y_true, y_pred):
        a, p, n = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        return triplet_loss(a, p, n)

    triplet_model.compile(optimizer='adam', loss=_triplet_loss)
    best_loss = float('inf')
    patience = 2
    wait = 0
    prev_loss = None
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            history = triplet_model.fit(triplet_gen, epochs=1)
            loss = history.history['loss'][0]
            if prev_loss is not None and loss > prev_loss:
                wait += 1
                print(f"Loss increased ({prev_loss:.4f} -> {loss:.4f}), patience: {wait}/{patience}")
                if wait >= patience:
                    print("Loss increased repeatedly, stopping early and saving model.")
                    embedding_model.save('asl_triplet_embedding_model.keras')
                    print('Embedding model saved to asl_triplet_embedding_model.keras')
                    return
            else:
                wait = 0
            prev_loss = loss
            gc.collect()
            tf.keras.backend.clear_session()
            print("Cleared cache after epoch", epoch+1)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving embedding model...")
        embedding_model.save('asl_triplet_embedding_model.keras')
        print('Embedding model saved to asl_triplet_embedding_model.keras')
        return
    except Exception as e:
        print("Exception during training:", e)
        traceback.print_exc()
        embedding_model.save('asl_triplet_embedding_model.keras')
        print('Embedding model saved to asl_triplet_embedding_model.keras')
        return
    embedding_model.save('asl_triplet_embedding_model.keras')
    print('Embedding model saved to asl_triplet_embedding_model.keras')

# --- Visualize Embedding Space ---
def visualize_embeddings(embedding_model, data_dir, webcam_data_dir=None, img_size=112, n_per_class=50, class_names=None, method='tsne'):
    import os
    import cv2
    import random
    import numpy as np
    from sklearn.manifold import TSNE
    import plotly.express as px

    try:
        from umap import UMAP
    except ImportError:
        UMAP = None

    if class_names is None:
        class_names = CLASS_NAMES

    # Collect images and labels
    image_label_list = []
    for class_name in class_names:
        imgs = []
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            imgs += [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        if webcam_data_dir:
            webcam_class_dir = os.path.join(webcam_data_dir, class_name)
            if os.path.isdir(webcam_class_dir):
                imgs += [os.path.join(webcam_class_dir, f) for f in os.listdir(webcam_class_dir) if f.lower().endswith('.jpg')]
        if imgs:
            selected = random.sample(imgs, min(n_per_class, len(imgs)))
            for img_path in selected:
                image_label_list.append((img_path, class_name))

    # Compute embeddings
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    X, labels = [], []
    for img_path, label in image_label_list:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_input = img_resized.astype('float32') / 255.0
        # Extract landmarks
        results = hands.process(img_resized)
        landmark_vec = np.zeros(21 * 3, dtype=np.float32)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                landmark_vec = np.array(coords, dtype=np.float32)
                break
        emb = embedding_model.predict({'image': np.expand_dims(img_input, 0), 'landmark': np.expand_dims(landmark_vec, 0)}, verbose=0)[0]
        X.append(emb)
        labels.append(label)

    X = np.array(X)

    # Dimensionality reduction
    if method == 'umap' and UMAP is not None:
        reducer = UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    # Create interactive plot
    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=labels,
        hover_name=labels,
        labels={'x': 'Dim 1', 'y': 'Dim 2'},
        title=f"ASL Embedding Space ({method.upper()}) - Interactive",
        width=1000,
        height=800
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.show()


# --- 4. REAL-TIME WEBCAM INFERENCE WITH MEDIAPIPE ---
def run_webcam_inference(model, class_names, img_size=224, nothing_threshold=0.7):
    import cv2
    import numpy as np
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    with hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmark_vec = np.zeros(21 * 3, dtype=np.float32)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coords = []
                    h, w, _ = frame.shape
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    landmark_vec = np.array(coords, dtype=np.float32)
                    break  # Only use the first detected hand

                # Predict only when a hand is detected
                img_input = cv2.resize(frame, (img_size, img_size))
                img_input = img_input.astype('float32') / 255.0
                img_input = np.expand_dims(img_input, axis=0)
                landmark_input = np.expand_dims(landmark_vec, axis=0)

                preds = model.predict({'image': img_input, 'landmark': landmark_input})
                class_id = np.argmax(preds[0])
                class_prob = preds[0][class_id]
                if class_prob < nothing_threshold:
                    class_name = 'nothing'
                else:
                    class_name = class_names[class_id]

                # Display prediction
                cv2.putText(frame, f'{class_name}: {class_prob:.2f}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('ASL Hand Gesture Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# --- 4. REAL-TIME WEBCAM INFERENCE WITH CLASSIFIER + TRIPLET REFINEMENT ---
def run_realtime_inference_with_triplet(classifier_model, embedding_model, class_names, known_embeddings, known_labels, img_size=224, conf_threshold=0.7, k=3):
    import numpy as np
    import cv2
    import mediapipe as mp
    from sklearn.metrics.pairwise import cosine_similarity
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                x1, y1 = max(0, int(x_min * w) - 20), max(0, int(y_min * h) - 20)
                x2, y2 = min(w, int(x_max * w) + 20), min(h, int(y_max * h) + 20)
                hand_img = frame[y1:y2, x1:x2]
                if hand_img.size == 0:
                    continue
                # Preprocess for classifier
                input_img = cv2.resize(hand_img, (img_size, img_size))
                input_img = input_img.astype('float32') / 255.0
                input_img = np.expand_dims(input_img, axis=0)
                pred = classifier_model.predict(input_img)
                class_idx = np.argmax(pred)
                conf = pred[0][class_idx]
                label = class_names[class_idx]
                # If confidence is high, accept
                if conf >= conf_threshold:
                    final_label = label
                    method = 'Classifier'
                else:
                    # Use embedding model for refinement
                    emb = embedding_model.predict(cv2.resize(hand_img, (112, 112)).astype('float32')[None, ...] / 255.0)[0]
                    sims = cosine_similarity([emb], known_embeddings)[0]
                    topk = np.argsort(sims)[-k:][::-1]
                    votes = [known_labels[i] for i in topk]
                    # Majority vote or top-1
                    final_label = max(set(votes), key=votes.count)
                    method = 'Triplet'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'{final_label} ({method})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('ASL Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def build_known_embeddings(embedding_model, data_dir, webcam_data_dir=None, img_size=112, class_names=None, n_per_class=20):
    """Build a dictionary of class_name -> [embeddings] for k-NN/cosine refinement."""
    import collections
    emb_dict = collections.defaultdict(list)
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    for class_name in class_names:
        imgs = []
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            imgs += [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        if webcam_data_dir:
            webcam_class_dir = os.path.join(webcam_data_dir, class_name)
            if os.path.isdir(webcam_class_dir):
                imgs += [os.path.join(webcam_class_dir, f) for f in os.listdir(webcam_class_dir) if f.lower().endswith('.jpg')]
        if imgs:
            selected = random.sample(imgs, min(n_per_class, len(imgs)))
            for img_path in selected:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (img_size, img_size))
                img_input = img_resized.astype('float32') / 255.0
                # Extract landmarks
                results = hands.process(img_resized)
                landmark_vec = np.zeros(21 * 3, dtype=np.float32)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])
                        landmark_vec = np.array(coords, dtype=np.float32)
                        break
                emb = embedding_model.predict({'image': np.expand_dims(img_input, 0), 'landmark': np.expand_dims(landmark_vec, 0)}, verbose=0)[0]
                emb_dict[class_name].append(emb)
    return emb_dict

def hybrid_inference(classifier_model, embedding_model, class_names, known_embeddings, img_size=224, threshold=0.7):
    """Run real-time webcam inference, using embedding refinement for ambiguous predictions."""
    import scipy.spatial
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    with hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            img_input = cv2.resize(frame, (img_size, img_size))
            img_input = img_input.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            preds = classifier_model.predict(img_input)
            class_id = np.argmax(preds[0])
            class_prob = preds[0][class_id]
            class_name = class_names[class_id]
            # If classifier is not confident, use embedding refinement
            if class_prob < threshold:
                emb = embedding_model.predict(img_input)[0]
                best_class, best_sim = None, -1
                for cname, embs in known_embeddings.items():
                    sims = [1 - scipy.spatial.distance.cosine(emb, e) for e in embs]
                    mean_sim = np.mean(sims) if sims else -1
                    if mean_sim > best_sim:
                        best_sim = mean_sim
                        best_class = cname
                class_name = best_class if best_class else class_name
                class_prob = best_sim if best_sim > 0 else class_prob
            cv2.putText(frame, f'{class_name}: {class_prob:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('ASL Hybrid Inference', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

def collect_webcam_images(class_names, output_dir='webcam_data', img_size=224, n_per_class=100):
    """Collect hand images from webcam and save to webcam_data/{class}/. Does not overwrite existing files and waits for user key to proceed to next class."""
    import cv2
    import mediapipe as mp
    import os
    import time
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    for class_name in class_names:
        print(f"\nCollecting images for class: {class_name}")
        class_dir = os.path.join(output_dir, class_name)
        print(f"Saving images to: {class_dir}")
        try:
            os.makedirs(class_dir, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Could not create/access directory {class_dir}: {e}")
            continue
        # Find the next available index for this class
        existing = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg') and f.startswith(class_name+'_')]
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing if f.split('_')[-1].split('.')[0].isdigit()]
        start_idx = max(existing_indices) + 1 if existing_indices else 1
        count = 0
        # --- Countdown before starting capture ---
        print(f"Get ready to show the sign for class '{class_name}'.")
        print("Press any key to start immediately, or wait for countdown...")
        countdown = 3
        while countdown > 0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Starting in {countdown}...", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, f"Class: {class_name}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Collect Hand Images', frame)
            key = cv2.waitKey(1000) & 0xFF
            if key != 255:  # Any key pressed
                break
            countdown -= 1
        print("Starting capture!")
        # --- Main capture loop ---
        while count < n_per_class:
            ret, frame = cap.read()
            if not ret:
                print("Webcam error.")
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            hand_img = None
            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min = min([lm.x for lm in hand_landmarks.landmark])
                    y_min = min([lm.y for lm in hand_landmarks.landmark])
                    x_max = max([lm.x for lm in hand_landmarks.landmark])
                    y_max = max([lm.y for lm in hand_landmarks.landmark])
                    # Convert to pixel coordinates
                    x1, y1 = int(x_min * w), int(y_min * h)
                    x2, y2 = int(x_max * w), int(y_max * h)
                    # Expand box with margin
                    margin = 60
                    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
                    # Make the crop square
                    box_w, box_h = x2 - x1, y2 - y1
                    side = max(box_w, box_h)
                    # Center the square on the hand
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    half_side = side // 2
                    sq_x1 = max(0, cx - half_side)
                    sq_y1 = max(0, cy - half_side)
                    sq_x2 = min(w, sq_x1 + side)
                    sq_y2 = min(h, sq_y1 + side)
                    # Adjust if crop goes out of bounds
                    if sq_x2 - sq_x1 < side:
                        sq_x1 = max(0, sq_x2 - side)
                    if sq_y2 - sq_y1 < side:
                        sq_y1 = max(0, sq_y2 - side)
                    # Draw square on frame for visualization
                    cv2.rectangle(frame, (sq_x1, sq_y1), (sq_x2, sq_y2), (255, 255, 0), 2)
                    hand_img = frame[sq_y1:sq_y2, sq_x1:sq_x2]
                    if hand_img.size == 0 or hand_img.shape[0] < 40 or hand_img.shape[1] < 40:
                        print(f"[WARN] Hand region too small or empty for class {class_name} (w={hand_img.shape[1]}, h={hand_img.shape[0]})")
                        continue
                    hand_img = cv2.resize(hand_img, (img_size, img_size))
                    img_path = os.path.join(class_dir, f"{class_name}_{start_idx+count}.jpg")
                    print(f"[DEBUG] Attempting to save: {img_path}")
                    try:
                        if not os.path.exists(img_path):
                            cv2.imwrite(img_path, hand_img)
                            count += 1
                            print(f"Saved {img_path} ({count}/{n_per_class})", end='\r')
                    except Exception as e:
                        print(f"[ERROR] Could not save image {img_path}: {e}")
                    hand_detected = True
                    break  # Only save one hand per frame
            cv2.putText(frame, f'Class: {class_name}  Count: {count}/{n_per_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Press ESC to stop, Enter to next class', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not hand_detected:
                cv2.putText(frame, 'No hand detected! Try changing angle/pose.', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow('Collect Hand Images', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\nCollection interrupted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 13:  # Enter
                print("\nProceeding to next class...")
                break
        print(f"\nFinished class {class_name}. Press Enter to continue to next class.")
        while True:
            if cv2.waitKey(0) & 0xFF == 13:
                break
    cap.release()
    cv2.destroyAllWindows()
    print("\nImage collection complete.")

def build_landmark_only_model(num_classes, dropout_rate=0.3):
    """
    Model that takes only the 21*3 hand landmark vector as input and predicts the class.
    Now with additional Dense and Dropout layers for increased capacity.
    """
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.models import Model
    landmark_input = Input(shape=(21*3,), name='landmark')
    x = Dense(256, activation='relu')(landmark_input)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=landmark_input, outputs=output)
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=5e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_landmark_only_model(class_names=None):
    """
    Train a model using only the 21*3 hand landmark vectors as input.
    """
    train_gen, val_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE, class_names=class_names)
    # Wrap generators to only return landmark input
    class LandmarkOnlyWrapper(tf.keras.utils.Sequence):
        def __init__(self, base_gen):
            self.base_gen = base_gen
        def __len__(self):
            return len(self.base_gen)
        def __getitem__(self, idx):
            (inputs, labels) = self.base_gen[idx]
            return inputs['landmark'], labels
        def on_epoch_end(self):
            self.base_gen.on_epoch_end()
    train_landmark_gen = LandmarkOnlyWrapper(train_gen)
    val_landmark_gen = LandmarkOnlyWrapper(val_gen)
    model = build_landmark_only_model(len(class_names) if class_names else NUM_CLASSES)
    model.summary()
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max')
    val_drop_stop = ValidationDropEarlyStopping(patience=3)
    try:
        model.fit(
            train_landmark_gen,
            epochs=EPOCHS,
            validation_data=val_landmark_gen,
            callbacks=[early_stop, val_drop_stop]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save('asl_landmark_only.h5')
        print(f"Model saved to asl_landmark_only.h5")
        raise
    model.save('asl_landmark_only.h5')
    print(f"Model saved to asl_landmark_only.h5")

def main():
    # Fixed config for this project: do not override via CLI
    DATA_DIR = r'C:\Users\kenjo\ASLModel\asl_alphabet_train\asl_alphabet_train'
    WEBCAM_DATA_DIR = 'webcam_data'
    IMG_SIZE = 224
    EMBED_IMG_SIZE = 112
    EMBEDDING_DIM = 128
    MODEL_PATH = 'asl_mobilenetv2.h5'
    EMBEDDING_MODEL_PATH = 'asl_triplet_embedding_model.keras'
    N_PER_CLASS = 100
    parser = argparse.ArgumentParser(description='ASL Hand Gesture Recognizer Pipeline (fixed config)')
    parser.add_argument('--train', action='store_true', help='Train MobileNetV2 classifier')
    parser.add_argument('--train_triplet', action='store_true', help='Train triplet embedding model')
    parser.add_argument('--visualize', action='store_true', help='Visualize embedding space')
    parser.add_argument('--inference', action='store_true', help='Run real-time webcam inference (classifier only)')
    parser.add_argument('--hybrid', action='store_true', help='Run hybrid classifier+embedding inference')
    parser.add_argument('--build_embeddings', action='store_true', help='Build known embeddings for hybrid inference')
    parser.add_argument('--collect', action='store_true', help='Collect hand images from webcam for each class')
    parser.add_argument('--collect_classes', nargs='+', type=str, help='Specify class/classes to collect webcam data for (e.g. --collect_classes A B C)')
    parser.add_argument('--method', type=str, default='umap', help='Visualization method: tsne or umap')
    parser.add_argument('--train_landmark', action='store_true', help='Train classifier using only hand landmarks')
    args = parser.parse_args()

    if args.train:
        train_model(CLASS_NAMES)
    elif args.train_triplet:
        train_triplet_model(DATA_DIR, WEBCAM_DATA_DIR, img_size=EMBED_IMG_SIZE, embedding_dim=EMBEDDING_DIM)
    elif args.visualize:
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
        visualize_embeddings(embedding_model, DATA_DIR, webcam_data_dir=WEBCAM_DATA_DIR, img_size=EMBED_IMG_SIZE, n_per_class=N_PER_CLASS, class_names=CLASS_NAMES, method=args.method)
    elif args.inference:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        run_webcam_inference(model, CLASS_NAMES, img_size=IMG_SIZE)
    elif args.hybrid:
        classifier_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
        print('Building known embeddings for hybrid inference...')
        known_embeddings = build_known_embeddings(embedding_model, DATA_DIR, webcam_data_dir=WEBCAM_DATA_DIR, img_size=EMBED_IMG_SIZE, class_names=CLASS_NAMES, n_per_class=N_PER_CLASS)
        print('Starting hybrid inference...')
        hybrid_inference(classifier_model, embedding_model, CLASS_NAMES, known_embeddings, img_size=IMG_SIZE)
    elif args.build_embeddings:
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
        known_embeddings = build_known_embeddings(embedding_model, DATA_DIR, webcam_data_dir=WEBCAM_DATA_DIR, img_size=EMBED_IMG_SIZE, class_names=CLASS_NAMES, n_per_class=N_PER_CLASS)
        import pickle
        with open('known_embeddings.pkl', 'wb') as f:
            pickle.dump(known_embeddings, f)
    elif args.collect:
        if args.collect_classes:
            collect_webcam_images(args.collect_classes, output_dir=WEBCAM_DATA_DIR, img_size=IMG_SIZE, n_per_class=N_PER_CLASS)
        else:
            collect_webcam_images(CLASS_NAMES, output_dir=WEBCAM_DATA_DIR, img_size=IMG_SIZE, n_per_class=N_PER_CLASS)
    elif args.train_landmark:
        train_landmark_only_model(CLASS_NAMES)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
