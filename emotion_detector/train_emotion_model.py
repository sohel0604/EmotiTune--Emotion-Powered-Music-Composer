import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import hog

# Paths
DATA_DIR = "Data/FER-2013/train"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_svc.joblib")

# Create model folder if not exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Emotion labels used in FER-2013 dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def load_images(data_dir, img_size=(48, 48)):
    X, y = [], []
    print("Loading images...")

    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Skipping missing folder: {emotion_path}")
            continue

        for file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                X.append(features)
                y.append(idx)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    print(f"✅ Loaded {len(X)} samples.")
    return np.array(X), np.array(y)


def train_emotion_model():
    X, y = load_images(DATA_DIR)

    if len(X) == 0:
        print("❌ No images found. Please check your FER-2013 dataset path.")
        return

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    print("Saving model...")
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved successfully at {MODEL_PATH}")


if __name__ == "__main__":
    train_emotion_model()
