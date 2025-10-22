from pathlib import Path
import joblib
import numpy as np
from utils import get_logger
from utils.preprocessing import load_image, image_to_hog

logger = get_logger("predict_emotion")

class EmotionPredictor:
    def __init__(self, model_path="models/emotion_svc.joblib", image_size=(48, 48)):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")

        # Load the SVC model directly
        self.model = joblib.load(str(self.model_path))
        self.image_size = image_size

        # Define your label names (since no label encoder was saved)
        self.labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def predict(self, image_path: str, top_k: int = 1):
        x = load_image(image_path, size=self.image_size)
        feat = image_to_hog(x)
        probs = self.model.predict_proba([feat])[0]
        idxs = probs.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({"label": self.labels[i], "score": float(probs[i])})
        logger.info("Predicted: %s", results)
        return results
