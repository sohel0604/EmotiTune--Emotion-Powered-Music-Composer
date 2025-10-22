# emotion_detector/__init__.py
from .train_emotion_model import train_emotion_model
from .predict_emotion import EmotionPredictor

__all__ = ["train_emotion_model", "EmotionPredictor"]
