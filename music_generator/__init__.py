# music_generator/__init__.py
from .generate_music import EmotionToMidi
from .train_music_model import train_music_model
__all__ = ["EmotionToMidi", "train_music_model"]
