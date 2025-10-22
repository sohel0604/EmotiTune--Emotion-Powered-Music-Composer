# music_generator/train_music_model.py
from utils import get_logger
logger = get_logger("train_music_model")

def train_music_model(midi_folder: str = "midi_music", out_model: str = "models/music_model_placeholder.joblib"):
    """
    Placeholder: a full music model (e.g., an RNN or Transformer on MIDI) requires substantial
    dataset preparation. This stub exists so project structure is complete.

    If you later want to implement: parse MIDI -> sequence of notes -> train sequence model -> save weights.
    """
    logger.warning("train_music_model is a placeholder. No model trained.")
    return None
