# integrator/emotion_to_music.py
from pathlib import Path
from utils import get_logger
from emotion_detector import EmotionPredictor
from music_generator import EmotionToMidi

logger = get_logger("integrator")

def generate_music_from_image(image_path: str, model_path: str = "models/emotion_svc.joblib", midi_out_dir: str = "midi_music"):
    """
    Given an image, predict the emotion and generate a piece of music (MIDI).
    Returns a dict with predicted emotion and path to generated MIDI.
    """
    predictor = EmotionPredictor(model_path=model_path)
    preds = predictor.predict(image_path, top_k=1)
    if not preds:
        raise RuntimeError("No prediction returned.")
    top = preds[0]
    emotion_label = top["label"]
    score = top["score"]
    logger.info("Top emotion: %s (score=%.3f)", emotion_label, score)

    composer = EmotionToMidi(out_dir=midi_out_dir)
    midi_path = composer.compose(emotion_label)
    return {"emotion": emotion_label, "score": score, "midi": midi_path}
