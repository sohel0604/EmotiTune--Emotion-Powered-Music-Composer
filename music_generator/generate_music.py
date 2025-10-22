# music_generator/generate_music.py
import pretty_midi
from pathlib import Path
import numpy as np
from utils import get_logger

logger = get_logger("generate_music")

# simple mapping from emotion -> (scale, tempo, base_note)
EMOTION_MAPPINGS = {
    "happy":    {"tempo": 140, "scale": "major", "base": 60},
    "sad":      {"tempo": 70,  "scale": "minor", "base": 48},
    "angry":    {"tempo": 150, "scale": "phrygian", "base": 55},
    "surprise": {"tempo": 120, "scale": "major", "base": 62},
    "neutral":  {"tempo": 100, "scale": "major", "base": 57},
    "fear":     {"tempo": 110, "scale": "locrian", "base": 50},
    "disgust":  {"tempo": 85,  "scale": "minor", "base": 50},
}

SCALES = {
    "major": [0,2,4,5,7,9,11],
    "minor": [0,2,3,5,7,8,10],
    "phrygian": [0,1,3,5,7,8,10],
    "locrian": [0,1,3,5,6,8,10],
}

class EmotionToMidi:
    def __init__(self, out_dir="midi_music"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_melody(self, mapping, bars=8, beats_per_bar=4, division=4):
        """
        Build a simple melody (list of (start, end, pitch, velocity)).
        division e.g., 4 -> quarter notes as unit, 8 -> eighth notes
        """
        scale = SCALES.get(mapping["scale"], SCALES["major"])
        base = int(mapping["base"])
        tempo = mapping["tempo"]
        seconds_per_beat = 60.0 / tempo
        beat_dur = seconds_per_beat
        unit = beat_dur / (division / 4)  # if division=4, unit is quarter note length

        notes = []
        total_notes = bars * beats_per_bar * (division // 4)
        t = 0.0
        rng = np.random.RandomState(seed=42)
        for i in range(total_notes):
            degree = rng.choice(len(scale), p=None)
            octave = rng.choice([0, 0, 1])  # prefer base octave
            pitch = base + scale[degree] + 12 * octave
            dur_units = rng.choice([1,1,2])  # mostly short notes
            dur = dur_units * unit
            vel = int(rng.randint(70, 110))
            notes.append((t, t + dur, int(pitch), vel))
            t += dur
        return notes, tempo

    def compose(self, emotion_label: str, filename: str = None):
        mapping = EMOTION_MAPPINGS.get(emotion_label.lower(), EMOTION_MAPPINGS["neutral"])
        notes, tempo = self._build_melody(mapping)
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        for (start, end, pitch, vel) in notes:
            note = pretty_midi.Note(velocity=vel, pitch=int(pitch), start=float(start), end=float(end))
            inst.notes.append(note)
        pm.instruments.append(inst)
        if filename is None:
            filename = f"{emotion_label}_{int(tempo)}.mid"
        out_path = self.out_dir / filename
        pm.write(str(out_path))
        logger.info("Saved MIDI to %s", out_path)
        return str(out_path)
