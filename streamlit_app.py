# streamlit_app.py
import streamlit as st
from pathlib import Path
from io import BytesIO
import numpy as np
import soundfile as sf
import pretty_midi
import base64
from typing import Tuple, List

# Import project modules
from utils import get_logger
from emotion_detector.predict_emotion import EmotionPredictor
from music_generator.generate_music import EmotionToMidi

logger = get_logger("streamlit_app")

st.set_page_config(page_title="EmotiTune", layout="centered", page_icon="🎵")

# ---------------------------
# Helper: synthesize MIDI -> WAV (simple additive synth)
# ---------------------------
def midi_to_wav_bytes(midi_path: str, sr: int = 22050) -> bytes:
    """
    Read midi via pretty_midi and synthesize a simple audio buffer using sine waves per note.
    This is a lightweight synth (not realistic) but good for quick playback in the browser.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    # compute total duration
    duration = pm.get_end_time() + 0.5
    audio = np.zeros(int(duration * sr), dtype=np.float32)

    def note_wave(freq, start_idx, end_idx, sr, velocity=100):
        t = np.linspace(0, (end_idx - start_idx) / sr, end_idx - start_idx, False)
        # amplitude scaled by velocity
        amp = (velocity / 127.0) * 0.2
        # simple ADSR-like envelope
        env = np.ones_like(t)
        # quick attack and release
        a = int(0.01 * sr)
        r = int(0.03 * sr)
        if a + r < len(env):
            env[:a] = np.linspace(0, 1.0, a)
            env[-r:] = np.linspace(1.0, 0, r)
        else:
            env = np.linspace(0, 1.0, len(env))
        wave = amp * env * np.sin(2 * np.pi * freq * t)
        return wave.astype(np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            start_idx = int(note.start * sr)
            end_idx = int(note.end * sr)
            if end_idx <= start_idx:
                continue
            freq = pretty_midi.note_number_to_hz(note.pitch)
            w = note_wave(freq, start_idx, end_idx, sr, velocity=note.velocity)
            audio[start_idx:end_idx] += w[: end_idx - start_idx]

    # Soft clip / normalize
    maxv = np.max(np.abs(audio)) + 1e-9
    audio = audio / maxv * 0.95

    # write to bytes
    buf = BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()

# ---------------------------
# Helpers for UI
# ---------------------------
@st.cache_resource
def load_predictor(model_path: str = "models/emotion_svc.joblib", image_size=(48,48)):
    return EmotionPredictor(model_path=model_path, image_size=image_size)

@st.cache_resource
def load_composer(out_dir: str = "midi_music"):
    return EmotionToMidi(out_dir=out_dir)

def make_download_link(file_path: str, label: str):
    """
    Returns a Streamlit download button for a file path.
    """
    p = Path(file_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f"data:application/octet-stream;base64,{b64}"
    return href

# ---------------------------
# UI: Title + sidebar
# ---------------------------
st.title("EmotiTune 🎭 ➜ 🎶")
st.markdown(
    "Upload a face image (or pick a sample) → detect the emotion → generate a MIDI composition → play or download audio."
)

st.sidebar.header("Options")
model_path = st.sidebar.text_input("Emotion model path", value="models/emotion_svc.joblib")
midi_out_dir = st.sidebar.text_input("MIDI output folder", value="midi_music")
synth_sr = st.sidebar.number_input("Synthesis sample rate", value=22050, step=1024)
top_k = st.sidebar.slider("Top-K predictions to show", min_value=1, max_value=5, value=3)

# ---------------------------
# Load predictor & composer (cached)
# ---------------------------
try:
    predictor = load_predictor(model_path=model_path)
except Exception as e:
    st.error(f"Failed to load emotion predictor: {e}")
    predictor = None

composer = load_composer(out_dir=midi_out_dir)

# ---------------------------
# Input: upload or sample selector
# ---------------------------
st.subheader("Input image")
col1, col2 = st.columns([3, 1])

uploaded = col1.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

sample_root = Path("Data/FER-2013")
sample_images = []
if sample_root.exists():
    # gather up to 50 sample images
    for p in sample_root.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            sample_images.append(str(p))
            if len(sample_images) >= 50:
                break

sample_choice = None
if sample_images:
    sample_choice = col2.selectbox("Or pick a sample", options=["--"] + sample_images, index=0)

# choose image path & display
input_image_path = None
if uploaded is not None:
    # save temp uploaded file
    tpath = Path("logs") / "tmp_uploaded_image"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "wb") as f:
        f.write(uploaded.getbuffer())
    input_image_path = str(tpath)
    st.image(uploaded, caption="Uploaded image", use_column_width=True)
elif sample_choice and sample_choice != "--":
    input_image_path = sample_choice
    st.image(sample_choice, caption=f"Sample: {Path(sample_choice).name}", use_column_width=True)
else:
    st.info("Upload an image or select a sample to begin.")

# ---------------------------
# Predict button
# ---------------------------
if input_image_path and predictor is not None:
    if st.button("🔍 Predict Emotion & Generate Music"):
        try:
            with st.spinner("Predicting emotion..."):
                preds = predictor.predict(input_image_path, top_k=top_k)
            st.success("Prediction complete")
            # display top-k
            st.subheader("Predictions")
            for p in preds:
                st.write(f"- **{p['label']}** : {p['score']:.3f}")

            top_pred = preds[0]["label"]
            st.write("---")
            st.subheader(f"Generate music for **{top_pred}**")

            # Generate MIDI
            with st.spinner("Composing MIDI..."):
                midi_path = composer.compose(top_pred)
            st.success(f"MIDI generated: {midi_path}")
            st.write(f"**MIDI:** `{midi_path}`")

            # MIDI download
            href = make_download_link(midi_path, "Download MIDI")
            if href:
                st.markdown(f"[Download MIDI]({href})")

            # Synthesize WAV
            with st.spinner("Synthesizing audio (lightweight synth)..."):
                wav_bytes = midi_to_wav_bytes(midi_path, sr=synth_sr)
            st.success("Audio synthesized")

            st.subheader("Listen")
            st.audio(wav_bytes, format="audio/wav")

            # Provide WAV download
            wav_name = Path(midi_path).with_suffix(".wav").name
            st.download_button(
                label="Download WAV",
                data=wav_bytes,
                file_name=wav_name,
                mime="audio/wav",
            )
        except Exception as e:
            logger.exception("Error during pipeline: %s", e)
            st.error(f"Pipeline failed: {e}")

# ---------------------------
# Footer / tips
# ---------------------------
st.markdown("---")
st.caption("Notes: This app uses a lightweight SVM on HOG features for emotion detection and a simple rule-based composer. Audio is synthesized using a basic sine-wave synth for immediate playback (not realistic piano). For higher quality audio, integrate a soundfont-based synth (fluidsynth) or export MIDI and render with a DAW / soundfont.")
