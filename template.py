import os

# Define all folders required for EmotiTune project
folders = [
    "data/FER-2013/train",
    "data/FER-2013/test",
    "data/midi_music",
    "emotion_detector",
    "music_generator",
    "integrator",
    "utils",
    "notebooks",
    "models",
    "logs"
]

# Define essential files
files = [
    "app.py",
    "emotion_detector/__init__.py",
    "emotion_detector/train_emotion_model.py",
    "emotion_detector/predict_emotion.py",
    "music_generator/__init__.py",
    "music_generator/train_music_model.py",
    "music_generator/generate_music.py",
    "integrator/__init__.py",
    "integrator/emotion_to_music.py",
    "utils/__init__.py",
    "utils/logger.py",
    "utils/exception.py",
    "utils/preprocessing.py",
    "requirements.txt",
    "README.md",
    "setup.py",
    ".gitignore"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f" Created folder: {folder}")

# Create files
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f" Created file: {file}")

print("\n EmotiTune project structure created successfully!")
