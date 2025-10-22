# utils/preprocessing.py
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.feature import hog
from typing import List, Tuple

def find_image_files(root_dir: str, extensions=(".png", ".jpg", ".jpeg")) -> List[Tuple[str,str]]:
    """
    Walk root_dir expecting structure:
      root_dir/
        emotion_label_1/
          img1.png
        emotion_label_2/
          img2.png
    Returns list of (file_path, label)
    """
    files = []
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root_dir}")
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for f in label_dir.rglob("*"):
            if f.suffix.lower() in extensions:
                files.append((str(f.resolve()), label))
    return files

def load_image(path: str, as_gray=True, size=(48,48)):
    """
    Load an image, convert to grayscale and resize to `size`.
    Returns numpy array float32 scaled 0..1.
    """
    img = Image.open(path).convert("L" if as_gray else "RGB")
    img = img.resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

def image_to_hog(image_array: np.ndarray, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9):
    """
    Convert a 2D grayscale image array into HOG features.
    """
    # scikit-image's hog expects 2D arrays for grayscale
    features = hog(image_array,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm="L2-Hys",
                   visualize=False,
                   feature_vector=True)
    return features
