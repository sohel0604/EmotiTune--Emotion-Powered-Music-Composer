# utils/logger.py
import logging
import os
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "emotitune.log"

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
