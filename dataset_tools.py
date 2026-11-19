"""Shared, read-only checks for YOLO dice datasets."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def group_id(path):
    return Path(path).stem.split('__', 1)[0]


def read_labels(path, check_bounds=True):
    """Return validated (class_id, x, y, width, height) normalized boxes."""
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('expected five fields')
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if cls not in range(6):
                raise ValueError('class must be 0..5')
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError('non-finite coordinate')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('coordinates outside normalized range')
            if check_bounds and (min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
                raise ValueError('box extends outside image')
            boxes.append((cls, x, y, w, h))
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
    if not boxes:
        raise ValueError(f'{path}: empty annotation')
    return boxes
