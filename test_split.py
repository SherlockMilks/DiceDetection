import pytest
from pathlib import Path

RAW_DIR = Path("data/original")
SPLIT_DIR = Path("data/dice_yolo")

RAW_IMAGES_DIR = RAW_DIR / "images"
RAW_LABELS_DIR = RAW_DIR / "labels"

SPLIT_IMAGES_DIR = SPLIT_DIR / "images"
SPLIT_LABELS_DIR = SPLIT_DIR / "labels"

def test_split_images_count():
    original_images = [p for p in RAW_IMAGES_DIR.iterdir()]

    test_images = [p for p in (SPLIT_IMAGES_DIR / "test").iterdir()]
    val_images = [p for p in (SPLIT_IMAGES_DIR / "val").iterdir()]
    train_images = [p for p in (SPLIT_IMAGES_DIR / "train").iterdir()]

    assert len(original_images) == len(test_images) + len(val_images) + len(train_images)


def test_split_labels_count():
    original_labels = [p for p in RAW_LABELS_DIR.iterdir()]

    test_labels = [p for p in (SPLIT_LABELS_DIR / "test").iterdir()]
    val_labels = [p for p in (SPLIT_LABELS_DIR / "val").iterdir()]
    train_labels = [p for p in (SPLIT_LABELS_DIR / "train").iterdir()]

    assert len(original_labels) == len(test_labels) + len(val_labels) + len(train_labels)


def test_all_test_images_have_labels():
    label_stems = []
    for labels in (SPLIT_LABELS_DIR / "test").iterdir():
        label_stems.append(labels.stem)

    for image in (SPLIT_IMAGES_DIR / "test").iterdir():
            assert label_stems.count(image.stem) == 1


def test_all_train_images_have_labels():
    label_stems = []
    for labels in (SPLIT_LABELS_DIR / "train").iterdir():
        label_stems.append(labels.stem)

    for image in (SPLIT_IMAGES_DIR / "train").iterdir():
            assert label_stems.count(image.stem) == 1


def test_all_val_images_have_labels():
    label_stems = []
    for labels in (SPLIT_LABELS_DIR / "val").iterdir():
        label_stems.append(labels.stem)

    for image in (SPLIT_IMAGES_DIR / "val").iterdir():
            assert label_stems.count(image.stem) == 1


def test_no_modified_duplicates():
    test_image_ids = set()
    for image in (SPLIT_LABELS_DIR / "test").iterdir():
        test_image_ids.add(image.stem.split("__",maxsplit=1)[0])

    train_image_ids = set()
    for image in (SPLIT_LABELS_DIR / "train").iterdir():
        train_image_ids.add(image.stem.split("__",maxsplit=1)[0])

    val_image_ids = set()
    for image in (SPLIT_LABELS_DIR / "val").iterdir():
        val_image_ids.add(image.stem.split("__",maxsplit=1)[0])

    assert test_image_ids.isdisjoint(train_image_ids)
    assert test_image_ids.isdisjoint(val_image_ids)
    assert train_image_ids.isdisjoint(val_image_ids)

