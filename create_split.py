from pathlib import Path
import random
import shutil
import numpy as np
import json
import cv2


RAW_DIRS = [Path("data/dice_detection_upper"), Path("data/my_pictures")]
OUTPUT_DIR = Path("data/dice_yolo")

SEED = 42
IMAGE_SIZE = 832

CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]


def resize_image_and_labels(image_path, label_path, size):
    image = cv2.imread(str(image_path))

    original_height, original_width = image.shape[:2]

    scale = min(
        size / original_width,
        size / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (size, size, 3),
        0,
        dtype=np.uint8
    )

    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    new_labels = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if not values:
                continue

            class_id, cx, cy, w, h = values[:5]

            cx = float(cx)
            cy = float(cy)
            w = float(w)
            h = float(h)

            new_cx = (cx * new_width + pad_x) / size
            new_cy = (cy * new_height + pad_y) / size

            new_w = w * new_width / size
            new_h = h * new_height / size

            new_labels.append(
                f"{class_id} {new_cx:.8f} {new_cy:.8f} "
                f"{new_w:.8f} {new_h:.8f}"
            )

    return canvas, new_labels


def create_split(raw_dirs, output_dir):
    random_generator = random.Random(SEED)

    groups = {}

    for source_idx, raw_dir in enumerate(raw_dirs):

        images_dir = raw_dir / "images"
        labels_dir = raw_dir / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Hiányzó mappák:\n{images_dir}\n{labels_dir}"
            )

        images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        for image in images:

            label_path = labels_dir / f"{image.stem}.txt"

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Nem található címkefájl: {label_path}"
                )

            group_id = image.stem.split("__", maxsplit=1)[0]

            group_key = (source_idx, group_id)

            groups.setdefault(group_key, []).append(
                (image, label_path, source_idx)
            )

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for source_idx in range(len(raw_dirs)):
        group_ids = [
            key for key in groups
            if key[0] == source_idx
        ]

        random_generator.shuffle(group_ids)

        n_groups = len(group_ids)

        n_train = round(n_groups * 0.8)
        n_val = round(n_groups * 0.1)

        splits["train"].extend(group_ids[:n_train])
        splits["val"].extend(group_ids[n_train:n_train + n_val])
        splits["test"].extend(group_ids[n_train + n_val:])

    manifest_splits = {}

    for split_name, split_groups in splits.items():

        image_out = output_dir / "images" / split_name
        label_out = output_dir / "labels" / split_name

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        manifest_splits[split_name] = []

        for group_key in split_groups:

            for image, label_path, source_idx in groups[group_key]:

                output_name = f"src{source_idx}_{image.name}"
                output_label_name = f"src{source_idx}_{image.stem}.txt"

                resized_image, new_labels = resize_image_and_labels(
                    image,
                    label_path,
                    IMAGE_SIZE
                )

                output_image_path = image_out / output_name
                output_label_path = label_out / output_label_name

                cv2.imwrite(
                    str(output_image_path),
                    resized_image
                )

                with open(
                    output_label_path,
                    "w",
                    encoding="utf-8"
                ) as file:
                    file.write("\n".join(new_labels))

                manifest_splits[split_name].append(output_name)

    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            i: name for i, name in enumerate(CLASS_NAMES)
        },
    }

    with (output_dir / "data.yaml").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(yaml_data, file, indent=2)

    manifest = {
        "seed": SEED,
        "ratios": [0.8, 0.1, 0.1],
        "image_size": f"{IMAGE_SIZE}",
        "resize_method": "letterbox",
        "group_rule": "filename stem before __",
        "splits": manifest_splits,
    }

    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    total_images = sum(
        len(images) for images in groups.values()
    )

    print(f"Total images: {total_images}")
    print(f"Total image groups: {len(groups)}")
    print()

    for split_name, split_groups in splits.items():

        count = sum(
            len(groups[group_id])
            for group_id in split_groups
        )

        percentage = (
            100 * count / total_images
            if total_images > 0 else 0
        )

        print(
            f"{split_name:5}: {count:5} images "
            f"({percentage:5.1f}%), "
            f"{len(split_groups)} groups"
        )

    print("\nSplit successfully created!")


if __name__ == '__main__':
    create_split(RAW_DIRS, OUTPUT_DIR)
