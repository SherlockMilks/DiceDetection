from pathlib import Path
import random
import shutil
import json


RAW_DIR = Path("data/original")
OUTPUT_DIR = Path("data/dice_yolo")

IMAGES_DIR = RAW_DIR / "images"
LABELS_DIR = RAW_DIR / "labels"

SEED = 42
SPLIT_RATIOS = (0.8, 0.1)

CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]

def create_split(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if output_dir == raw_dir or output_dir in raw_dir.parents or raw_dir in output_dir.parents:
        raise ValueError('Input and output directories must not contain each other')
    if output_dir.exists():
        raise FileExistsError(f'Output already exists; use a new directory: {output_dir}')
    IMAGES_DIR, LABELS_DIR, OUTPUT_DIR = raw_dir/'images', raw_dir/'labels', output_dir
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected directories:\n{IMAGES_DIR}\n{LABELS_DIR}"
        )

    images = sorted(
        p for p in IMAGES_DIR.iterdir()
    )

    group_ids = {}
    for image in images:
        group_id = Path(image.name).stem.split("__", maxsplit=1)[0]
        group_ids.setdefault(group_id, []).append(image)


    sorted_ids = sorted(group_ids)
    random.Random(SEED).shuffle(sorted_ids)


    n_groups = len(sorted_ids)
    n_train = round(n_groups * SPLIT_RATIOS[0])
    n_val = round(n_groups * SPLIT_RATIOS[1])


    splits = {
        "train": sorted_ids[:n_train],
        "val": sorted_ids[n_train:n_train + n_val],
        "test": sorted_ids[n_train + n_val:],
    }
    if any(not ids for ids in splits.values()):
        raise ValueError('Too few original groups for three nonempty splits')

    for split_name, ids in splits.items():
        image_out = OUTPUT_DIR / "images" / split_name
        label_out = OUTPUT_DIR / "labels" / split_name

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for group_id in ids:
            for image in group_ids[group_id]:
                shutil.copy2(image, image_out / image.name)
                shutil.copy2(
                    LABELS_DIR / f"{image.stem}.txt",
                    label_out / f"{image.stem}.txt",
                )

    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    with (OUTPUT_DIR / "data.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        # JSON syntax is also valid YAML; no extra dependency is necessary.
        json.dump(yaml_data, file, indent=2)
    manifest = {'seed': SEED, 'ratios': [0.8, 0.1, 0.1],
                'group_rule': 'filename stem before __',
                'splits': {s: [p.name for gid in ids for p in group_ids[gid]] for s, ids in splits.items()}}
    (OUTPUT_DIR/'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    total_images = len(images)
    print(f"Original images: {total_images}")
    print(f"Original image groups: {n_groups}")
    print()

    for split_name, ids in splits.items():
        count = sum(len(group_ids[group_id]) for group_id in ids)
        percentage = 100 * count / total_images
        print(
            f"{split_name:5}: {count:5} images "
            f"({percentage:5.1f}%), {len(ids)} original groups"
        )

    print(f"\nDataset split successfully created!")


if __name__ == '__main__':
    create_split()
