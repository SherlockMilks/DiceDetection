import argparse
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data" / "dice_yolo" / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="The directory containing the desired model (dice_yolo26n_adam_lr0.001_wd0.0005)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    weights_dir = args.weights
    weights_path = (
            PROJECT_DIR
            / "runs"
            / weights_dir
            / "weights"
            / "best.pt"
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {weights_dir}")
    print("Evaluating on the test set")

    model = YOLO(str(weights_path))

    temp_dir = PROJECT_DIR / "_temp_validation"
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=832,
        batch=16,
        device=device,
        plots=False,
        verbose=True,
        project=str(temp_dir)
    )
    shutil.rmtree(temp_dir, ignore_errors=True)

    results = {
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }

    print("\nTest results:")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    output_path = PROJECT_DIR / "test_results" / f"{weights_dir}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)


    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {weights_dir}\n")

        for name, value in results.items():
            file.write(f"{name}: {value:.6f}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
