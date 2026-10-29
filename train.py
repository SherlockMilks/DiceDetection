import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data" / "dice_yolo" / "data.yaml"

MODEL = "yolo26n.pt"

OUTPUT_DIR = PROJECT_DIR / "runs"

OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer (default: AdamW)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"

    run_name = (
        f"dice_yolo26n_{args.optimizer.lower()}"
        f"_lr{args.lr0:g}_wd{args.weight_decay:g}"
    )

    print(f"Device: {device}")
    print(f"Dataset configuration: {DATA_YAML}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr0}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Run name: {run_name}")

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=832,
        batch=16,
        device=device,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        save=True,
        plots=True,
        patience=20,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
    )

    print("Training finished.")
    print(f"Best weights: {OUTPUT_DIR / run_name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
