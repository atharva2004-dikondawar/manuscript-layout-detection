import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n for manuscript layout detection."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the YOLO dataset YAML file.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device, e.g. 'cpu' or '0'. Default: cpu.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}"
        )

    runs_dir = ROOT_DIR / "training" / "runs"

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_path),
        epochs=60,
        imgsz=640,
        batch=4,
        device=args.device,
        workers=2,
        patience=15,
        optimizer="auto",

        # Conservative document augmentation
        degrees=2.0,
        translate=0.03,
        scale=0.10,
        shear=1.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.2,

        # Reproducibility
        seed=42,

        # Training outputs
        project=str(runs_dir),
        name="yolo11n_manuscript_baseline_60ep",

        pretrained=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()