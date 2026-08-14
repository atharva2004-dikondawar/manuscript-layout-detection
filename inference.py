#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

from src.detector import ManuscriptDetector
from src.output import (
    draw_annotations,
    save_annotated_image,
    save_json,
)
from src.postprocess import process_detections


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Batch manuscript page layout detection using YOLO."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing input manuscript page images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where inference results will be written.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best.pt"),
        help="Path to trained YOLO model. Default: models/best.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
        help="Detection confidence threshold. Default: 0.15",
    )

    return parser.parse_args()


def collect_images(input_directory: Path) -> list[Path]:
    """Collect supported image files from an input directory."""

    images = [
        path
        for path in input_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images)


def process_image(
    image_path: Path,
    detector: ManuscriptDetector,
    output_directory: Path,
    confidence: float,
) -> None:
    """Run inference and write outputs for one image."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Unable to read image: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    result = detector.predict(
        image_path=image_path,
        confidence=confidence,
    )

    detections = process_detections(
        result=result,
        image_width=image_width,
        image_height=image_height,
    )

    annotated_image = draw_annotations(
        image=image,
        detections=detections,
    )

    image_output_directory = output_directory / "images"
    metadata_output_directory = output_directory / "metadata"

    annotated_image_path = (
        image_output_directory / image_path.name
    )

    metadata_path = (
        metadata_output_directory
        / f"{image_path.stem}.json"
    )

    save_annotated_image(
        output_path=annotated_image_path,
        image=annotated_image,
    )

    save_json(
        output_path=metadata_path,
        image_name=image_path.name,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
    )

    print(
        f"[OK] {image_path.name}: "
        f"{len(detections)} detections"
    )


def main():
    """Main CLI entry point."""

    args = parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input directory does not exist: {args.input}"
        )

    if not args.input.is_dir():
        raise SystemExit(
            f"Input path is not a directory: {args.input}"
        )

    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit(
            "Confidence must be between 0.0 and 1.0."
        )

    images = collect_images(args.input)

    if not images:
        raise SystemExit(
            f"No supported images found in: {args.input}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Manuscript Layout Detection")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Model:      {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Images:     {len(images)}")
    print("=" * 60)

    detector = ManuscriptDetector(
        model_path=args.model,
    )

    successful = 0
    failed = 0

    for image_path in images:
        try:
            process_image(
                image_path=image_path,
                detector=detector,
                output_directory=args.output,
                confidence=args.confidence,
            )
            successful += 1

        except Exception as exc:
            failed += 1
            print(
                f"[ERROR] {image_path.name}: {exc}"
            )

    print()
    print("=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Results:    {args.output}")
    print("=" * 60)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()