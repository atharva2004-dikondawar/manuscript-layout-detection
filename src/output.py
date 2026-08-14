import json
from pathlib import Path

import cv2


def draw_annotations(
    image,
    detections: list[dict],
):
    """
    Draw bounding boxes and class labels on an image.

    Args:
        image: OpenCV image.
        detections: Processed detections.

    Returns:
        Annotated image.
    """

    annotated = image.copy()

    for detection in detections:
        bbox = detection["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        class_name = detection["class"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        # Determine text size.
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        # Keep label background inside image.
        label_y1 = max(0, y1 - text_height - baseline - 4)
        label_y2 = y1

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_width + 4, label_y2),
            (255, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(text_height, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def save_json(
    output_path: str | Path,
    image_name: str,
    image_width: int,
    image_height: int,
    detections: list[dict],
) -> None:
    """
    Save detection metadata as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "image": image_name,
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def save_annotated_image(
    output_path: str | Path,
    image,
) -> None:
    """
    Save an annotated image without modifying the original.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save annotated image: {output_path}"
        )