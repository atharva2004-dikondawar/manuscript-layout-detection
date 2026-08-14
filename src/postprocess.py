from typing import Any


CLASS_NAMES = {
    0: "main_text",
    1: "header_text",
    2: "footer_text",
    3: "side_text",
    4: "filler",
}


def clip_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Clip a bounding box to image boundaries.

    Returns:
        (x1, y1, x2, y2)
    """

    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(0.0, min(float(x2), float(image_width)))
    y2 = max(0.0, min(float(y2), float(image_height)))

    # Convert to integer pixel coordinates.
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    return x1, y1, x2, y2


def validate_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    """
    Validate that a bounding box lies within the image
    and has positive area.
    """

    x1, y1, x2, y2 = bbox

    return (
        0 <= x1 < x2 <= image_width
        and
        0 <= y1 < y2 <= image_height
    )


def process_detections(
    result: Any,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Convert Ultralytics detections into clean metadata.

    Args:
        result: Ultralytics result for one image.
        image_width: Original image width.
        image_height: Original image height.

    Returns:
        List of processed detections.
    """

    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()

    for coordinates, confidence, class_id in zip(
        xyxy,
        confidences,
        class_ids,
    ):
        x1, y1, x2, y2 = coordinates

        bbox = clip_bbox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=image_width,
            image_height=image_height,
        )

        # Ignore invalid/zero-area boxes after clipping.
        if not validate_bbox(
            bbox,
            image_width=image_width,
            image_height=image_height,
        ):
            continue

        class_id = int(class_id)

        class_name = CLASS_NAMES.get(
            class_id,
            str(result.names.get(class_id, class_id)),
        )

        detections.append(
            {
                "class": class_name,
                "confidence": round(float(confidence), 4),
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
            }
        )

    return detections