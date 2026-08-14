from pathlib import Path
from typing import Any

from ultralytics import YOLO


class ManuscriptDetector:
    """YOLO-based manuscript page layout detector."""

    def __init__(self, model_path: str | Path):
        """
        Initialize the detector.

        Args:
            model_path: Path to the trained YOLO model.
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image_path: str | Path,
        confidence: float = 0.15,
    ) -> Any:
        """
        Run inference on one image.

        Args:
            image_path: Path to input image.
            confidence: Minimum confidence threshold.

        Returns:
            Ultralytics YOLO result object.
        """
        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False,
        )

        if not results:
            raise RuntimeError(
                f"No inference result returned for: {image_path}"
            )

        return results[0]