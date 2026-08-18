# Test Images

This folder is intentionally **empty** — no sample images are included in the repository.

## Purpose

Place your own manuscript page images here before running inference. This is the default input location used in the examples throughout the main README.

## Usage

1. Copy your manuscript images into this folder (`data/test_images/`).
2. Run inference from the repository root:

```bash
python inference.py --input ./data/test_images --output ./results
```

## Supported Formats

| Format  |
| ------- |
| `.jpg`  |
| `.jpeg` |
| `.png`  |
| `.bmp`  |
| `.tif`  |
| `.tiff` |

## Notes

- This folder is **not required** to be used exactly as-is — you can point `--input` at any other directory containing your images instead:
  ```bash
  python inference.py --input /path/to/your/images --output ./results
  ```
- Images placed here are **never modified** by the pipeline. All output (annotated images and JSON metadata) is written separately to the `results/` folder.
- No sample or training images are shipped in this folder by design, to keep the repository lightweight and avoid including image data unrelated to the actual code deliverable.