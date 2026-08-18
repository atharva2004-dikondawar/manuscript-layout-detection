# Training

This directory contains the training configuration used to produce the submitted manuscript layout detection model.

## Important

**Training is optional for evaluation.** The final trained model is already included at:

```
models/best.pt
```

Therefore, training is **not required** to run the submission — this directory exists purely for transparency and reproducibility.

---

## Dataset

The full training dataset (173 manually annotated manuscript images, including original images and YOLO-format label files) is available for download here:

**Dataset download:** https://drive.google.com/drive/folders/1wDzXJT2WOGMleG9ejMIIcVL01J5QW4Pd

Download and extract the folder, then organize it into the following structure (this matches what `training/data.yaml` already expects, so no further configuration is needed if you place it exactly as shown):

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image in `images/` must have a matching YOLO-format `.txt` label file with the same filename in `labels/`.

### Classes

| ID | Class |
|---|---|
| 0 | `main_text` |
| 1 | `header_text` |
| 2 | `footer_text` |
| 3 | `side_text` |
| 4 | `filler` |

---

## Training Configuration

The submitted model was trained using the following setup:

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Maximum epochs | 60 |
| Image size | 640 |
| Batch size | 4 |
| Device | CPU |
| Workers | 2 |
| Early stopping patience | 15 |
| Optimizer | Auto |
| Pretrained weights | Yes |
| Seed | 42 |

### Augmentation

| Setting | Value |
|---|---|
| Rotation | 2.0° |
| Translation | 0.03 |
| Scale | 0.10 |
| Shear | 1.0 |
| Horizontal flip | Disabled |
| Vertical flip | Disabled |
| Mosaic | 0.2 |

If you place the downloaded `dataset/` folder as a direct sibling of `training/` (i.e. `manuscript-layout-detection/dataset/`), the included `training/data.yaml` will find it automatically — no path editing required.

---

## Running Training

Install dependencies from the repository root first:

```bash
pip install -r requirements.txt
```

Then run training, pointing `--data` at your own dataset YAML:

```bash
python training/train.py --data /path/to/data.yaml
```

On Windows PowerShell:

```powershell
python training/train.py --data "C:\path\to\data.yaml"
```

A reference `data.yaml` template is included in this folder (`training/data.yaml`) — copy and adjust the `path:` field to point at your own dataset location, or pass a different YAML file directly via `--data`.

**Note:** the first run will automatically download the base `yolo11n.pt` pretrained weights if not already cached locally, which requires an internet connection.

### GPU Training

CPU is the default. GPU training can be requested if a compatible PyTorch/CUDA environment is available:

```bash
python training/train.py --data /path/to/data.yaml --device 0
```

---

## Output

Training outputs (weights, plots, logs) are saved under:

```
training/runs/
```

This process **never overwrites** the submitted model at `models/best.pt`.

A newly trained model will be available separately at:

```
training/runs/yolo11n_manuscript_baseline_60ep/weights/best.pt
```

If you want to use a newly trained model for inference instead of the submitted one, copy it into `models/` and point `inference.py`'s `--model` argument at it explicitly, or replace `models/best.pt` directly.