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

The training dataset itself is **not included** in this repository (image collections of this size aren't practical to commit directly). To retrain, organize your dataset in the following structure:

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

Augmentation was kept deliberately conservative, since aggressive geometric augmentation (large rotations, flips) can distort real page-layout structure in ways that don't reflect how manuscripts actually vary — a flipped page, for instance, doesn't represent a realistic manuscript orientation.

| Setting | Value |
|---|---|
| Rotation | 2.0° |
| Translation | 0.03 |
| Scale | 0.10 |
| Shear | 1.0 |
| Horizontal flip | Disabled |
| Vertical flip | Disabled |
| Mosaic | 0.2 |

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