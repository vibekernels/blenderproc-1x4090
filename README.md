# Beer Pong Synthetic Dataset Generator

Generates synthetic beer pong table images using BlenderProc for training cup
detection models (e.g. for beer pong robots). Includes a full pipeline from
rendering through YOLOv8 fine-tuning.

## Published Artifacts

- **Dataset**: [pwalker/beer-pong-synthetic](https://huggingface.co/datasets/pwalker/beer-pong-synthetic) — 1,000 rendered images with 3D cup metadata
- **Model**: [pwalker/beer-pong-yolov8](https://huggingface.co/pwalker/beer-pong-yolov8) — YOLOv8n fine-tuned on the dataset (mAP50=0.917)

## Quick Start

```bash
# Install system deps (for headless Blender)
sudo bash install-debs.sh

# Generate dataset (250 scenes x 4 views = 1,000 images, ~1.7h on RTX 4090)
./generate.sh --num-scenes 250 --views-per-scene 4

# Prepare YOLO labels and train/val/test split
python3 prepare_yolo_dataset.py

# Fine-tune YOLOv8 and evaluate (~5 min on RTX 4090)
python3 train_yolo.py

# Upload to Hugging Face (requires `huggingface-cli login`)
python3 upload_dataset.py
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `generate_dataset.py` | BlenderProc rendering pipeline — randomizes cups, table, lighting, camera, and outputs PNG images + JSON metadata with 3D positions and camera matrices |
| `generate.sh` | Shell wrapper that sets up the venv and runs `generate_dataset.py` via `blenderproc run` |
| `make_yolo_labels.py` | Projects 3D bounding box corners into 2D to produce YOLO-format labels. Supports `--margin-top` to crop to cup rims only |
| `prepare_yolo_dataset.py` | Splits dataset into train/val/test and creates `dataset.yaml` for ultralytics |
| `train_yolo.py` | Fine-tunes YOLOv8n and evaluates on the test set |
| `upload_dataset.py` | Uploads the dataset to Hugging Face Hub |

## Scene Randomization

Each scene varies:
- Cup count per side (0–10), with random removals from triangle formations
- Tipped-over cups (5% chance per cup)
- Table color (8 variants), floor color (7 variants), background color
- 3-point lighting with random position, intensity, and color temperature
- Ping pong ball positions (table, floor, hidden, mid-air)
- Camera at humanoid height (1.5–1.9m), random angle around the table

## Dataset Format

```
/workspace/beer_pong_dataset/
  images/               PNG images (1920x1080)
  labels/               JSON metadata per image:
                          - camera_intrinsics (3x3 K matrix)
                          - camera_extrinsics_4x4 (camera pose in world)
                          - cups[].position_3d, rotation_euler_rad
                          - cups[].bbox_3d_corners (8 world-space corners)
                          - cups[].tipped_over, cups[].side
  yolo_labels_cup_top/  YOLO format txt files (class cx cy w h)
```

2D bounding boxes are derived by projecting the 3D bbox corners through the
camera matrices — no manual annotation needed.

## Results

See [FINETUNING.md](FINETUNING.md) for full training details. Summary:

| Metric | Score |
|--------|-------|
| mAP50 | 0.917 |
| mAP50-95 | 0.671 |
| Precision | 0.926 |
| Recall | 0.858 |
