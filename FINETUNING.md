# YOLOv8 Fine-Tuning on Synthetic Beer Pong Data

## Dataset

Generated 1,000 images (250 scenes x 4 camera views) using BlenderProc with
randomized:

- Cup count per side (0-10), with random removals from triangle formations
- Tipped-over cups (5% chance per cup)
- Table color (8 variants: wood, black, grey, green, blue, light wood, white, red)
- Floor color (7 variants)
- Background color (dark/dim, randomized RGB)
- 3-point lighting (key/fill/back) with random position, intensity, color temperature
- Ping pong ball positions (table, floor, hidden, mid-air)
- Camera at humanoid height (1.5-1.9m), random angle around table

Each image has a companion JSON file with full 3D metadata (cup positions,
rotations, 3D bounding box corners, camera intrinsics/extrinsics). YOLO labels
are derived by projecting 3D bounding box corners into 2D and computing
axis-aligned bounding boxes, cropped to the top 40% of each cup (`--margin-top 0.6`)
to focus on the cup rim.

**Split**: 800 train / 100 val / 100 test (80/10/10, shuffled with seed 42)

**Cup visibility**: 9,851 / 9,992 total cups (98.6%) were visible after
projection. A cup is considered visible when >= 30% of its 3D bounding box
corners project in front of the camera with a nonzero 2D bounding box within
image bounds.

## Model

- **Architecture**: YOLOv8n (nano) — 3M parameters, 8.1 GFLOPs
- **Base weights**: COCO-pretrained `yolov8n.pt`
- **Input size**: 640x640
- **Classes**: 1 (`cup_top`)

## Training

- **Epochs**: 50 (with patience=10 early stopping, ran all 50)
- **Batch size**: 16
- **Optimizer**: auto (AdamW)
- **Learning rate**: 0.01 -> 0.0001 (cosine)
- **Augmentation**: default ultralytics (mosaic, randaugment, hsv jitter, fliplr)
- **Hardware**: NVIDIA RTX 4090
- **Duration**: 5.2 minutes (0.087 hours)

## Results (Test Set)

| Metric     | Score |
|------------|-------|
| mAP50      | 0.917 |
| mAP50-95   | 0.671 |
| Precision  | 0.926 |
| Recall     | 0.858 |

Confusion matrix: 93% true positive rate, 7% false negative rate (missed cups),
~0% false positive rate (no hallucinated detections from background).

## Failure Modes

The main failure mode is small/distant cups being missed — cups far from the
camera are only a few pixels wide at 640px input resolution. This accounts for
most of the 7% FN rate. No false positives were observed.

## Reproduction

```bash
# 1. Generate dataset (~1.7 hours on RTX 4090)
./generate.sh --num-scenes 250 --views-per-scene 4

# 2. Prepare YOLO dataset with cup-top labels
python3 prepare_yolo_dataset.py

# 3. Train and evaluate (~5 min on RTX 4090)
python3 train_yolo.py
```

## Output Locations

| Path | Contents |
|------|----------|
| `/workspace/beer_pong_dataset/images/` | 1,000 rendered PNGs (1920x1080) |
| `/workspace/beer_pong_dataset/labels/` | JSON metadata (3D positions, camera matrices) |
| `/workspace/beer_pong_yolo/` | YOLO-formatted dataset + dataset.yaml |
| `/workspace/beer_pong_yolo/runs/cup_detector/weights/best.pt` | Best model weights |
