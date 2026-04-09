"""
Download the beer-pong YOLOv8 model from Hugging Face and export to ONNX format.

Requirements:
    pip install ultralytics huggingface_hub

Usage:
    python convert_model.py
"""

import json
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download the best weights from Hugging Face
print("Downloading model from pwalker/beer-pong-yolov8...")
model_path = hf_hub_download(
    repo_id="pwalker/beer-pong-yolov8",
    filename="weights/best.pt",
)

# Load the model
print(f"Loading model from {model_path}...")
model = YOLO(model_path)

# Extract class names
names = model.names
print(f"Class names: {names}")

# Save class names as JSON for the web app
with open("model/classes.json", "w") as f:
    json.dump(names, f, indent=2)
print("Saved class names to model/classes.json")

# Export to ONNX
print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, simplify=True, opset=17)

# The exported file will be next to the .pt file; copy it to our model dir
import shutil
from pathlib import Path

onnx_path = Path(model_path).with_suffix(".onnx")
dest = Path("model/best.onnx")
shutil.copy2(onnx_path, dest)
print(f"Copied ONNX model to {dest}")
print("Done! You can now serve the site.")
