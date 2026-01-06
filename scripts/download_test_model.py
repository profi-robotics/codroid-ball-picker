#!/usr/bin/env python3
"""Download a YOLO model for testing object detection."""

from pathlib import Path
from ultralytics import YOLO

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Download YOLOv8n (nano) - smallest and fastest for testing
print("Downloading YOLOv8n model...")
model = YOLO("yolov8n.pt")

# The model is automatically downloaded to ~/.ultralytics/
# Copy it to our models directory
model_path = MODEL_DIR / "yolov8n.pt"
print(f"Saving model to {model_path}")
model.save(model_path)

print(f"\n✓ Model ready at: {model_path}")
print("\nYOLOv8n is trained on COCO dataset with 80 classes:")
print("  0: person, 39: bottle, 41: cup, 32: sports ball, etc.")
print("\nUpload this model in the Detection tab to test.")
