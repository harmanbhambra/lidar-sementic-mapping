from grounding_dino import GroundingDINODetector
from groundingdino.util.inference import annotate

import cv2
import os


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinT_OGC.py"
)

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "groundingdino_swint_ogc.pth"
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "bus.jpg"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "bus_prediction.jpg"
)


# Create the detector
detector = GroundingDINODetector(CONFIG_PATH, WEIGHTS_PATH)

# Run detection
image_source, boxes, logits, phrases = detector.detect(
    IMAGE_PATH,
    "bus"
)

print("\nDetected Objects:")
print(phrases)

print("\nConfidence Scores:")
print(logits)

# Draw detections
annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases,
)

cv2.imwrite(OUTPUT_PATH, annotated_frame)

print(f"\nPrediction saved to:\n{OUTPUT_PATH}")