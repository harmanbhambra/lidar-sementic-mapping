import os
import sys
import cv2
import json


# Project paths
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from detectors.grounding_dino import GroundingDINODetector
from segmentation import SAMSegmenter
from visualization import Visualizer
from semantic import DetectedObject, Scene

# Model paths
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

SAM_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sam_vit_h_4b8939.pth"
)

# Generalization test image
IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "generalization_test",
    "test_image.jpg"
)

# Separate output folder
OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "generalization_test"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Text concepts to search for
TEXT_PROMPT = (
    "vase . lamp . bed . chair . mirror . window . "
    "flower pot . table"
)

print("=== GENERALIZATION TEST ===")
print(f"Image: {IMAGE_PATH}")
print(f"Prompt: {TEXT_PROMPT}")

# Load models
detector = GroundingDINODetector(
    config_path=CONFIG_PATH,
    weights_path=WEIGHTS_PATH
)

segmenter = SAMSegmenter(
    checkpoint_path=SAM_CHECKPOINT,
    model_type="vit_h",
    device="cpu"
)

visualizer = Visualizer(alpha=0.5)


# -------------------------
# Detection
# -------------------------

print("\nStep 1: Starting detection...")

image_source, boxes, logits, phrases = detector.detect(
    image_path=IMAGE_PATH,
    text_prompt=TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25
)

print(f"Detection complete. Found {len(boxes)} objects.")


# -------------------------
# Segmentation
# -------------------------

print("Step 2: Starting segmentation...")

masks = segmenter.segment(
    image=image_source,
    boxes=boxes
)

print(f"Generated {len(masks)} segmentation masks.")


# -------------------------
# Semantic objects
# -------------------------

detected_objects = []

for box, score, label, mask in zip(
    boxes,
    logits,
    phrases,
    masks
):

    detected_object = DetectedObject(
        label=label,
        confidence=score.item(),
        bounding_box=box,
        mask=mask
    )

    detected_objects.append(detected_object)


scene = Scene(
    image=image_source,
    objects=detected_objects
)


# -------------------------
# Visualization
# -------------------------

annotated_image = visualizer.draw(
    image=scene.image,
    objects=scene.objects
)

output_image_path = os.path.join(
    OUTPUT_FOLDER,
    "generalization_result.jpg"
)

cv2.imwrite(
    output_image_path,
    annotated_image
)


# -------------------------
# Save semantic JSON
# -------------------------

scene_data = scene.to_dict()

json_output_path = os.path.join(
    OUTPUT_FOLDER,
    "generalization_scene.json"
)

with open(json_output_path, "w") as file:
    json.dump(scene_data, file, indent=4)


# -------------------------
# Results
# -------------------------

print("\n=== GENERALIZATION RESULTS ===")

print(f"Total objects detected: {len(scene.objects)}")

for obj in scene.objects:
    print(f"  - {obj.label}: {obj.confidence:.2f}")

print(f"\nAnnotated image saved to: {output_image_path}")
print(f"Semantic scene saved to: {json_output_path}")