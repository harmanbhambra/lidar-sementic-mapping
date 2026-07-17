import os
from detectors.grounding_dino import GroundingDINODetector
from groundingdino.util.inference import annotate
import cv2

#where the project is located
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

CONFIG_PATH=os.path.join(PROJECT_ROOT, "GroundingDINO","groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

WEIGHTS_PATH=os.path.join(PROJECT_ROOT, "models", "groundingdino_swint_ogc.pth")

DATASET_PATH= r"C:\Users\DELL\mu-hall-yolov8"

IMAGE_PATH=os.path.join(DATASET_PATH, "train","images")

OUTPUT_FOLDER= os.path.join(PROJECT_ROOT, "results")

image_files = sorted(
    file
    for file in os.listdir(IMAGE_PATH)
    if file.lower().endswith((".jpg", ".png", ".jpeg"))
)

print(f"Found {len(image_files)} images.")

#Here we are going to instantiate the detector. It calls the init method which loads the config, the trained weights and creates the model
detector= GroundingDINODetector(config_path= CONFIG_PATH, weights_path=WEIGHTS_PATH)

first_image=image_files[0]
image_path=os.path.join(IMAGE_PATH, first_image)

#called the detector
image_source, boxes, logits, phrases = detector.detect(
    image_path=image_path,
    text_prompt="chair . table . person . laptop . door . window",
    box_threshold=0.35,
    text_threshold=0.25
)

#Print the result
print("\nDetected Objects:")

for phrase, score in zip(phrases, logits):
    print(f"{phrase}: {score:.2f}")

