from ultralytics import YOLO
import os
import json

# Load your trained model
model = YOLO(
    r"C:\Users\DELL\lidar-sementic-mapping\runs\detect\runs\yolov8s_training\weights\best.pt"
)

image_folder = r"C:\Users\DELL\mu-hall-yolov8\test\images"

semantic_data = {}

for image_name in os.listdir(image_folder):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_folder, image_name)

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    detections = {}

    for result in results:

        for cls in result.boxes.cls:

            class_name = result.names[int(cls)]

            if class_name in detections:
                detections[class_name] += 1
            else:
                detections[class_name] = 1

    semantic_data[image_name] = detections

# Save JSON
with open("semantic_map.json", "w") as f:
    json.dump(semantic_data, f, indent=4)

print("Semantic map saved to semantic_map.json")