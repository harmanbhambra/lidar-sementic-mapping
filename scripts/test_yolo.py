from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Run object detection on sample image
results = model(
    "https://ultralytics.com/images/bus.jpg",
    save=True
)

print("Detection complete!")