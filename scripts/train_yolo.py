from ultralytics import YOLO
model= YOLO("yolov8n.pt")

results = model.train(
    data=r"C:\Users\DELL\mu-hall-yolov8\envodata.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    project="runs",
    name="envodat_training"
)

print("Training completed!")