from ultralytics import YOLO
model= YOLO("yolov8s.pt")

results = model.train(
    data=r"C:\Users\DELL\mu-hall-yolov8\envodata.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    workers=0,
    project="runs",
    name="yolov8s_training"
)

print("Training completed!")