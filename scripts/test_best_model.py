from ultralytics import YOLO

model = YOLO(
    r"C:\Users\DELL\lidar-sementic-mapping\runs\detect\runs\envodat_training\weights\best.pt"
)

results = model.predict(
    source=r"C:\Users\DELL\mu-hall-yolov8\test\images",
    save=True,
    conf=0.25
)

print("Inference completed using trained model!")