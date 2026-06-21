from ultralytics import YOLO

model=YOLO("yolov8n.pt")

results=model.predict(source=r"C:\Users\DELL\mu-hall-yolov8\test\images",
save=True,
conf=0.25)

print("Interference complete")