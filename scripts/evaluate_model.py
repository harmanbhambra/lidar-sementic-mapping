from ultralytics import YOLO

model = YOLO(
    r"C:\Users\DELL\lidar-sementic-mapping\runs\detect\runs\envodat_training\weights\best.pt"
)

metrics = model.val(
    data=r"C:\Users\DELL\mu-hall-yolov8\envodata.yaml"
)

print(metrics)