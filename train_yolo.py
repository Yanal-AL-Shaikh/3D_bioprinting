from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="config/yanal2.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    lr0=0.01
)
