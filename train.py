from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data="config_yolo.yaml", device="mps", plots=True)