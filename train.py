from ultralytics import YOLO

if __name__ == '__main__': 
    model = YOLO("yolo11m.pt") 
    model.train(data="dataset_custom.yaml", imgsz=640, batch=8,
                epochs=100, workers=0, device=0) 