from ultralytics import YOLO

def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

def predict(model, image_path):
    results = model(image_path)
    return results[0]  # 첫 결과 반환