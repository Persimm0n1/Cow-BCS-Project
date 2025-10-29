from ultralytics import YOLO
import os

# Create the models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

MODEL_SAVE_PATH = 'models/yolov8n.pt' # 'n' is the small, fast "nano" version

print("Downloading YOLOv8n model...")
model = YOLO('yolov8n.pt') 
model.save(MODEL_SAVE_PATH)

print(f"Success! YOLO model saved to: {MODEL_SAVE_PATH}")