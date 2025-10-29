import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from ultralytics import YOLO
from xgboost import XGBClassifier
from PIL import Image

# --- 1. INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- 2. LOAD YOUR 3 MODELS INTO MEMORY ---
print("Loading models, please wait...")

# Define paths to your models (assuming they are in a 'models' folder)
YOLO_PATH = 'models/yolov8.pt' 
EXTRACTOR_PATH = 'models/feature_extractor.h5'
SCORER_PATH = 'models/xgb_scorer.json'

# Part 1: YOLO Detector
yolo_model = YOLO(YOLO_PATH) 

# Part 2: VGG/ResNet Feature Extractor
IMG_WIDTH, IMG_HEIGHT = 224, 224
feature_extractor = tf.keras.models.load_model(EXTRACTOR_PATH)

# Part 3: XGBoost Scorer
xgb_model = XGBClassifier()
xgb_model.load_model(SCORER_PATH)

# ======================================================================
# !!! IMPORTANT !!!
# YOU MUST CREATE THIS YOURSELF.
# This map links the "column number" from your feature vector
# to a human-readable "Reason".
# You find these numbers by training your XGBoost model in Colab
# and printing the "feature importance"
# ======================================================================
REASONS_MAP = {
    512: "Lack of fat cover on loin",
    2048: "Rump angle needs improvement",
    1024: "Ribs are too visible",
    4000: "Tail head has insufficient fat",
    # ... add all your top features here
}
print("Models loaded successfully!")

# --- 3. HELPER FUNCTION FOR IMAGE PREPROCESSING ---
def preprocess_image(img_pil):
    """Converts a PIL image to the format VGG/ResNet expects."""
    img = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add a 'batch' dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array) 
    return img_array

# --- 4. DEFINE THE API ENDPOINT ---
# This creates a URL for your PHP to call: "http://your-site.com/predict"
@app.route("/predict", methods=["POST"])
def predict():
    """Main function to handle the image and return a score."""
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    try:
        img_pil = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # --- MODEL PIPELINE ---
    
    # === Part 1: Run YOLO Cow Detector ===
    results = yolo_model(img_pil)
    
    if len(results[0].boxes) == 0:
        return jsonify({"error": "No cow detected in the image"}), 400

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    cropped_img = img_pil.crop(box)

    # === Part 2: Run Feature Extractor ===
    preprocessed_img = preprocess_image(cropped_img)
    feature_vector = feature_extractor.predict(preprocessed_img)

    # === Part 3: Run XGBoost Scorer ===
    score_prediction = xgb_model.predict(feature_vector)
    importances = xgb_model.feature_importances_
    
    top_feature_indices = np.argsort(importances)[-3:][::-1] # Get top 3 indices
    
    reasons = []
    for idx in top_feature_indices:
        if idx in REASONS_MAP:
            reasons.append(REASONS_MAP[idx])

    # --- 5. RETURN THE FINAL JSON RESPONSE ---
    final_score = int(score_prediction[0])
    scaled_score = final_score * 2 # Scale from 1-5 to 1-10

    return jsonify({
        "score": scaled_score,
        "reasons": reasons
    })

# --- 6. RUN THE FLASK APP ---
if __name__ == "__main__":
    # The 'port=int(os.environ.get("PORT", 8080))' is important for hosting platforms
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
