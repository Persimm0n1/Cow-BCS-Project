import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from ultralytics import YOLO
from xgboost import XGBClassifier
from PIL import Image
import json

# --- 1. INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- 2. LOAD YOUR MODELS AND MAPPINGS ---
print("Loading models, please wait...")

# Define paths
YOLO_PATH = 'models/yolov8n.pt' # Using the 'n' model we downloaded
EXTRACTOR_PATH = 'models/feature_extractor.h5'
SCORER_PATH = 'models/xgb_scorer.json'
CLASSES_PATH = 'models/score_classes.json' # Loading the score mapping file

# Part 1: YOLO Detector
yolo_model = YOLO(YOLO_PATH)

# Part 2: VGG/ResNet Feature Extractor
IMG_WIDTH, IMG_HEIGHT = 224, 224
feature_extractor = tf.keras.models.load_model(EXTRACTOR_PATH)

# Part 3: XGBoost Scorer
xgb_model = XGBClassifier()
xgb_model.load_model(SCORER_PATH)

# Part 3b: Load the Score Class Mapping
with open(CLASSES_PATH, 'r') as f:
    SCORE_CLASSES = json.load(f) # List like [2.0, 2.5, 3.0, ...]
print(f"Loaded score classes: {SCORE_CLASSES}")

# ======================================================================
# !!! IMPORTANT: REASONS_MAP UPDATED WITH YOUR NUMBERS !!!
# You MUST change the text descriptions (e.g., "Reason 1") to match
# what these features actually represent based on your project plan
# (e.g., "Loin Cover", "Rib Visibility", "Rump Angle").
# ======================================================================
REASONS_MAP = {
    9175: "Reason 1 (Top Feature)",       # Importance: 0.0036
    15869: "Reason 2",                     # Importance: 0.0034
    1693: "Reason 3",                      # Importance: 0.0032
    9396: "Reason 4",                      # Importance: 0.0031
    18732: "Reason 5",                     # Importance: 0.0030
    11119: "Reason 6",                     # Importance: 0.0030
    11787: "Reason 7",                     # Importance: 0.0030
    9261: "Reason 8",                      # Importance: 0.0029
    18555: "Reason 9",                     # Importance: 0.0029
    11931: "Reason 10 (Lowest Importance)", # Importance: 0.0028
}
print("Models and mappings loaded successfully!")

# --- 3. HELPER FUNCTION FOR IMAGE PREPROCESSING ---
def preprocess_image(img_pil):
    """Converts a PIL image to the format VGG/ResNet expects."""
    img = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add a 'batch' dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# --- ADD THIS SIMPLE TEST ROUTE ---
@app.route("/", methods=["GET"])
def hello():
    """A simple route to check if the server is responding."""
    print("Received request to / route!") # Add this print statement
    return jsonify({"message": "Flask server is running!"})
# --- END OF TEST ROUTE ---

# --- 4. DEFINE THE API ENDPOINT ---
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

    # Predicts the 0-indexed class (e.g., 2)
    score_prediction_index = xgb_model.predict(feature_vector)

    # Get the *actual score* (e.g., 3.5) from our loaded mapping
    class_index = int(score_prediction_index[0])

    if class_index < 0 or class_index >= len(SCORE_CLASSES):
        return jsonify({"error": f"Model predicted an invalid class index: {class_index}"}), 500

    final_score = SCORE_CLASSES[class_index]

    # Get Top 3 Reasons based on overall feature importance
    importances = xgb_model.feature_importances_
    top_feature_indices = np.argsort(importances)[-3:][::-1] # Get top 3 indices

    reasons = []
    for idx in top_feature_indices:
        if idx in REASONS_MAP:
            reasons.append(REASONS_MAP[idx])

    # --- 5. RETURN THE FINAL JSON RESPONSE ---
    return jsonify({
        "score": final_score,
        "reasons": reasons
    })

# --- 6. RUN THE FLASK APP ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

