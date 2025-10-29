import pandas as pd
from xgboost import XGBClassifier
import os
import json

# --- 1. CONFIGURE YOUR PATHS ---
TRAINING_CSV_PATH = 'training_data.csv'
MODEL_SAVE_PATH = 'models/xgb_scorer.json'
CLASSES_SAVE_PATH = 'models/score_classes.json' # <-- New file to save the mapping

# --- 2. LOAD YOUR TRAINING DATA ---
print(f"Loading data from: {TRAINING_CSV_PATH}")
df = pd.read_csv(TRAINING_CSV_PATH).dropna()

X = df.drop('bcs_score_label', axis=1)
y = df['bcs_score_label']

# --- 3. TRAIN THE XGBOOST MODEL ---
print("Training XGBoost model...")

# Convert scores to 0-indexed integer classes
# y_labels is the 0,1,2... array for training
# unique_scores is the mapping array [2.0, 2.5, 3.0, 3.5...]
y_labels, unique_scores = pd.factorize(y, sort=True) # Sort to ensure consistent order
num_classes = len(unique_scores)

print(f"Found {num_classes} unique score classes: {list(unique_scores)}")

xgb_model = XGBClassifier(objective='multi:softmax', 
                          num_class=num_classes, 
                          n_estimators=100,
                          learning_rate=0.1,
                          use_label_encoder=False,
                          eval_metric='mlogloss')

xgb_model.fit(X, y_labels) # Train on the 0-indexed labels
print("Model training complete.")

# --- 4. SAVE THE MODEL ---
os.makedirs('models', exist_ok=True)
xgb_model.save_model(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# --- 5. SAVE THE CLASS MAPPING (THE CRITICAL NEW STEP) ---
# Convert numpy/pd.Index to a plain list for JSON
class_mapping_list = [float(score) for score in unique_scores] 
with open(CLASSES_SAVE_PATH, 'w') as f:
    json.dump(class_mapping_list, f)
print(f"Score mapping saved to: {CLASSES_SAVE_PATH}")

# --- 6. PRINT FEATURE IMPORTANCES (YOUR REASONS) ---
print("\n" + "="*40)
print("--- TOP 10 IMPORTANT FEATURES (REASONS) ---")
print("="*40)
importances = xgb_model.feature_importances_
top_indices = importances.argsort()[-10:][::-1]
print("(Use these 'Column Index' numbers in your 'REASONS_MAP' in app.py)\n")
for i in top_indices:
    feature_name = X.columns[i] 
    column_index = int(feature_name.split('_')[1])
    print(f"Column Index: {column_index}    (Importance: {importances[i]:.4f})")
print("="*40)
