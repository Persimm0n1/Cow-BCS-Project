import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import os
from tqdm import tqdm # A progress bar library

# --- 1. CONFIGURE YOUR PATHS ---

# Path to your raw VIA annotation file
ANNOTATION_CSV_PATH = 'via_project_25Oct2025_19h14m_csv (1).csv'

# Path to the folder containing all your .jpg image files
# !!! UPDATE THIS PATH !!!
IMAGE_FOLDER_PATH = r'C:\Users\naman\Cow-BCS-Project\images'

# Path to your feature extractor model
EXTRACTOR_PATH = 'models/feature_extractor.h5'

# Name for your final output file
OUTPUT_CSV_PATH = 'training_data.csv'

# Model settings
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- 2. LOAD THE MODEL AND ANNOTATIONS ---

print(f"Loading feature extractor: {EXTRACTOR_PATH}")
feature_extractor = tf.keras.models.load_model(EXTRACTOR_PATH)

print(f"Loading annotation file: {ANNOTATION_CSV_PATH}")
df_annotations = pd.read_csv(ANNOTATION_CSV_PATH)

# --- 3. PREPARE THE DATA ---

# Your CSV has multiple rows per image. We only need unique filenames.
# We also parse the JSON to get the 'bcs_score'
unique_files = {}
for index, row in df_annotations.iterrows():
    filename = row['filename']
    if filename not in unique_files:
        try:
            # The score is stored as a JSON string, e.g., {"bcs_score":"2.5"}
            attributes = json.loads(row['file_attributes'])
            score = attributes.get('bcs_score')
            
            if score is not None:
                unique_files[filename] = float(score)
        except json.JSONDecodeError:
            print(f"Warning: Skipping row for {filename}, invalid JSON in 'file_attributes'")

print(f"Found {len(unique_files)} unique images to process.")

# This list will hold all our final data
processed_data = []

# --- 4. RUN THE FEATURE EXTRACTION LOOP ---

# Use tqdm for a nice progress bar
for filename, score in tqdm(unique_files.items(), desc="Processing Images"):
    
    full_image_path = os.path.join(IMAGE_FOLDER_PATH, filename)
    
    if not os.path.exists(full_image_path):
        print(f"Warning: Skipping {filename}, file not found at {full_image_path}")
        continue

    try:
        # Load the image
        img = load_img(full_image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Add 'batch' dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for the VGG model
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        # Get the feature vector
        feature_vector = feature_extractor.predict(img_array, verbose=0)
        
        # Flatten the vector from 2D to 1D
        feature_vector_flat = feature_vector.flatten()
        
        # Add the score to the end of the feature vector
        final_row = np.append(feature_vector_flat, score)
        processed_data.append(final_row)
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# --- 5. SAVE THE FINAL CSV ---

print("\nLoop finished. Creating final DataFrame...")

# Get the number of features from the first row
num_features = len(processed_data[0]) - 1 

# Create column names: feature_0, feature_1, ..., bcs_score_label
columns = [f'feature_{i}' for i in range(num_features)]
columns.append('bcs_score_label')

# Create the final DataFrame
df_final = pd.DataFrame(processed_data, columns=columns)

# Save to disk
df_final.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- SUCCESS! ---")
print(f"Final training data saved to: {OUTPUT_CSV_PATH}")