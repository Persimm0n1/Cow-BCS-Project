import tensorflow as tf
import os

# Create the models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

MODEL_SAVE_PATH = 'models/feature_extractor.h5'

print("Loading and saving pre-trained VGG16 model...")

base_model = tf.keras.applications.VGG16(weights='imagenet', 
                                         include_top=False, 
                                         input_shape=(224, 224, 3))
base_model.trainable = False 

model_input = tf.keras.Input(shape=(224, 224, 3))
feature_block = base_model(model_input, training=False)
feature_vector = tf.keras.layers.Flatten()(feature_block)

feature_extractor_model = tf.keras.Model(inputs=model_input, outputs=feature_vector)

feature_extractor_model.save(MODEL_SAVE_PATH)

print(f"Success! Feature extractor saved to: {MODEL_SAVE_PATH}")