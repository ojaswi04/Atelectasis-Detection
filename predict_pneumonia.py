import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from _03_train_model import MODEL_SAVE_DIR, MODEL_FILENAME 

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
IMAGE_SIZE = (224, 224) 
CLASS_LABELS = ['NORMAL', 'PNEUMONIA']
THRESHOLD = 0.5 # Classification threshold (50% probability)

#  aditi REPLACE THIS PATH with a path to a single X-ray image for testing !!!
SINGLE_IMAGE_PATH = 'path/to/a/new/single_xray_image.jpeg'


def preprocess_single_image(img_path):
    """
    Loads, resizes, and normalizes a single image for prediction.
    """
    try:
        # Load the image and resize it to the model's required input size
        img = image.load_img(img_path, target_size=IMAGE_SIZE, color_mode='rgb')
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch of size 1 (required by Keras)
    # Shape changes from (224, 224, 3) to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values (scale to 0-1), consistent with data_preparation.py
    img_array = img_array / 255.0
    
    return img_array

def predict_single_image(img_path):
    """
    Loads the model and performs prediction on a single image.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure '03_train_model.py' has been run successfully.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    preprocessed_img = preprocess_single_image(img_path)
    if preprocessed_img is None:
        return

    print(f"Making prediction on image: {os.path.basename(img_path)}")

    # The output is a probability score between 0 and 1 for the positive class (PNEUMONIA)
    raw_probability = model.predict(preprocessed_img)[0][0]
    
  
    confidence_pneumonia = raw_probability * 100
    
    # Determine the final classification and report confidence as a percentage
    if raw_probability >= THRESHOLD:
        prediction_label = CLASS_LABELS[1] # PNEUMONIA
        final_confidence = confidence_pneumonia
        confidence_label = f"{final_confidence:.2f}%"
    else:
        prediction_label = CLASS_LABELS[0] # NORMAL
        final_confidence = 100.0 - confidence_pneumonia
        confidence_label = f"{final_confidence:.2f}%"
        
    print("\n--- Prediction Result ---")
    print(f"Image Classified As: **{prediction_label}**")
    print(f"Model Confidence: {confidence_label}")
    
    # Optional: Display the image
    img_display = image.load_img(img_path, target_size=(250, 250))
    plt.imshow(img_display)
    plt.title(f"Prediction: {prediction_label} ({confidence_label})")
    plt.axis('off')
    plt.show() 


if __name__ == '__main__':
    predict_single_image(SINGLE_IMAGE_PATH)
