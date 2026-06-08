import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# --- Configuration Settings (Must match data_preparation.py) ---
IMAGE_SIZE = (224, 224) 
INPUT_SHAPE = IMAGE_SIZE + (3,) # 3 channels for RGB, even if X-rays are grayscale
LEARNING_RATE = 0.0001
METRICS = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]


def create_transfer_learning_model():
    """
    Creates a pneumonia detection model using Transfer Learning (VGG16).
    """
    # include_top=False: Excludes the model's original final classification layer
    # weights='imagenet': Uses weights pre-trained on the massive ImageNet dataset
    base_model = VGG16(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
  
    # We freeze the VGG16 layers so their weights are not updated during initial training.
    # This prevents the pre-trained features from being corrupted by the new, small dataset.
    for layer in base_model.layers:
        layer.trainable = False
        
    # This part takes the features extracted by VGG16 and performs the binary classification.
    model = Sequential([
        base_model, # VGG16 feature extractor
        Flatten(),  # Converts 2D feature maps to a 1D vector
        
        # Custom Layers for Classification
        Dense(512, activation='relu'),
        BatchNormalization(), # Improves training speed and stability
        Dropout(0.5), # Regularization to prevent overfitting
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Output Layer: Single neuron with Sigmoid activation for binary classification.
        # Output is a probability (Pneumonia risk) between 0 and 1.
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        # Binary Cross-Entropy is the standard loss for binary classification problems
        loss='binary_crossentropy',
        metrics=METRICS
    )
    model.summary()
    
    return model

if __name__ == '__main__':
    # Test the model creation and print its summary
    pneumonia_model = create_transfer_learning_model()
    print("\nModel architecture defined successfully.")
