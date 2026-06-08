import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import datetime

from _01_data_preparation import get_data_generators
from _02_model_architecture import create_transfer_learning_model

EPOCHS = 20 
MODEL_SAVE_DIR = 'model_outputs'
MODEL_FILENAME = 'best_pneumonia_detector.h5'

# Ensure the model output directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)


def train_pneumonia_model():
    """
    Loads data generators, creates the model, defines callbacks, and trains the model.
    """
    
    # 1. Load Data Generators
    print("Loading data generators...")
    train_generator, validation_generator, test_generator = get_data_generators()
    
    # Calculate steps per epoch for efficiency
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    
    # 2. Create Model
    print("Creating model architecture (VGG16 Transfer Learning)...")
    model = create_transfer_learning_model()
    
    # 3. Define Callbacks
    # Callbacks are functions executed during training to save the model, adjust learning rate, etc.
    
    # Saves the model weights only when validation accuracy improves
    checkpoint = ModelCheckpoint(
        filepath=SAVE_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max' # Higher is better for accuracy
    )
    
    # Stops training if validation loss does not improve for 'patience' epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True # Restores the weights of the best epoch
    )

    # Reduces the learning rate if validation loss plateaus, helping the model converge better
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Reduce LR by 80% (new_lr = current_lr * 0.2)
        patience=3,
        min_lr=0.000001,
        verbose=1
    )

    callbacks_list = [checkpoint, early_stopping, reduce_lr]
    
    # 4. Model Training
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\nTraining complete.")
    print(f"Best model weights saved to: {SAVE_PATH}")
    
    # Return the training history for visualization later
    return model, history, test_generator

if __name__ == '__main__':
    # Execute the training process
    trained_model, training_history, final_test_generator = train_pneumonia_model()
    
    # You can now proceed to evaluate the model using the test_generator
    print("\nNext step: Run '04_evaluate_and_report.py' for final evaluation.")
