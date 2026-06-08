import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import seaborn as sns
from _01_data_preparation import get_data_generators # Used just to get the test_generator
from _03_train_model import MODEL_SAVE_DIR, MODEL_FILENAME 

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
CLASS_LABELS = ['NORMAL', 'PNEUMONIA']


def evaluate_pneumonia_model():
    """
    Loads the best trained model, evaluates it on the test set, and reports all metrics.
    """
    
    # 1. Load Data Generator (Test Set)
    print("Loading test data generator...")
    # We only need the test_generator from the data preparation file
    _, _, test_generator = get_data_generators()
    
    # Calculate steps for evaluation
    test_steps = test_generator.samples // test_generator.batch_size
    
    # 2. Load the Best Trained Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure '03_train_model.py' has been run successfully.")
        return

    print(f"Loading best model from: {MODEL_PATH}")
    # Load the best weights saved by ModelCheckpoint during training
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating model on the test set...")
    loss, acc, prec, rec = model.evaluate(test_generator, steps=test_steps, verbose=1)
    
    print("\n--- Core Evaluation Metrics ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")

    # Reset generator to ensure predictions start from the beginning of the test set
    test_generator.reset()
    
    # Get raw probability predictions (scores between 0 and 1)
    Y_pred_raw = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
    
    # Convert probabilities to binary class predictions (0 or 1) using a 0.5 threshold
    Y_pred_classes = (Y_pred_raw > 0.5).astype(int).flatten()
    
    # Get true labels from the test generator
    Y_true = test_generator.classes[:len(Y_pred_classes)]
    print("\n--- Detailed Classification Report ---")
    print(classification_report(Y_true, Y_pred_classes, target_names=CLASS_LABELS))
    cm = confusion_matrix(Y_true, Y_pred_classes)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 7. ROC Curve and AUC Score (Crucial for Medical Models)
    # The raw probability for the positive class (PNEUMONIA) is at index 0
    Y_pred_proba = Y_pred_raw.flatten() 
    
    fpr, tpr, _ = roc_curve(Y_true, Y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show() 
    
    print(f"\nArea Under the Curve (AUC): {roc_auc:.4f}")


if __name__ == '__main__':
    evaluate_pneumonia_model()
