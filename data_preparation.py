import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32 
DATA_DIR = 'path/to/your/main_dataset_directory'

VALIDATION_SPLIT = 0.2 


# A. Training Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,            
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    rotation_range=20,         
    width_shift_range=0.1,     
    height_shift_range=0.1,    
    validation_split=VALIDATION_SPLIT 
)
# B. Testing Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

# 1. Training Generator
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  
    subset='training'     
)

# 2. Validation Generator
validation_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'   
)
test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'test'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False 
)

print("\n--- Dataset Summary ---")
print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {validation_generator.samples}")
print(f"Total testing images: {test_generator.samples}")
print(f"Class Indices: {train_generator.class_indices}") 

def get_data_generators():
    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    print("Data preparation complete. Data generators are ready.")

