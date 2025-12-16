# Standard library
import os
import numpy as np

# Keras utilities for image preprocessing and augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# TensorFlow and model building blocks (MobileNetV2 transfer learning)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Training helpers: callbacks for checkpointing, early stopping and LR scheduling
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# For plotting training history
import matplotlib.pyplot as plt


# Path configuration: set where your dataset 'train' and 'val' folders live
base_dir = 'Dataset' 

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

IMG_SIZE = 224 
BATCH_SIZE = 32

# Data augmentation for the training set: use mild transforms appropriate for faces
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator: apply the same preprocessing but no augmentation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,)

# Load datasets from folders. The directory structure should be: train/<class>/*.jpg
print("--- Loading data ---")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"--> Found {num_classes} emotion classes: {list(train_generator.class_indices.keys())}")

# Build model: MobileNetV2 as a feature extractor + custom head for our classes
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base model during initial training to train only the new head
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model: optimizer, loss and metrics
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks: save best model, stop early and reduce LR on plateau
checkpoint = ModelCheckpoint(
    'Emotion_With_MobileNetV2.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# Phase 1: train the head layers while the base is frozen
print("\n -----Starting training...")
history = model.fit(
    train_generator,
    epochs=30, 
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

print("-----Phase 1 of training complete!-----")

base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Unfreeze the top layers of the base model and continue training (fine-tuning)
print("-----Starting fine-tuning...")
history_fine = model.fit(
    train_generator,
    epochs=20, 
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

print("----Training complete!-----")

# Plot training history
print("----Plotting training history----")
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


