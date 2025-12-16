"""Test and evaluate a trained emotion recognition model.

This script loads a trained model, runs predictions on a test
dataset stored in folders (one subfolder per class), prints
accuracy and a classification report, and displays a confusion
matrix heatmap.
"""

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Keras utilities for loading the model and preparing test images
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Scikit-learn metrics
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------------
# Configuration
# -------------------------------
# Path to test images (folder containing one subfolder per class)
TEST_DATA_DIR = 'Dataset/test'
# Path to the trained model file
MODEL_PATH = 'model/Best_Emotion_MobileNetV2.keras'    

# Image dimensions expected by the model
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Batch size for prediction/evaluation
BATCH_SIZE = 32


# Create a test data generator (no augmentation, only preprocessing)
print("Loading test data...")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False     # keep ordering to match labels
)

class_labels = list(test_generator.class_indices.keys()) 
print(f"Found emotion classes: {class_labels}")

# Load the trained model from disk
print(f"Loading model from: {MODEL_PATH} ...")
model_test = load_model(MODEL_PATH)


# Run predictions on the entire test set
print("Performing predictions on the test set...")
predictions = model_test.predict(test_generator, verbose=1)




# Convert predicted probabilities to class indices
y_pred = np.argmax(predictions, axis=1)

# Ground truth labels as inferred from folder order
y_true = test_generator.classes

# Evaluate model on the test set and print accuracy
test_loss, test_acc = model_test.evaluate(test_generator, verbose=0)
print("\n" + "="*40)
print(f"ACCURACY: {test_acc * 100:.2f}%")
print("="*40 + "\n")

print("Detailed classification report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()