import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess images from dataset directories
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (224, 224))  # Resize image to fit the model
    img = img.astype('float32') / 255.0  # Normalize the pixel values
    return img

# Dataset directories
valid_dir = r'C:\Users\ayush\OneDrive\Desktop\Signature_verification\datasets\valid_signatures'
forged_dir = r'C:\Users\ayush\OneDrive\Desktop\Signature_verification\datasets\forged_signatures'

# Load images and labels
image_paths = []
labels = []

# Load genuine signature images
for filename in os.listdir(valid_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(os.path.join(valid_dir, filename))
        labels.append(1)  # Label 1 for valid signatures

# Load forged signature images
for filename in os.listdir(forged_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(os.path.join(forged_dir, filename))
        labels.append(0)  # Label 0 for forged signatures

# Prepare data (preprocess images and split into training/test sets)
images = np.array([preprocess_image(img_path) for img_path in image_paths])
images = images[..., np.newaxis]  # Add a channel dimension (required for CNN)

# Convert labels to numpy array
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stopping])

# Save the trained model
model.save('signature_verification_model.h5')
