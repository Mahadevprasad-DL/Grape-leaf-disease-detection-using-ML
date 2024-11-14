import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir):
    images = []
    labels = []
    
    # Load images and labels from the dataset
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Check if it's a directory
            for image_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, image_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))  # Resize to fit model input
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model():
    data_dir = 'dataset/'  # Path to your dataset folder
    images, labels = load_data(data_dir)
    
    # Normalize images
    images = images / 255.0  
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    # Define the model architecture for disease classification
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(set(labels)), activation='softmax')  # Change to number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model for disease classification
    model.save('models/grape_leaf_model.h5')

    # Define and train the classifier model (if applicable)
    classifier_model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(1, activation='sigmoid')  # Binary classification (grape leaf or not)
    ])

    classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Assuming you have a separate dataset for the classifier
    classifier_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained classifier model
    classifier_model.save('models/grape_leaf_classifier.h5')
    
    # Save the label encoder for later use
    np.save('label_encoder.npy', label_encoder.classes_)

if __name__ == "__main__":
    train_model()