from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
import os

app = Flask(__name__)

# Paths for the disease model and the grape leaf classifier model
model_path = os.path.join(os.getcwd(), 'models', 'grape_leaf_model.h5')
classifier_path = os.path.join(os.getcwd(), 'models', 'grape_leaf_classifier.h5')

# Debugging: Print model paths and current directory contents
print(f"Model Path: {model_path}")
print(f"Classifier Path: {classifier_path}")
print("Files in current directory:", os.listdir(os.getcwd()))
print("Files in models directory:", os.listdir(os.path.join(os.getcwd(), 'models')))

# Load both models
try:
    if os.path.exists(model_path) and os.path.exists(classifier_path):
        model = load_model(model_path)
        grape_leaf_classifier = load_model(classifier_path)
        print("Models loaded successfully!")
    else:
        raise FileNotFoundError("Model files not found!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    grape_leaf_classifier = None

# Disease info dictionary
disease_info = {
    0: {
        "name": "Black Rot",
        "description": "Black Rot is a fungal disease affecting grape leaves, causing black spots that expand and damage the foliage.",
        "fertilizer": "Use a fungicide with Mancozeb or Copper sulfate to control Black Rot."
    },
    1: {
        "name": "Esca (Black Measles)",
        "description": "Esca is a fungal disease that causes discoloration and black spots on grape leaves, also known as black measles.",
        "fertilizer": "Use a balanced fertilizer with Potassium to help prevent spread and boost plant resilience."
    },
    2: {
        "name": "Healthy",
        "description": "The leaf is healthy with no visible signs of disease. Keep monitoring to ensure continued health.",
        "fertilizer": "No fertilizer required. Maintain regular care and monitoring."
    },
    3: {
        "name": "Leaf Blight",
        "description": "Leaf Blight is a common disease caused by fungi, leading to brown spots and wilting of grape leaves.",
        "fertilizer": "Use a fungicide with Chlorothalonil and ensure proper soil drainage to prevent blight."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if models are loaded
    if model is None or grape_leaf_classifier is None:
        error_message = "Model not loaded. Please check the model paths and compatibility."
        print(f"Error: {error_message}")  # Log the error for debugging purposes
        return render_template('error.html', message=error_message)
    
    if 'file' not in request.files:
        return render_template('error.html', message="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="No selected file")
    
    try:
        # Read and process the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return render_template('error.html', message="Error reading the image.")
        
        img_resized = cv2.resize(img, (100, 100)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        # Step 1: Check if the image is a grape leaf using the classifier
        is_grape_leaf = grape_leaf_classifier.predict(img_input)[0][0]  # Assuming binary output (1 = grape leaf)
        if is_grape_leaf < 0.5:
            return render_template('error.html', message="Please upload a valid grape leaf image.")

        # Step 2: Make a prediction on the grape leaf disease
        prediction = model.predict(img_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Retrieve disease information
        disease = disease_info.get(predicted_class, {
            "name": "Unknown Disease",
            "description": "No description available.",
            "fertilizer": "No fertilizer suggestion available."
        })

        # Render the results
        return render_template(
            'predict.html',
            disease_name=disease["name"],
            disease_description=disease["description"],
            disease_fertilizer=disease["fertilizer"]
        )
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('error.html', message="Error processing the image or model prediction.")

if __name__ == "__main__":
    app.run(debug=True)