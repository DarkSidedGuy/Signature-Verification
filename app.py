from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained signature verification model
model = load_model('signature_verification_model.h5')

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the file is part of the request
    if 'signature_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['signature_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to the "uploads" folder
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Preprocess the image for prediction
    image = preprocess_image(img_path)
    
    if image is not None:
        # Make a prediction using the trained model
        prediction = model.predict(np.expand_dims(image, axis=0))  # Model expects a batch of images
        result = "Valid Signature" if prediction[0][0] >= 0.5 else "Forged Signature"
        return jsonify({'result': result})  # Send back the result as JSON
    else:
        return jsonify({'error': 'Error in image processing'}), 400

def preprocess_image(image_path):
    """Preprocess image for signature verification."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))  # Resize the image to match model input
    img = img.astype('float32') / 255.0  # Normalize the pixel values
    return img

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
