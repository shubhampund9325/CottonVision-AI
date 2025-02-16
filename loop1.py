from flask import Flask, request, jsonify, render_template, send_from_directory
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
import uuid

app = Flask(__name__)

# Ensure upload and processed directories exist
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "latestmodel.h5"
model = load_model(MODEL_PATH)

# Class names (modify based on your dataset)
class_names = ['fresh cotton leaf', 'fresh cotton plant', 'diseased cotton leaf', 'diseased cotton plant']

# Function to predict image class
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    if predicted_index >= len(class_names):
        raise ValueError(f"Predicted index {predicted_index} exceeds number of class labels.")

    return class_names[predicted_index], confidence

# Function to generate a bounding box (Dummy logic)
def generate_bounding_box(image):
    height, width, _ = image.shape
    x1, y1 = np.random.randint(0, width // 4), np.random.randint(0, height // 4)
    x2, y2 = np.random.randint(width // 2, width - 1), np.random.randint(height // 2, height - 1)
    return (x1, y1, x2, y2)

# Function to draw bounding box on the image
def draw_bounding_box(image_path):
    image = cv2.imread(image_path)
    bbox = generate_bounding_box(image)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box

    processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(processed_path, image)
    return processed_path

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Make prediction
    try:
        predicted_class, confidence = predict_image(file_path)

        # If the image is diseased, apply bounding box
        if "diseased" in predicted_class.lower():
            processed_path = draw_bounding_box(file_path)
        else:
            processed_path = file_path  # No bounding box for healthy images

        return render_template('result.html',
                               predicted_class=predicted_class,
                               confidence=round(confidence * 100, 2),
                               image_url=f'/processed/{os.path.basename(processed_path)}')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve processed images
@app.route('/processed/<filename>')
def serve_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
