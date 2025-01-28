from flask import Flask, request, jsonify, render_template
#from tensorflow.keras.models import load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the model
#MODEL_PATH = "/Users/shubhampund9767/Cotton-Disease-Recognition-using-YOLO-Algorithm--1/cotton_disease_mobilenet.h5"
MODEL_PATH="/Users/shubhampund9767/Cotton-Disease-Recognition-using-YOLO-Algorithm--1/latestmodel.h5"
model = load_model(MODEL_PATH)

# Class names
class_names = ['fresh cotton leaf', 'fresh cotton plant', 'diseased cotton leaf', 'diseased cotton plant']

# Prediction function
def predict_image(image_path, class_names):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    if predicted_index >= len(class_names):
        raise ValueError(f"Predicted index {predicted_index} exceeds number of class labels.")

    predicted_class = class_names[predicted_index]
    return predicted_class, confidence

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Make prediction
    try:
        predicted_class, confidence = predict_image(file_path, class_names)
        return jsonify({'class': predicted_class, 'confidence':  float(round(confidence * 100, 2))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
