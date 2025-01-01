

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import base64

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Load YOLO and VGG16 models
# yolo_model = YOLO('yolov5s.pt')  # Replace with the correct YOLO model path

# mobilenet_model = load_model('/workspaces/Cotton-Disease-Recognition-using-YOLO-Algorithm-/cotton_disease_mobilenet.h5')  # Replace with your VGG16 model path

# # Class names based on the dataset
# classes = ['Diseased Cotton Leaf', 'Diseased Cotton Plant', 'Fresh Cotton Leaf', 'Fresh Cotton Plant']

# # Confidence threshold for YOLO detection
# CONFIDENCE_THRESHOLD = 0.7


# def process_image(image):
#     """
#     Detect objects using YOLO and classify them using VGG16.
    
#     Args:
#     - image (numpy array): The input image.

#     Returns:
#     - detections (list): List of detections with labels, confidence, and bounding box.
#     """
#     # Run YOLO model on the image
#     results = yolo_model(image)

#     detections = []
#     for result in results:
#         if result.boxes is not None:
#             for detection in result.boxes.data:
#                 confidence = float(detection[4])  # Confidence score
#                 if confidence >= CONFIDENCE_THRESHOLD:
#                     # Extract bounding box coordinates
#                     x1, y1, x2, y2 = map(int, detection[:4])
#                     crop = image[y1:y2, x1:x2]

#                     # Resize and preprocess the crop for VGG16
#                     if crop.size > 0:  # Check if crop is valid
#                         crop_resized = cv2.resize(crop, (224, 224))
#                         crop_preprocessed = preprocess_input(np.expand_dims(crop_resized, axis=0))

#                         # Predict using VGG16
#                         mobilenet_prediction = mobilenet_model.predict(crop_preprocessed, verbose=0)
#                         class_index = np.argmax(mobilenet_prediction)
#                         confidence_score = np.max(mobilenet_prediction)
#                         detections.append({
#                             "label": classes[class_index],
#                             "confidence": float(confidence_score),
#                             "box": [x1, y1, x2, y2]
#                         })
#     return detections


# @app.route('/process', methods=['POST'])
# def process():
#     """
#     Flask API endpoint to process uploaded images for detection and classification.

#     Returns:
#     - JSON response with detections and annotated image.
#     """
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     # Read the uploaded image
#     file = request.files['file']
#     file_bytes = np.frombuffer(file.read(), np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     if image is None:
#         return jsonify({"error": "Invalid image format"}), 400

#     # Process the image for detections
#     detections = process_image(image)

#     # Draw bounding boxes and labels on the image
#     for detection in detections:
#         x1, y1, x2, y2 = detection['box']
#         label = f"{detection['label']} ({detection['confidence']:.2f})"

#         # Draw bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # Put label text
#         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Encode the image with bounding boxes to base64
#     _, buffer = cv2.imencode('.jpg', image)
#     encoded_image = base64.b64encode(buffer).decode('utf-8')

#     return jsonify({"detections": detections, "image": encoded_image})


# if __name__ == "__main__":
#     # Run Flask app
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64

# Initialize Flask app
app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})
# Load YOLO and MobileNet models
yolo_model = YOLO('yolov5s.pt')  # Replace with the correct YOLO model path

mobilenet_model = load_model('/workspaces/Cotton-Disease-Recognition-using-YOLO-Algorithm-/cotton_disease_mobilenet.h5')  # Replace with the correct path

# Class names based on the dataset
classes = ['Diseased Cotton Leaf', 'Diseased Cotton Plant', 'Fresh Cotton Leaf', 'Fresh Cotton Plant']

# Confidence threshold for YOLO detection
CONFIDENCE_THRESHOLD = 0.7


def process_image(image):
    """
    Detect objects using YOLO and classify them using MobileNet.
    
    Args:
    - image (numpy array): The input image.

    Returns:
    - detections (list): List of detections with labels, confidence, and bounding box.
    """
    # Run YOLO model on the image
    results = yolo_model(image)

    detections = []
    for result in results:
        if result.boxes is not None:
            for detection in result.boxes.data:
                confidence = float(detection[4])  # Confidence score
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, detection[:4])
                    crop = image[y1:y2, x1:x2]

                    # Resize and preprocess the crop for MobileNet
                    if crop.size > 0:  # Check if crop is valid
                        crop_resized = cv2.resize(crop, (224, 224))
                        crop_preprocessed = preprocess_input(np.expand_dims(crop_resized, axis=0))

                        # Predict using MobileNet
                        mobilenet_prediction = mobilenet_model.predict(crop_preprocessed, verbose=0)
                        class_index = np.argmax(mobilenet_prediction)
                        confidence_score = np.max(mobilenet_prediction)
                        detections.append({
                            "label": classes[class_index],
                            "confidence": float(confidence_score),
                            "box": [x1, y1, x2, y2]
                        })
    return detections


@app.route('/process', methods=['POST'])
def process():
    """
    Flask API endpoint to process uploaded images for detection and classification.

    Returns:
    - JSON response with detections and annotated image.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Read the uploaded image
    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Process the image for detections
    detections = process_image(image)

    # Draw bounding boxes and labels on the image
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['label']} ({detection['confidence']:.2f})"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label text
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode the image with bounding boxes to base64
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"detections": detections, "image": encoded_image})


if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)