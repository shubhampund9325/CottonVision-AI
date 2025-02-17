# from flask import Flask, request, render_template, jsonify, send_from_directory
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import uuid
# from ultralytics import YOLO

# app = Flask(__name__)

# # Create necessary directories
# UPLOAD_FOLDER = "uploads"
# PROCESSED_FOLDER = "processed"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # Update path to use the .pt model instead of .torchscript
# YOLO_MODEL_PATH = "/Users/shubhampund9767/Desktop/Yolo Project/runs/detect/train3/weights/best.pt"

# def initialize_model():
#     """Initialize YOLO model with proper error handling"""
#     try:
#         # Verify model file exists
#         if not os.path.exists(YOLO_MODEL_PATH):
#             raise FileNotFoundError(f"Custom YOLO model not found at {YOLO_MODEL_PATH}")
        
#         # Load custom YOLO model
#         model = YOLO(YOLO_MODEL_PATH)
#         # Set model parameters
#         model.conf = 0.25  # confidence threshold
#         model.iou = 0.45   # NMS IoU threshold
#         print("Custom YOLO model loaded successfully!")
#         return model
            
#     except Exception as e:
#         print(f"Error loading YOLO model: {str(e)}")
#         raise

# # Initialize model
# try:
#     yolo_model = initialize_model()
# except Exception as e:
#     print(f"Failed to initialize model: {str(e)}")
#     raise

# # Your custom class names - update these to match your training classes
# class_names = ['fresh cotton leaf', 'fresh cotton plant', 'diseased cotton leaf', 'diseased cotton plant']

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
    
#     image = request.files['image']
#     if not image.filename:
#         return jsonify({'error': 'No selected file'}), 400
        
#     filename = f"{uuid.uuid4()}.jpg"
#     image_path = os.path.join(UPLOAD_FOLDER, filename)
#     processed_filename = f"processed_{filename}"
#     processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
#     try:
#         # Save the uploaded image
#         image.save(image_path)
        
#         # Process the image
#         img = cv2.imread(image_path)
#         if img is None:
#             return jsonify({'error': 'Failed to read image'}), 400

#         # Run YOLO detection with your custom model
#         results = yolo_model.predict(
#             source=img,
#             save=False,
#             save_txt=False,
#             exist_ok=True,
#             verbose=False
#         )
        
#         # Process detections
#         detections = []
        
#         # Get the first result (assuming single image)
#         result = results[0]
        
#         if len(result.boxes) > 0:
#             for box in result.boxes:
#                 # Get box coordinates and class
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 conf = float(box.conf[0].item())
#                 class_id = int(box.cls[0].item())
                
#                 # Ensure class_id is within bounds
#                 if class_id < len(class_names):
#                     class_name = class_names[class_id]
#                 else:
#                     class_name = f"class_{class_id}"
                
#                 # Add detection if confidence is high enough
#                 if conf > 0.25:  # Lower confidence threshold
#                     detections.append({
#                         "class": class_name,
#                         "confidence": conf,
#                         "box": [x1, y1, x2, y2]
#                     })
                    
#                     # Draw on image
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(img, f"{class_name} {conf:.2f}", 
#                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Save the processed image
#         cv2.imwrite(processed_path, img)

#         # Return JSON response
#         return jsonify({
#             'status': 'success',
#             'detections': detections,
#             'processed_image': processed_filename
#         })

#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         return jsonify({'error': str(e)}), 500

#     finally:
#         # Clean up uploaded file
#         if os.path.exists(image_path):
#             os.remove(image_path)

# @app.route('/processed/<filename>')
# def processed_image(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from ultralytics import YOLO

app = Flask(__name__)

# Create necessary directories
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Update path to use the .pt model instead of .torchscript
YOLO_MODEL_PATH = "/Users/shubhampund9767/Desktop/Yolo Project/runs/detect/train3/weights/best.pt"

# Disease information dictionary
disease_info = {
    0: {"name": "Aphid", "description": "Small, soft-bodied insects that suck sap.", "precaution": "Use resistant varieties, introduce natural predators.", "medicine": "Neem oil, Imidacloprid."},
    1: {"name": "Aphids", "description": "Rapidly multiplying sap-sucking pests.", "precaution": "Field monitoring, remove infected plants.", "medicine": "Pyrethroid-based insecticides."},
    2: {"name": "Army Worm", "description": "Caterpillars that chew leaves.", "precaution": "Pheromone traps, biological control.", "medicine": "Bacillus thuringiensis, Spinosad."},
    3: {"name": "Bacterial Blight", "description": "Water-soaked lesions leading to wilting.", "precaution": "Use disease-free seeds, crop rotation.", "medicine": "Copper-based bactericides, Streptomycin."},
    4: {"name": "Cotton Boll Rot", "description": "Fungal infection causing decay.", "precaution": "Ensure aeration, remove infected bolls.", "medicine": "Carbendazim-based fungicides."},
    5: {"name": "Green Cotton Boll", "description": "Immature bolls remain green.", "precaution": "Proper irrigation, control bollworms.", "medicine": "Mancozeb fungicide, growth regulators."},
    6: {"name": "Healthy", "description": "No disease present.", "precaution": "Good soil fertility, proper rotation.", "medicine": "No treatment needed."},
    7: {"name": "Powdery Mildew", "description": "White powdery fungal spots on leaves.", "precaution": "Avoid humidity, remove infected leaves.", "medicine": "Sulfur-based fungicides, Tebuconazole."},
    8: {"name": "Target Spot", "description": "Circular brown spots leading to defoliation.", "precaution": "Improve drainage, avoid high density.", "medicine": "Chlorothalonil, copper-based fungicides."}
}

def initialize_model():
    """Initialize YOLO model with proper error handling"""
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)
        model.conf = 0.25
        model.iou = 0.45
        print("YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

# Initialize model
yolo_model = initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    if not image.filename:
        return jsonify({'error': 'No selected file'}), 400
    
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
    try:
        image.save(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        results = yolo_model.predict(source=img, save=False, save_txt=False, exist_ok=True, verbose=False)
        detections = []
        result = results[0]
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                
                if class_id in disease_info:
                    class_data = disease_info[class_id]
                else:
                    class_data = {"name": f"Unknown {class_id}", "description": "N/A", "precaution": "N/A", "medicine": "N/A"}
                
                if conf > 0.25:
                    detections.append({
                        "class": class_data["name"],
                        "confidence": conf,
                        "box": [x1, y1, x2, y2],
                        "description": class_data["description"],
                        "precaution": class_data["precaution"],
                        "medicine": class_data["medicine"]
                    })
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_data['name']} {conf:.2f}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(processed_path, img)
        return jsonify({'status': 'success', 'detections': detections, 'processed_image': processed_filename})
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
