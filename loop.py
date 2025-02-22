from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import uuid
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv  # Import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file


# Create necessary directories
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Update the model path to be relative
YOLO_MODEL_PATH = "/Users/shubhampund9767/Documents/Cotton-Disease-Recognition-using-YOLO-Algorithm--1/best1.pt"  # Change from absolute path

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

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyBySXoWXAI5dnYLNhIEJZyNTWc54It7B1g"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.  Please set your Gemini API key.")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

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

def generate_gemini_response(prompt):
    """Generates a response using the Gemini API."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Sorry, I couldn't generate a response at this time."

@app.route('/')
def home():
    return render_template('main.html')

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
                conf = float(box.conf[0].item())  # Original confidence score (0-1)
                class_id = int(box.cls[0].item())
                
                if conf > 0.25:  # Confidence threshold
                    if class_id in disease_info:
                        class_data = disease_info[class_id]
                        detection_text = class_data['name']
                        confidence_score = min(conf * 100, 100)  # Ensure confidence doesn't exceed 100%
                        
                        detections.append({
                            "class": class_data["name"],  # Use the actual disease name
                            "confidence": confidence_score,  # Properly scaled confidence
                            "box": [x1, y1, x2, y2],
                            "description": class_data["description"],
                            "precaution": class_data["precaution"],
                            "medicine": class_data["medicine"]
                        })
                        
                        # Update image annotation
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{detection_text} ({confidence_score:.2f}%)", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 0), 2)
        
        if not detections:
            detections.append({
                "class": "Healthy",
                "confidence": 100.0,
                "description": disease_info[6]["description"],
                "precaution": disease_info[6]["precaution"],
                "medicine": disease_info[6]["medicine"]
            })

        cv2.imwrite(processed_path, img)

        # Prepare information for Gemini prompt
        disease_names = [d['class'] for d in detections]
        disease_list_str = ", ".join(disease_names)

        # Create a prompt for Gemini based on the detections
        if disease_names:
            prompt = f"""
            I have detected the following diseases in a cotton field: {disease_list_str}. 
            Generate an HTML document that includes a well-structured table summarizing each disease. The table should have the following columns:
            
            - **Disease Name**  
            - **Symptoms**  
            - **Precautions**  
            - **Recommended Medicines**  

            The table should be properly styled using Bootstrap CDN for better readability. Ensure the HTML document includes:
            
            - A proper `<DOCTYPE html>` declaration.  
            - `<head>` section with Bootstrap CSS linked.  
            - `<body>` section with a heading and the table.  
            - Some additional styling for improved UI, such as padding, margins, and font styling.

            Return the complete HTML document with these details.
            """
        else:
            prompt = """
            Generate a simple HTML document that displays a message saying 'No diseases detected!' 
            Style the message with basic CSS for readability (e.g., centered text, font styling, padding). 
            Use Bootstrap CDN if possible.
            """
        # Get the response from Gemini
        gemini_response = generate_gemini_response(prompt)

        return jsonify({
            'status': 'success',
            'detections': detections,
            'processed_image': processed_filename,
            'gemini_response': gemini_response  # Include the Gemini response in the JSON
        })
    
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
    app.run()
