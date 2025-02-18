import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import tempfile

# Title
st.title("Cotton Disease Detection System")

# Tabs for Camera & Upload
tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload Image"])

# API URL (Replace with your backend API)
API_URL = "http://127.0.0.1:5000/predict"

# Disease Information Dictionary
disease_info = {
    "Aphid": {"description": "Small, soft-bodied insects that suck sap.", 
              "precaution": "Use resistant varieties, introduce natural predators.", 
              "medicine": "Neem oil, Imidacloprid."},

    "Bacterial Blight": {"description": "Water-soaked lesions leading to wilting.", 
                         "precaution": "Use disease-free seeds, crop rotation.", 
                         "medicine": "Copper-based bactericides, Streptomycin."}
}

# 📷 **Camera Capture**
with tab1:
    st.header("Live Camera Feed")
    cap = cv2.VideoCapture(0)  # Open camera
    
    capture = st.button("📸 Capture & Detect")

    if capture:
        ret, frame = cap.read()
        cap.release()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(img, caption="Captured Image", use_column_width=True)
            
            # Save image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            Image.fromarray(img).save(temp_file.name)

            # Send image to API
            with open(temp_file.name, "rb") as f:
                response = requests.post(API_URL, files={"file": f})

            if response.status_code == 200:
                result = response.json()
                st.success(f"Disease Detected: **{result['disease']}**")
                st.write(f"Description: {disease_info[result['disease']]['description']}")
                st.write(f"Precaution: {disease_info[result['disease']]['precaution']}")
                st.write(f"Medicine: {disease_info[result['disease']]['medicine']}")
            else:
                st.error("Error in detection. Try again.")

# 📁 **Upload Image for Detection**
with tab2:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a cotton leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Send image to API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            with open(temp_file.name, "rb") as f:
                response = requests.post(API_URL, files={"file": f})

        if response.status_code == 200:
            result = response.json()
            st.success(f"Disease Detected: **{result['disease']}**")
            st.write(f"Description: {disease_info[result['disease']]['description']}")
            st.write(f"Precaution: {disease_info[result['disease']]['precaution']}")
            st.write(f"Medicine: {disease_info[result['disease']]['medicine']}")
        else:
            st.error("Error in detection. Try again.")