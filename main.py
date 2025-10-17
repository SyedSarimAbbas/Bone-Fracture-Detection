import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import json
import base64
# -----------------------------
# Load your trained YOLO model
# -----------------------------
model = YOLO("bone_fracture_detection.pt")

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")



def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("rm373batch15-bg-11.jpg")

# Title
st.title("🦴 Bone Fracture Detection App")
st.write("Upload an X-ray image to check for possible bone fractures.")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Convert image for model
    img_array = np.array(image)

    # Button to predict
    if st.button("🔍 Detect Fracture"):
        with st.spinner("Analyzing image..."):
            
            # Run YOLO inference
            results = model.predict(img_array)

            predictions = []
            for box in results[0].boxes:
                cls_id = int(box.cls.item())                 # class ID
                class_name = model.names[cls_id]             # class label
                conf = float(box.conf.item())                # confidence
                xyxy = box.xyxy[0].tolist()                  # bounding box [x1,y1,x2,y2]

                predictions.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": xyxy
                })

            # Save predictions in your required JSON format
            output_data = {
                "Fracture Detected": predictions
            }

            json_filename = "detection_results.json"
            with open(json_filename, "w") as f:
                json.dump(output_data, f, indent=4)

            # Show YOLO detection result image (with bounding boxes)
            result_img = results[0].plot()  # numpy array (BGR)
            st.image(result_img, caption="Detection Result", use_container_width=True)
