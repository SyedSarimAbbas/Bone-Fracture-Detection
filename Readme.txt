# Project Overview

This project aims to develop an AI-powered system that detects bone fractures from medical images (such as X-rays) using deep learning and computer vision techniques.
The system leverages YOLOv8 for object detection and classification, and provides a user-friendly interface built with Flask/Streamlit for real-time predictions.

The goal of this project is to assist medical professionals by:

Speeding up fracture detection.

Reducing diagnostic errors.

Providing a scalable, cost-effective solution for healthcare.

 # Features

 Upload X-ray images for real-time bone fracture detection.

 Deep learning-based YOLOv8 model for accurate fracture identification.

 Web application using Flask + HTML/CSS (or Streamlit UI).

 Visualize predictions with bounding boxes on detected fractures.

 Download processed images with predictions.

 Easily extensible to other medical image detection tasks.

 Project Structure
 Bone-Fracture-Detection/
 │── app.py                 # Flask/Streamlit app (main entry point)
 │── requirements.txt        # Required Python libraries
 │── static/                 # CSS, JS, and images
 │── templates/              # HTML templates (index.html, predict.html, result.html)
 │── models/                 # Trained YOLOv8 weights
 │── dataset/                # X-ray dataset (train/val/test)
 │── utils/                  # Helper functions (image processing, prediction, etc.)
 │── README.md               # Project documentation

# Model Training

Dataset

Dataset contains X-ray images of bones (fractured and non-fractured).

Annotated using LabelImg in YOLO format (.txt files).

Model

Trained using YOLOv8.

Classes: Fracture and Normal.

T# raining Command

yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640


# Evaluation

yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

 # Installation
 Clone the repository
git clone https://github.com/yourusername/bone-fracture-detection.git
cd bone-fracture-detection

 Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Download trained weights

Place your trained YOLOv8 weights (e.g., best.pt) inside the models/ folder.

 Usage
Run Flask App
python app.py


Visit:http://localhost:8501/

Run Streamlit App
streamlit run app.py

# Results

Accuracy: ~92% on test dataset

Precision/Recall: Balanced across fracture detection

Inference Speed: <80ms per image on GPU

