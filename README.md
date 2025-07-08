# 3D Bioprinting Monitoring System

## Introduction  
An intelligent monitoring system developed using AI and deep learning techniques to oversee 3D bioprinting processes that print organs, tissues, and living cells for transplantation.  
The goal is to detect printing errors early, such as **under-extrusion**, and automatically stop the printer to avoid material waste.

## Features  
- Real-time video analysis of the bioprinting process.  
- Utilizes a trained YOLOv8-seg model to classify print quality as "under" or "ok".  
- User-friendly GUI built with PyQt5.  
- Adjustable automatic stop duration based on consecutive faulty frames.  
- Visual border flashing and audio alerts when printing stops.

## Technologies Used  
- Python 3.x  
- Ultralytics YOLOv8 library  
- PyQt5 for GUI  
- OpenCV for video processing  
- Git & GitHub for version control
- 
## Data Preparation  
The dataset was annotated and segmented using Roboflow, which facilitated accurate labeling of bioprinting defects. Data augmentation techniques were applied to increase dataset variability and improve model robustness.

## How to Run  
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   
2.Run the main application:   
python main.py


3.Load a bioprinting video using the "Load Video" button.

4.View results on the interface and adjust stop settings as needed.

Project Files
main.py: Main GUI application.

train_yolo.py: YOLOv8 training script (optional).

best.pt: Trained YOLO model weights (included or downloadable separately).

README.md: This file.

requirements.txt: List of required Python packages.




