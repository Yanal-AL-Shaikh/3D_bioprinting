# 3D Bioprinting Monitoring System

##  Introduction
An intelligent monitoring system developed using AI and deep learning techniques to oversee 3D bioprinting processes that print organs, tissues, and living cells for transplantation.  
The goal is to **detect printing errors early**, such as **under-extrusion**, and automatically stop the printer to avoid material waste.

---

##  Features
- Real-time video analysis of the bioprinting process.
- Utilizes a trained **YOLOv8-seg** model to classify print quality as `"under"` or `"ok"`.
- User-friendly GUI built with **PyQt5**.
- Adjustable auto-stop duration based on faulty frame count.
- Visual border flashing and **audio alerts** when errors are detected.

---

##  Technologies Used
- Python 3.x  
- Ultralytics **YOLOv8**  
- **PyQt5** for GUI  
- **OpenCV** for video handling  
- **Git** & **GitHub** for version control  

---

##  Data Preparation
The dataset was annotated and segmented using **Roboflow**, enabling precise labeling of bioprinting defects like under-extrusion.  
**Data augmentation** was applied to improve generalization (rotation, brightness changes, noise, etc.).

---

##  How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. Run the main application:
python main.py

4. Load a bioprinting video using the "Load Video" button.

5. Watch results in real-time and adjust stop settings as neede

 Notes
Ensure you are using a machine with CUDA support for faster YOLO predictions.

This system is designed for demonstration and research purposes only.

  Developed by
Yanal Al Shaikh Ali
Tafila Technical University
Field: Artificial Intelligence and Data Science


   
