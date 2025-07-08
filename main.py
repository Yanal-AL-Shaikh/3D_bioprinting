import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import torch
import winsound
#"C:\Users\ADMIN\Desktop\final_Update333333333333\segment_train_results\weights\best.pt" 21/5
#"C:\Users\ADMIN\Desktop\final_Update333333333333\segment_train_results\weights\best.pt" 7/6
# Load the trained model
model_path =r"C:\Users\ADMIN\Desktop\مشروع\final_Update333333333333\segment_train_results\weights\best.pt"
model = YOLO(model_path)

# Class names from your trained model
class_names = ['under', 'ok']

class BioprintingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Bioprinting Monitoring System")
        self.setGeometry(100, 100, 900, 700)

        # Main layout
        main_layout = QVBoxLayout()

        # Header with title and logo
        header_layout = QHBoxLayout()
        self.title_label = QLabel("3D Bioprinting Monitoring System")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #1E90FF;")
        header_layout.addWidget(self.title_label)

        self.logo_label = QLabel()
        self.logo_label.setFixedSize(50, 50)
        self.logo_label.setStyleSheet("background-color: #1E90FF; border-radius: 25px;")
        header_layout.addWidget(self.logo_label, alignment=Qt.AlignRight)
        main_layout.addLayout(header_layout)

        # Video display
        self.video_label = QLabel("No video loaded")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.default_border_style = "border: 2px solid #1E90FF; border-radius: 5px;"
        self.video_label.setStyleSheet(self.default_border_style)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Status label
        self.status_label = QLabel("Status: Waiting for video...")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setStyleSheet("color: #32CD32;")
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        # Detection result label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("color: #FF4500;")
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        # Statistics label
        self.stats_label = QLabel("Under Frames: 0 | OK Frames: 0")
        self.stats_label.setFont(QFont("Arial", 12))
        self.stats_label.setStyleSheet("color: #FFFFFF; background-color: #333333; padding: 5px; border-radius: 3px;")
        main_layout.addWidget(self.stats_label, alignment=Qt.AlignCenter)

        # Stop duration input
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Stop Duration (seconds):")
        duration_label.setFont(QFont("Arial", 12))
        duration_label.setStyleSheet("color: #1E90FF;")
        duration_layout.addWidget(duration_label)

        self.duration_input = QLineEdit("5")
        self.duration_input.setFont(QFont("Arial", 12))
        self.duration_input.setFixedWidth(50)
        self.duration_input.setStyleSheet("padding: 5px; border: 1px solid #1E90FF; border-radius: 3px;")
        duration_layout.addWidget(self.duration_input)
        duration_layout.addStretch()
        main_layout.addLayout(duration_layout)

        # Load video button
        self.load_button = QPushButton("Load Video")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #1E90FF;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4682B4;
            }
        """)
        self.load_button.clicked.connect(self.load_video)
        main_layout.addWidget(self.load_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Timer for border flashing
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_border_color)
        self.is_red_border = False

        self.cap = None

        # Variables for stopping logic and statistics
        self.under_frame_count = 0
        self.under_threshold = 150  # Will be updated based on user input
        self.is_stopped = False
        self.under_frames = 0  # Counter for 'under' frames
        self.ok_frames = 0    # Counter for 'ok' frames

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.status_label.setText("Status: Printing in Progress...")
                self.result_label.setText("")
                self.stats_label.setText("Under Frames: 0 | OK Frames: 0")
                self.is_stopped = False
                self.under_frame_count = 0
                self.under_frames = 0
                self.ok_frames = 0
                # Stop flashing and reset border style
                self.flash_timer.stop()
                self.video_label.setStyleSheet(self.default_border_style)
                # Update threshold based on user input (assuming 30 FPS)
                try:
                    stop_duration = float(self.duration_input.text())
                    self.under_threshold = int(stop_duration * 30)  # 30 FPS * seconds
                except ValueError:
                    self.under_threshold = 150  # Default to 5 seconds if invalid input
                self.timer.start(30)
            else:
                self.status_label.setText("Status: Failed to open video")

    def toggle_border_color(self):
        if self.is_red_border:
            self.video_label.setStyleSheet(self.default_border_style)
        else:
            self.video_label.setStyleSheet("border: 2px solid #FF0000; border-radius: 5px;")
        self.is_red_border = not self.is_red_border

    def update_frame(self):
        if self.cap and not self.is_stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Status: Video finished")
                self.cap.release()
                self.timer.stop()
                self.flash_timer.stop()
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                imgsz=640,
                conf=0.5
            )

            detected_classes = []
            annotated_frame = frame_rgb.copy()

            for result in results:
                boxes = result.boxes
                masks = result.masks

                for box in boxes:
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    detected_classes.append(class_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{class_names[class_id]} ({conf:.2f})"
                    color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )

                if masks is not None:
                    for mask in masks:
                        mask_data = mask.data[0].cpu().numpy()
                        mask_data = cv2.resize(mask_data, (frame_rgb.shape[1], frame_rgb.shape[0]))
                        mask_data = mask_data > 0.5
                        annotated_frame[mask_data] = annotated_frame[mask_data] * 0.5 + np.array([0, 255, 255]) * 0.5

            # Update statistics and stopping logic
            if 1 in detected_classes:  # ok
                self.result_label.setText("Detected class: ok")
                self.ok_frames += 1
                self.under_frame_count = 0
            elif 0 in detected_classes:  # under
                self.result_label.setText("Detected class: under")
                self.under_frames += 1
                self.under_frame_count += 1
                if self.under_frame_count >= self.under_threshold:
                    self.is_stopped = True
                    self.status_label.setText("Status: Printing Stopped")
                    self.result_label.setText("The printing process has been stopped")
                    winsound.Beep(1000, 1000)
                    self.timer.stop()
                    # Start flashing the border
                    self.flash_timer.start(500)  # Flash every 500ms
            else:
                self.result_label.setText("No under or ok detected")
                self.under_frame_count = 0

            # Update statistics label
            self.stats_label.setText(f"Under Frames: {self.under_frames} | OK Frames: {self.ok_frames}")

            # Convert frame to QImage for display
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(
                annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qt_img).scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
            )
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.flash_timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BioprintingApp()
    window.show()
    sys.exit(app.exec_())
