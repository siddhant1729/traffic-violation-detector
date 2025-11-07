# 🚦 Traffic Violation Detection System (ML + Computer Vision)

A hybrid **Machine Learning + Computer Vision** system that detects traffic violations from live or recorded video feeds.  
It identifies **vehicles**, detects **violations** (red-light jump, overspeeding, wrong-lane usage, etc.), logs them with **time, image, and license plate**, and generates daily reports.

---

## 🧠 Overview

**Goal:**  
To build a complete end-to-end intelligent system that can:
- Detect vehicles using **YOLOv8 + OpenCV**
- Track vehicle movements in real-time
- Identify violations using **ML classifiers (Random Forest, Decision Tree, Logistic Regression)**
- Log violation details with timestamps and cropped images
- Visualize data using **Streamlit Dashboard or FastAPI**

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| **Detection & Vision** | OpenCV, YOLOv8 (Ultralytics) |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **OCR (License Plate)** | EasyOCR, Tesseract (optional) |
| **Dashboard & Reporting** | Streamlit / FastAPI, Matplotlib |
| **Database / Storage** | SQLite3, CSV |
| **Version Control** | Git & GitHub |

---

## 🧩 Folder Structure

traffic-violation-detector/
├── data/ # Input videos and datasets
├── logs/ # Saved violations, cropped images
├── models/ # YOLO weights / trained ML models
├── src/
│ ├── ml_models/ # ML models (.pkl)
│ ├── database.py # Handles SQLite logging
│ ├── detection.py # YOLOv8 detection logic
│ ├── tracking.py # Object tracking logic
│ ├── features.py # Feature extraction for ML
│ └── main.py # Main control script
├── requirements.txt # All dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/siddhant1729/traffic-violation-detector.git
cd traffic-violation-detector
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
.\venv\Scripts\activate       # On Windows
source venv/bin/activate      # On Linux/Mac
3️⃣ Install Requirements
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the System
bash
Copy code
python src/main.py
🎯 Features
✅ Vehicle Detection using YOLOv8
✅ Object Tracking using OpenCV (CSRT / DeepSORT)
✅ Violation Detection

Red Light Jump

Wrong Lane

Overspeeding
✅ ML Integration for intelligent classification
✅ License Plate Recognition (OCR)
✅ SQLite Logging + Image Capture
✅ Streamlit Dashboard / FastAPI API
✅ Daily Report Generation

📊 Example Output (Coming Soon)
Annotated video showing detected violations

Daily CSV report

Streamlit dashboard screenshots

👨‍💻 Author
Siddhant
CSE Undergrad | Competitive Programmer | AI/ML Enthusiast
📍 JIIT Noida
🔗 LinkedIn
🔗 GitHub

🏁 License
This project is open-source and available under the MIT License