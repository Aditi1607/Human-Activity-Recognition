# Human Activity Recognition using MediaPipe & LSTM

Human Activity Recognition (HAR) project that detects and classifies human actions such as **walking, running, jogging, boxing, hand waving, and clapping** in real time using a **webcam**, **MediaPipe Pose** for keypoint extraction, and an **LSTM** model for sequence classification.

> 📌 This repository contains the full pipeline: data collection → preprocessing → model training → real-time inference.

---

## ✨ Features

- Real-time webcam input for activity recognition  
- Pose estimation using **MediaPipe Pose** (33 body landmarks)  
- Sequence modeling with **LSTM** for temporal motion understanding  
- Supports multiple activities:
  - Walking  
  - Running  
  - Jogging  
  - Boxing  
  - Hand Waving  
  - Clapping  
- Modular code for:
  - Data collection & CSV generation  
  - Model training  
  - Inference & live demo  

---

## 🧠 Project Architecture

### 1. Data Collection

1. Capture frames from webcam  
2. Use **MediaPipe Pose** to detect skeleton  
3. Extract (x, y, visibility) for each of the 33 keypoints  
4. Save pose sequences to `.csv` along with the activity label  

### 2. Model Training

1. Load pose sequences from CSV  
2. Normalize / scale keypoints  
3. Create sliding windows of frames (e.g., 30–60 frames per sample)  
4. Train an **LSTM-based classifier** to predict activity from pose sequence  
5. Save trained model as `model.h5` or similar

### 3. Inference (Real-Time)

1. Read live frames from webcam  
2. Run MediaPipe Pose to get keypoints  
3. Build a sequence buffer of recent frames  
4. Pass sequence to LSTM model  
5. Display predicted activity label on the video stream

---

## 🧩 Tech Stack

- **Language:** Python  
- **Pose Estimation:** [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose)  
- **Deep Learning:** TensorFlow / Keras (LSTM) or PyTorch (if you prefer)  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** OpenCV, Matplotlib  
- **Environment:** Jupyter Notebook / Python scripts

---

## 📁 Repository Structure (suggested)

```bash
.
├── data/
│   ├── raw/                 # Raw recorded CSVs
│   ├── processed/           # Cleaned / merged datasets
├── models/
│   └── lstm_har_model.h5    # Saved trained model
├── notebooks/
│   └── exploration.ipynb    # EDA / experiments
├── src/
│   ├── collect_data.py      # Webcam capture & CSV generation
│   ├── train_model.py       # LSTM training script
│   ├── inference.py         # Real-time inference
│   └── utils.py             # Helper functions
├── assets/
│   ├── activities.png       # Activity collage image
│   ├── mediapipe_pose.png   # Skeleton diagram
│   └── pipeline.png         # Flowchart of the pipeline
├── README.md
└── requirements.txt
