# COVID-19 Detection from Chest CT Scans using CNN + Grad-CAM

This project is a web application that detects **COVID-19** from lung CT scans using a **Convolutional Neural Network (CNN)** and visualizes decision-making regions using **Grad-CAM**. The app is built with **Streamlit** for easy deployment and interactivity.

---

## Features

- Upload a CT scan image (`.jpg`, `.png`, `.jpeg`)
- CNN-based binary classification: **COVID** or **Non-COVID**
- Prediction confidence score
- Grad-CAM heatmap overlay to show important regions
- Built using:
  - TensorFlow/Keras
  - Streamlit
  - OpenCV
  - NumPy 

---

## Model Details

- Input size: `128x128`
- Architecture: Simple CNN with Conv2D, MaxPooling, Dropout
- Trained on labeled CT scan images (COVID / Non-COVID)
- Final layer: Sigmoid activation for binary classification

> The trained model is **not included** in this repo due to file size limits. Please download or train your own model and place it inside a `models/` folder.



