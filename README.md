---
title: Lung Disease Detection
emoji: 🫁
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Lung Disease Detection (DenseNet201)

A professional web application for detecting lung diseases from Chest X-ray images using a Transfer Learning approach with the **DenseNet201** architecture.

## 🚀 Features
- **Instant Analysis:** Predicts 7 categories of lung conditions (Atelectasis, Bacterial Pneumonia, COVID-19, Emphysema, Normal, Tuberculosis, Viral Pneumonia).
- **Grad-CAM Visualization:** Generates a heatmap showing exactly where the model is "looking" on the X-ray to make its decision.
- **Interactive UI:** A modern, dark-themed dashboard with radar and bar charts for confidence scoring.
- **Production Ready:** Powered by Flask, Gunicorn, and Docker for high-performance inference.

## 🛠️ Tech Stack
- **Framework:** Flask (Python)
- **Deep Learning:** TensorFlow / Keras (DenseNet201)
- **Image Processing:** OpenCV & Pillow
- **Frontend:** HTML5, Vanilla CSS, Chart.js
- **Deployment:** Docker & Hugging Face Spaces

## 📦 How to Run Locally
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd lung_disease
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
4. Open your browser at `http://localhost:7860`

## 🩺 Disclaimer
*This application is a prototype for educational and research purposes only. It is not intended for medical diagnosis or clinical use. Always consult a healthcare professional for medical advice.*
