# 🐱 Cat vs Dog Image Classification using MobileNetV2

A deep learning web application that classifies images as **Cat** or **Dog** using **Transfer Learning with MobileNetV2**. The model is deployed as an interactive multi-page **Streamlit Web Application** where users can upload images, view real-time predictions, check confidence scores, and inspect model metrics.

---

## 📌 Project Overview

This project fine-tunes a pre-trained MobileNetV2 architecture for binary image classification on the Kaggle Cat vs Dog dataset. It incorporates a custom confidence thresholding mechanism (< 70% confidence) to automatically flag low-confidence or out-of-distribution inputs with a **"Not a Cat or Dog"** warning.

---

## 🚀 Features

- **Real-Time Classification:** Upload `.jpg`, `.jpeg`, or `.png` images and obtain immediate prediction results.
- **Confidence Scoring:** View exact confidence percentages for every output prediction.
- **Out-of-Distribution Detection:** Automatically detects irrelevant or uncertain inputs using a confidence threshold (< 70%).
- **Multi-Page Dashboard:** Interactive Streamlit web interface featuring Home, Model Information, and Prediction pages.
- **Pre-processed Input Pipeline:** Automatic resizing (224×224 pixels) and color profile standardizations matching pre-trained MobileNetV2 standards.

---

## 📂 Dataset

- **Source:** [Kaggle - Cat VS Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)[cite: 3]
- **Total Images:** 24,961 images (12,491 Cat, 12,470 Dog)[cite: 3]
- **Target Dimensions:** 224 × 224 pixels (RGB)[cite: 3]

---

## 🧠 Model Architecture & Training Parameters

- **Base Architecture:** MobileNetV2 (Pre-trained on ImageNet)[cite: 3]
- **Framework:** TensorFlow / Keras[cite: 3]
- **Optimizer:** RMSprop[cite: 3]
- **Learning Rate:** 0.00001[cite: 3]
- **Dropout Rate:** 0.2[cite: 3]
- **Fine-Tuning:** Last 30 layers unfrozen for domain-specific feature extraction[cite: 3]

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.90%[cite: 3] |
| **Precision** | 98.62%[cite: 3] |
| **Recall** | 99.20%[cite: 3] |
| **F1-Score** | 98.91%[cite: 3] |

---

## 📁 Project Structure

```text
Cat_Dog_Classifier/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
├── PROJECT_REPORT.md                   # Comprehensive project analysis report
├── model/
│   └── fine_tuned_mobilenet_v2.keras   # Saved Keras model file
├── notebooks/
│   └── training.ipynb                  # Exploratory Data Analysis & training notebook
└── PetImages/                          # Local dataset directory
    ├── Cat/                            # Raw cat images
    └── Dog/                            # Raw dog images
