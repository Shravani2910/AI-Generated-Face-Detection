# AI-Generated-Face-Detection

# 📌 Overview

This project focuses on detecting AI-generated (synthetic) faces vs real human faces using deep learning. With the rise of deepfakes and generative AI, identifying fake facial images is crucial for digital forensics, media authenticity, and AI safety.

A CNN-based transfer learning model is trained on a high-quality, balanced dataset to accurately classify facial images as Real or AI-Generated.

# 🎯 Problem Statement

AI-generated faces are becoming increasingly realistic, making it difficult to distinguish them from real human faces. This project aims to build a robust image classification system that can automatically detect AI-generated facial images.

# 🗂 Dataset

Dataset Name: GRAVEX-200K / AI Face Detection Dataset

Total Images: 200,000

Classes:

Real Human Faces

AI-Generated (Fake) Faces

Image Size: 256 × 256

Split Ratio:

Train: 70%

Validation: 20%

Test: 10%

The dataset is balanced, curated from multiple sources, and preprocessed for deep learning tasks.

Due to GitHub size limits, the dataset is not included.
Download it from:
https://www.kaggle.com/datasets/muhammadbilal/gravax200k


# 🏗 Model Architecture

Base Model: EfficientNet (Transfer Learning)

Custom Layers:

Global Average Pooling

Fully Connected Dense Layer

Dropout for regularization

Sigmoid output layer

# ⚙️ Workflow

Dataset loading & preprocessing

Image augmentation

Transfer learning with EfficientNet

Model training & validation

Performance evaluation

Confusion Matrix & ROC Curve analysis

Deployment using Streamlit

# 📊 Evaluation Metrics

Accuracy

Confusion Matrix

ROC Curve & AUC Score

These metrics help assess model reliability and classification performance.

# 🚀 Deployment

The trained model is deployed using Streamlit, allowing users to upload an image and instantly get predictions indicating whether the face is Real or AI-Generated.

# 🛠 Tech Stack

Python

TensorFlow / Keras

EfficientNet

OpenCV

NumPy & Pandas

Matplotlib & Seaborn

Scikit-learn

Streamlit

Google Colab

# 📌 Use Cases

Deepfake detection

AI-generated content verification

Digital image forensics

Media authenticity analysis

📬 Developed by
Shravani Jagtap
AI/ML Engineer
