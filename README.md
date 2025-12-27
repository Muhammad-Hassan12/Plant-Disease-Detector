# 🌿 Plant Disease Detection AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.75%25-green)

## 📖 Overview
This project is an end-to-end Deep Learning application capable of identifying **38 different plant diseases** from leaf images. Built using **TensorFlow/Keras** and deployed with **Streamlit**, it uses Transfer Learning with the **MobileNetV2** architecture to achieve high accuracy (~99%) while remaining lightweight and fast.

This tool is designed to assist farmers and agricultural experts in early disease detection to prevent crop loss.

## ✨ Features
* **High Accuracy:** Achieved **98.75%** test accuracy on the PlantVillage dataset.
* **38 Class Classification:** Detects diseases across 14 distinct crop species (Apple, Tomato, Corn, etc.).
* **Robust Model:** Uses Fine-Tuned MobileNetV2 with baked-in data augmentation.
* **User-Friendly Interface:** Simple web app built with Streamlit for easy image uploading and analysis.
* **Real-time Inference:** Fast predictions using optimized model architecture.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API)
* **Model Architecture:** MobileNetV2 (Transfer Learning & Fine Tuning)
* **Web Framework:** Streamlit
* **Image Processing:** Pillow (PIL), NumPy
* **Data Visualization:** Matplotlib (for training history)

## 📊 Dataset
The model was trained on the **New Plant Diseases Dataset** "Link:https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset", which consists of approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 classes.

## 🧠 Model Architecture & Training
The model uses a **Transfer Learning** approach:
1.  **Base Model:** MobileNetV2 (Pre-trained on ImageNet), chosen for its efficiency.
2.  **Preprocessing:** Inputs are scaled to `[-1, 1]` using MobileNet's standard preprocessing.
3.  **Data Augmentation:** Random Flip, Rotation, and Zoom layers are integrated directly into the model pipeline.
4.  **Training Strategy:**
    * **Phase 1:** Feature Extraction (Base model frozen, Top layers trained).
    * **Phase 2:** Fine-Tuning (Top 55 layers of base model unfrozen, trained with a very low learning rate `1e-4`).

### Performance
* **Training Accuracy:** ~99%
* **Validation Accuracy:** ~98%
* **Test Accuracy:** **99.17%**

*(You can add your training graphs here by uploading `image_49d73c.png` to your repo and linking it)*
![Training Graphs](path/to/your/graph_image.png)



## 📂 Directory Structure
