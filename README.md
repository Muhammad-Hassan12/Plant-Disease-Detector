# 🌿 Plant Disease Detection AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.75%25-green)

## 📖 Overview
This project is an end-to-end Deep Learning application capable of identifying **38 different plant diseases** from leaf images. Built using **TensorFlow/Keras** and deployed with **Streamlit**, the final model utilizes Transfer Learning with **MobileNetV2** to achieve high accuracy (~99%) while remaining lightweight and fast.

This tool is designed to assist farmers and agricultural experts in early disease detection to prevent crop loss.

## ✨ Features
* **High Accuracy:** Achieved **98.75%** test accuracy on the PlantVillage dataset.
* **38 Class Classification:** Detects diseases across 14 distinct crop species (Apple, Tomato, Corn, etc.).
* **Robust Model:** Built after rigorous experimentation with 4 different model strategies.
* **User-Friendly Interface:** Simple web app built with Streamlit for easy image uploading and analysis.
* **Real-time Inference:** Fast predictions using optimized model architecture.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API)
* **Model Architecture:** MobileNetV2 (Transfer Learning & Fine Tuning)
* **Web Framework:** Streamlit
* **Image Processing:** Pillow (PIL), NumPy
* **Data Visualization:** Matplotlib (for training history)

## 📊 Dataset
The model was trained on the **PlantVillage Dataset** (Augmented), which consists of approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 classes.

## 🧠 Model Training & Experimentation
To ensure the best performance, **4 different experimental strategies** were conducted. The final model was selected based on validation accuracy, loss stability, and ability to generalize on unseen data.

### 🔬 The 4 Experiments:

| Exp | Strategy | Description | Outcome |
| :-- | :--- | :--- | :--- |
| **1** | **Simple CNN** | A basic Custom CNN trained from scratch. | Moderate accuracy, struggled with complex features. |
| **2** | **CNN + Augmentation** | Same CNN but with Data Augmentation (Flip, Rotation, Zoom). | Better generalization, but training was slow. |
| **3** | **Transfer Learning (Feature Extraction)** | **MobileNetV2** (Frozen base) + Custom Head. | High accuracy, very fast convergence. |
| **4** | **Transfer Learning (Fine-Tuning)** | **MobileNetV2** (Unfrozen top layers) + Fine-tuning with low learning rate (`1e-4`). | **🏆 Best Performance (98.75% Accuracy)** |

### 🏆 Final Model: MobileNetV2 Fine-Tuned
The final deployed model uses **Strategy 4**.
1.  **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
2.  **Preprocessing:** Baked directly into the model layers (rescaling pixels to `[-1, 1]`).
3.  **Data Augmentation:** Integrated random flip, rotation, and zoom layers that activate only during training.
4.  **Fine-Tuning:** The top 55 layers of the base model were unfrozen and retrained to adapt specifically to plant leaf textures.

![Training Graphs](Training_Graphs/model_4_mobilenet_finetuned.png)
*(Training vs Validation Accuracy & Loss for the Final Model)*

## 🚀 Installation & Local Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/plant-disease-detection.git](https://github.com/YOUR_USERNAME/plant-disease-detection.git)
    cd plant-disease-detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

4.  **Use the App**
    * Open your browser at `http://localhost:8501`
    * Upload a leaf image (JPG/PNG)
    * View the predicted disease and confidence score.

## 📂 Directory Structure
