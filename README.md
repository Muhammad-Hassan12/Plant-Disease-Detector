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
    git clone https://github.com/Muhammad-Hassan12/Plant-Disease-Detector.git
    cd Plant-Disease-Detector
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App and give it the Image for test**
    ```bash
    streamlit run app.py --image your_image_for_test.jpg
    ```

## 📂 Directory Structure
```console
📦 Plant-Disease-Detector
│── 📂 Deploy/                        Deployment Files
    │── 📜 app.py
    │── 📜 model_4_mobilenet_finetuned.keras
    │── 📜 requirments.txt
│── 📂 Models/                        Contains last 2 Models! (Because the first 2 were to large to upload :) )
    │── 📜 model_3_mobilenet_frozen.keras               # Model trained with the frozen weights of "MobileNetV2"
    │── 📜 model_3_mobilenet_finetuned.keras            # This one is the "Final Product"!
│── 📂 Notebooks/
    │── 📜 Complete Model Training.ipynb                # Complete Training Notebook!
│── 📂 Test_Model/
    │── 📜 app.py                                       #To test the model by your self on local machine
│── 📂 Training_Graph/                                  # Contains all the graphs of all the models :)
│── 📜 requirements.txt                                 # Requirments to download it before testing and training(If you want!) 
│── 📜 README.md                      # Project documentation
│── 📜 LICENSE
```

## 🌿 Supported Classes
The model can detect the following 38 classes:
<details>
<summary>Click to expand full list</summary>

1.  Apple___Apple_scab
2.  Apple___Black_rot
3.  Apple___Cedar_apple_rust
4.  Apple___Healthy
5.  Blueberry___Healthy
6.  Cherry___Powdery_mildew
7.  Cherry___Healthy
8.  Corn___Cercospora_leaf_spot Gray_leaf_spot
9.  Corn___Common_rust
10. Corn___Northern_Leaf_Blight
11. Corn___Healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___Healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___Healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___Healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___Healthy
24. Raspberry___Healthy
25. Soybean___Healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___Healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___Healthy
</details>

## ⚠️ Disclaimer
This AI tool is intended for educational and assistive purposes. While highly accurate, diagnoses should be confirmed by agricultural experts before taking large-scale chemical or biological action!

## 📜 License
This project is licensed under the MIT License!
