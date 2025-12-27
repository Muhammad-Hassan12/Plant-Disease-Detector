import tensorflow as tf
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# 1. Disable OneDNN warnings (optional, cleaner logs)
# --------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --------------------------------------------------
# 2. Load model SAFELY (important fix)
# --------------------------------------------------
MODEL_PATH = "model_4_mobilenet_finetuned.keras"

print("[INFO] Loading model (safe mode)...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,        # ✅ critical
    safe_mode=False       # ✅ critical
)
print("[INFO] Model loaded successfully!")

# --------------------------------------------------
# 3. Class Names (PlantVillage dataset)
# --------------------------------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --------------------------------------------------
# 4. Prediction Function
# --------------------------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds)
    label = class_names[idx]
    confidence = preds[0][idx] * 100

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{label}\nConfidence: {confidence:.2f}%")
    plt.show()

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

# --------------------------------------------------
# 5. CLI Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("[ERROR] Image not found!")
        exit(1)

    predict_image(args.image)
