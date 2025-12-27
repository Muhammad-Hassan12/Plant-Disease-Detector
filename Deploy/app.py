import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_4_mobilenet_finetuned.keras')

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Healthy',
    'Blueberry___Healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___Healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___Healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___Healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___Healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___Healthy', 'Raspberry___Healthy', 'Soybean___Healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___Healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Healthy'
]
@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ File not found: {MODEL_PATH}")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image to detect diseases. (Supported: Apple, Corn, Grape, Tomato, etc.)")


st.sidebar.title("ℹ️ Supported Plants & Diseases")
st.sidebar.markdown("This model recognizes the following **38 classes**:")

with st.sidebar.expander("See Full List of Diseases"):
    for name in CLASS_NAMES:
        st.write(f"- {name}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Leaf'):
        if model is not None:
            with st.spinner('Analyzing...'):
                try:
                    img = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)

                    predictions = model.predict(img_array)
                    score = predictions[0] 
                    
                    confidence = 100 * np.max(score)
                    class_idx = np.argmax(score)
                    predicted_class = CLASS_NAMES[class_idx]

                    CONFIDENCE_THRESHOLD = 70.0

                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ **Unknown Image Detected**")
                        st.write(f"The model is only **{confidence:.2f}%** sure. This might not be a plant leaf from the dataset.")
                        st.info("Please upload a clear image of a crop leaf (e.g., Apple, Tomato, Corn).")
                    else:
                        st.success(f"**Prediction:** {predicted_class}")
                        st.info(f"**Confidence:** {confidence:.2f}%")
                        st.progress(int(confidence))
                
                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
        else:
            st.error("Model is not loaded.")