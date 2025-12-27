import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration
MODEL_PATH = 'model_4_mobilenet_finetuned.keras'

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
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# UI Layout
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image to detect potential diseases.")

st.sidebar.title("About")
st.sidebar.info(
    "This AI model uses MobileNetV2 to classify plant diseases from leaf images. "
    "It supports 38 different classes including Apple, Corn, Potato, Tomato, and more."
)

# Processing
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Leaf'):
        if model is not None:
            with st.spinner('Analyzing...'):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                predictions = model.predict(img_array)
                score = predictions[0] 
                class_idx = np.argmax(score)
                predicted_class = CLASS_NAMES[class_idx]
                confidence = 100 * np.max(score)

                # 4. Display
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                st.progress(int(confidence))
        else:
            st.error("Model could not be loaded. Please check the file.")