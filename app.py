import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

# --- CLASS LABELS ---
# These must match the order in your notebook's classification report
CLASS_NAMES = [
    "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet", "fish sea_food red_sea_bream",
    "fish sea_food sea_bass", "fish sea_food shrimp",
    "fish sea_food striped_red_mullet", "fish sea_food trout"
]

# --- MODEL LOADING ---
@st.cache_resource
def load_fish_model():
    # Path to your best fine-tuned model as found in your notebook
    model_path = "C:\\Users\\z031906\\models\\mobilenetv2_fish.h5"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    
    # Load the model
    return tf.keras.models.load_model(model_path)

model = load_fish_model()

# --- APP INTERFACE ---
st.title("🐟 Multiclass Fish Classification")
st.markdown("Identify fish species using the **MobileNetV2** Fine-Tuned Model.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Calculating Prediction...'):
        # 1. Preprocess: Resize to (224, 224)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        
        # 2. Add Batch Dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Rescale: Must match your training generator (1./255)
        img_preprocessed = img_array / 255.0

        # 4. Predict
        predictions = model.predict(img_preprocessed)
        score = tf.nn.softmax(predictions[0]) # Use softmax if model output is logits
        
        # In your case, MobileNetV2 usually has a softmax Dense layer at the end
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx]) * 100

        # --- DISPLAY RESULTS ---
        st.success(f"### Result: **{CLASS_NAMES[class_idx]}**")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

        # Visualizing all probabilities
        st.write("#### Confidence for All Classes:")
        chart_data = dict(zip(CLASS_NAMES, predictions[0]))
        st.bar_chart(chart_data)