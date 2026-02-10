import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/crop_disease_model.h5")

class_names = ['Early Blight', 'Late Blight', 'Healthy']

st.title("AI-Based Crop Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Leaf Image")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
