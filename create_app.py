import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#load model
model = load_model("best_model.h5")
class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

#app title
st.title("Lung and colon cancer histopathology classifier")
#file uploader
uploaded_file = st.file_uploader("Upload histopathological images", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((180, 180))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    #prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.markdown(f"**Prediction:** {class_names[predicted_class_index]}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
