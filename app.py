pip install streamlit

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    model = tf.keras.models.load_model("digit_model.h5")
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")

model.save("digit_model.h5")
