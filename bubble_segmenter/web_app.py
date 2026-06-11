import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from pillow_heif import register_heif_opener
register_heif_opener()

from config import IMG_SIZE, MODEL_NAME
from segmenting_model import bce_dice_loss

# Load model
model = keras.models.load_model(
    f"{MODEL_NAME}.keras",
    custom_objects={"bce_dice_loss": bce_dice_loss}
)

# Upload image button
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "heic"])

# Handles uploaded images
if uploaded_image is not None:
    # Processes image
    image = Image.open(uploaded_image)
    image = np.array(image)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Put the uploaded image through the model
    results = model.predict(image)

    # Handle the model prediction
    mask = (results.squeeze() > 0.5).astype(np.uint8)
    image = (image.squeeze()*255).astype(np.uint8)
    binary_mask = mask.astype(bool)
    overlay = image.copy()
    strength = 0.3
    overlay[binary_mask] = overlay[binary_mask] * (1 - strength) + np.array([0, 255, 0]).astype(np.uint8) * strength

    # Display the original image along with the prediction
    st.image(overlay)
    st.image(image)
