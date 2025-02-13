import streamlit as st
import gdown
import os
import tensorflow as tf

# Configurar la URL del modelo en Google Drive
file_id = "1XWsHx07zWLDHRCAYjI87B1QSH2wX96Vp"
url = f"https://drive.google.com/uc?id={file_id}"
output = "modelo_cnn.h5"  # Nombre del archivo local

# Descargar el modelo si no existe
if not os.path.exists(output):
    st.info("Descargando el modelo, espera un momento...")
    gdown.download(url, output, quiet=False)

# Cargar el modelo de CNN
st.success("Modelo descargado con éxito. Cargando...")
model = tf.keras.models.load_model(output)

st.success("Modelo cargado correctamente.")

# Mostrar información del modelo en Streamlit
st.write(model.summary())

# --- Cargar una imagen para la predicción ---
from PIL import Image
import numpy as np

st.title("Clasificación de imágenes con CNN")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))  # Ajustar tamaño
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de batch

    # Hacer predicción con el modelo cargado
    prediction = model.predict(img_array)
    st.write("Predicción:", prediction)
