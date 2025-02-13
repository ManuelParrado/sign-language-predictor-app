import streamlit as st
import gdown
import os
import tensorflow as tf
from PIL import Image
import numpy as np

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
st.write("Resumen del modelo:")
st.text(model.summary())

# --- Cargar una imagen para la predicción ---
st.title("Clasificación de imágenes con CNN")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Abrir imagen y cambiar tamaño
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Preprocesar imagen
    im = np.asarray(image.resize((100, 100))) / 255.0  # Normalización
    im = im.reshape(1, 100, 100, 3)  # Ajustar dimensiones

    # Hacer la predicción
    prediction = model.predict(im)

    # Mostrar resultado en Streamlit
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    st.write('Predicción: ', classes[np.argmax(prediction)])
    st.write("Porcentaje de predicción: ", prediction)
