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
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

st.title("Clasificación de imágenes con CNN")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    %matplotlib inline 
    pil_im = Image.open(uploaded_file)
    im = np.asarray(pil_im.resize((100,100)))
    imshow(im)
    print(im.shape)
    
    # Hacemos la prediccion. Como es una imagen solo añadimos un 1 al principio
    im = im.reshape(1,100,100,3) # Una imagen de 100x100 con tres canales
    st.write("Predicción:", prediction)
