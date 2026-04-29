import streamlit as st
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np

st.title("Potato Disease Detector")

model = tf.keras.models.load_model("saved/versions/1/potato_disease_model.keras")
potato_class = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess(image):
    image = Image.open(image)
    image = image.resize((256,256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, 0)
    return img_array

data_batch = st.file_uploader("Upload an image of the plant leave", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

col1, col2, col3 = st.columns(3)

if data_batch:
    for i,image in enumerate(data_batch):
        processed = preprocess(image)
        prediction = model.predict(processed)
        i+=1
        if i%3==1:
            with col1:
                st.image(image, caption=f'Disease: {potato_class[np.argmax(prediction)]}')
                st.write(f'The confidence of the image is {(np.max(prediction)*100):.2f} %')
        if i%3==2:
            with col2:
                st.image(image, caption=f'Disease: {potato_class[np.argmax(prediction)]}')
                st.write(f'The confidence of the image is {(np.max(prediction)*100):.2f} %')
        if i%3==0:
            with col3:
                st.image(image, caption=f'Disease: {potato_class[np.argmax(prediction)]}')
                st.write(f'The confidence of the image is {(np.max(prediction)*100):.2f} %')
