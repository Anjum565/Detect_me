# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('Test_Disease.h5', compile=False)

# Name of Classes
CLASS_NAMES = ['Good_Leaf', 'Red_Rust', 'Red_Spider_Mite','Brown_Blight']

# Disease Information
disease_info = {
    'Good_Leaf': """
        **The is a Non diseased tea leaf**
    """,
    'Red_Rust': """
        **Description :** 
        
        Red rust is a common disease of tea plants. It is caused by a fungus that infects the leaves, stems, and buds of the plant. The disease is most common in warm, humid conditions. The fungus produces red, powdery spores that are easily spread by the wind. The spores can survive for long periods in the soil and on plant debris. The disease can be controlled by planting resistant varieties, pruning infected branches, and spraying the plants with fungicides. The disease can also be prevented by avoiding overhead watering and keeping the plants well-spaced to allow for good air circulation.

    """,
    'Red_Spider_Mite': """
        **Description :** 
        
        Red spider mites are common pests of indoor plants and those grown in greenhouses. They are extremely small, but can cause a lot of damage. They are usually found on the underside of leaves, where they spin protective silk webs. The mites feed by piercing the plant tissue and sucking out the sap. This causes the leaves to become stippled, discolored, and distorted. The mites are difficult to see without a magnifying glass, but their webs are usually visible. The mites are most active in warm, dry conditions. They can be controlled by increasing humidity, keeping the plants well-watered, and spraying the leaves with insecticidal soap.

    """,

    'Brown_Blight': """
        **Description :** 
        
        Brown blight is a common disease of tea plants. It is caused by a fungus that infects the leaves, stems, and buds of the plant. The disease is most common in warm, humid conditions. The fungus produces brown, powdery spores that are easily spread by the wind. The spores can survive for long periods in the soil and on plant debris. The disease can be controlled by planting resistant varieties, pruning infected branches, and spraying the plants with fungicides. The disease can also be prevented by avoiding overhead watering and keeping the plants well-spaced to allow for good air circulation.
    """
}

# Setting Title of App
st.markdown(f'<p style="font-size:24px;"><b>Tea Disease Detection(A Plant Disease Detection Tool)</b></p>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:18px;"><b>Upload an image of the plant leaf</b></p>', unsafe_allow_html=True)


st.markdown(f'<p style="font-size:18px;">Choose an image...</p>', unsafe_allow_html=True)
# Uploading the Plant image
plant_image = st.file_uploader("", type=["png","jpg"])
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (1000, 1000))

        # Convert image to 4 Dimension
        opencv_image.shape = (1, 1000, 1000, 3)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

         # Display disease information
        st.markdown(f'<p style="font-size:22px;"><b>This is a tea leaf with {result}</b></p>', unsafe_allow_html=True)  
        st.markdown(disease_info[result])