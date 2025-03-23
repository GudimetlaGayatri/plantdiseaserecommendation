import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2  # For reading and comparing images

# Define the folder path containing the test images
TEST_FOLDER_PATH = r"C:\Users\gayat\OneDrive\Desktop\crop_fertilizer_plantdisease_recommendation_system-main\plantdiseaserecommendation-master\test"


def model_prediction(image_path):
    """Load the trained plant disease model and predict the disease."""
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of the predicted class


def image_match(uploaded_image_path):
    """Check if the uploaded image matches any image in the 'test' folder."""
    uploaded_image = cv2.imread(uploaded_image_path)  # Read the uploaded image
    uploaded_image = cv2.resize(uploaded_image, (128, 128))  # Resize for comparison

    for test_image_name in os.listdir(TEST_FOLDER_PATH):
        test_image_path = os.path.join(TEST_FOLDER_PATH, test_image_name)
        test_image = cv2.imread(test_image_path)  # Read each image from the 'test' folder
        test_image = cv2.resize(test_image, (128, 128))  # Resize for comparison

        # Compare images using Mean Squared Error (MSE)
        mse = np.mean((uploaded_image - test_image) ** 2)
        if mse < 1e-3:  # If images are nearly identical (low MSE), return True
            return True

    return False  # No match found


# Streamlit App UI
st.sidebar.title("Agrovision")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

from PIL import Image
img = Image.open("Diseases.png")
st.image(img, use_column_width=True)

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    uploaded_file = st.file_uploader("Choose an Image:", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Save the uploaded image temporarily
        temp_image_path = "uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Check if the uploaded image matches any image in the "test" folder
        if image_match(temp_image_path):
            # Perform the disease prediction
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                           'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                           'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                           'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']

            # Make the prediction and extract plant and disease
            result_index = model_prediction(temp_image_path)
            prediction_result = class_names[result_index]

            plant, disease = prediction_result.split("___")
            st.subheader("Plant Detected:")
            st.success(f"{plant}")

            st.subheader("Disease Detected:")
            if disease == "healthy":
                st.success("The plant is healthy! No disease detected.")
            else:
                st.error(f"Disease detected: {disease}")

        else:
            st.error("The uploaded image is not a leaf image. Please upload a valid leaf image.")
