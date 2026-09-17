import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# -----------------------------------
# Page Settings
# -----------------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)


# -----------------------------------
# Load Model
# -----------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model/fine_tuned_mobilenet_v2.keras"
    )

    return model


model = load_model()


# -----------------------------------
# Home
# -----------------------------------

st.title("🐱 Cat vs Dog Image Classifier")

st.write(
    "Upload an image to predict whether it is a Cat or Dog."
)

st.write(
    "Images with low prediction confidence are classified "
    "as Not a Cat or Dog."
)


# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Model Information", "Prediction"]
)


# -----------------------------------
# Home Page
# -----------------------------------

if page == "Home":

    st.header("Welcome!")

    st.write(
        "This application uses a Fine-Tuned MobileNetV2 "
        "model for Cat and Dog image classification."
    )

    st.write("### Features")

    st.write("• Upload an image")
    st.write("• Predict Cat or Dog")
    st.write("• Show prediction confidence")
    st.write("• Detect low-confidence images")


# -----------------------------------
# Model Information
# -----------------------------------

elif page == "Model Information":

    st.header("Model Information")

    st.write("### Fine-Tuned MobileNetV2")

    st.write(
        "MobileNetV2 was trained using transfer learning "
        "and fine-tuning for Cat and Dog classification."
    )

    st.write("### Hyperparameters")

    st.write("• Image Size: 224 × 224")
    st.write("• Optimizer: RMSprop")
    st.write("• Learning Rate: 0.00001")
    st.write("• Dropout: 0.2")
    st.write("• Trainable Layers: 30")

    st.write("### Test Performance")

    st.write("• Accuracy: 98.90%")
    st.write("• Precision: 98.62%")
    st.write("• Recall: 99.20%")
    st.write("• F1-Score: 98.91%")


# -----------------------------------
# Prediction
# -----------------------------------

elif page == "Prediction":

    st.header("🐾 Predict Cat or Dog")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display original image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Resize image
        resized_image = image.resize((224, 224))

        # Convert image to array
        image_array = np.array(
            resized_image
        ).astype("float32")

        # IMPORTANT:
        # Do NOT divide by 255 here.
        # Your trained model already contains normalization.

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]

        # -----------------------------------
        # Confidence Calculation
        # -----------------------------------

        if prediction >= 0.5:

            predicted_class = "Dog"
            confidence = prediction

        else:

            predicted_class = "Cat"
            confidence = 1 - prediction

        confidence_percent = confidence * 100


        # -----------------------------------
        # Unknown Image Detection
        # -----------------------------------

        # Confidence threshold
        threshold = 70


        if confidence_percent < threshold:

            st.subheader("Prediction Result")

            st.warning(
                "⚠️ Not a Cat or Dog"
            )

            st.info(
                f"Model Confidence: "
                f"{confidence_percent:.2f}%"
            )

            st.write(
                "The model is not confident enough "
                "to classify this image as a Cat or Dog."
            )

        else:

            st.subheader("Prediction Result")

            if predicted_class == "Cat":

                st.success(
                    f"🐱 Prediction: {predicted_class}"
                )

            else:

                st.success(
                    f"🐶 Prediction: {predicted_class}"
                )

            st.info(
                f"Confidence: "
                f"{confidence_percent:.2f}%"
            )