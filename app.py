import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    sample_df = pd.read_csv("sample_df.csv")
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    return model, label_encoder, scaler, pca, sample_df, mobilenet

model, label_encoder, scaler, pca, sample_df, mobilenet = load_model_and_data()

st.title("📄 Document Classifier")
st.write("Upload a document image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Step 1: Extract features (shape: [1, 1280])
        features = mobilenet.predict(img_array)

        # Step 2: Scale features
        features_scaled = scaler.transform(features)

        # Step 3: Apply PCA to match training
        features_pca = pca.transform(features_scaled)

        # Step 4: Predict
        prediction = model.predict(features_pca)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"✅ Predicted Document Class: **{predicted_label}**")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

    st.write("Feature shape after MobileNet:", features.shape)
    st.write("After scaler:", features_scaled.shape)
    st.write("After PCA:", features_pca.shape)



    

    
 

    
