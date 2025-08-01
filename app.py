import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
import random


@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    sample_df = pd.read_csv("sample_df.csv")
    return model, label_encoder, scaler, pca, sample_df

# Load everything
model, label_encoder, scaler, pca, sample_df = load_model_and_data()

# --- Clean filepaths ---
sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip()
sample_df["filepath"] = sample_df["filepath"].apply(lambda x: x.replace("\r", "").replace("\n", "").strip())
sample_df["filepath"] = sample_df["filepath"].apply(os.path.normpath)
sample_df["exists"] = sample_df["filepath"].apply(os.path.exists)

if not sample_df["exists"].all():
    st.warning("⚠️ Some image files are missing. Please ensure all `sample_images/` are present.")
    st.dataframe(sample_df[~sample_df["exists"]][["filepath", "label"]])

# Map: label → list of filepaths for example retrieval
label_to_images = sample_df.groupby("label")["filepath"].apply(list).to_dict()


# --- Feature Extraction ---
def extract_features_manual(img: Image.Image):
    img = img.resize((64, 64)).convert("L")  # Grayscale
    raw = np.array(img).flatten() / 255.0    # Normalized 1D feature vector (4096,)

    # Apply scaler and PCA to match training pipeline
    scaled = scaler.transform([raw])         # Shape: (1, 4096)
    reduced = pca.transform(scaled)          # Shape: (1, 300)
    return reduced


# --- Prediction + Similar Image Retrieval ---
def predict_and_retrieve(img):
    features = extract_features_manual(img)  # Already PCA-reduced shape: (1, 300)

    # Feature size check
    if features.shape[1] != model.n_features_in_:
        st.error(f"❌ Feature size mismatch: model expects {model.n_features_in_}, but got {features.shape[1]}")
        return None, None

    # Predict class
    pred_encoded = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0] if label_encoder else pred_encoded

    # Retrieve a similar known example
    example_path = None
    if pred_label in label_to_images:
        example_path = random.choice(label_to_images[pred_label])

    return pred_label, example_path

    st.write("Feature shape:", features.shape)
    st.write("Model expects:", model.n_features_in_)

# --- Streamlit UI ---
st.title("📄 Financial Document Classifier")
st.write("Upload a document image and classify it into the correct category. Also explore known examples.")

uploaded_file = st.file_uploader("📤 Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    pred_class, example_path = predict_and_retrieve(img)

    if pred_class:
        st.markdown(f"✅ **Predicted Document Class:** `{pred_class}`")
        if example_path and os.path.exists(example_path):
            st.markdown("🔁 **Example from this class:**")
            st.image(example_path, caption=pred_class, use_container_width=True)
        else:
            st.warning("⚠️ No known example available for this predicted class.")

# --- Explore Known Examples ---
st.markdown("---")
st.header("📚 Explore Known Examples")

unique_labels = sorted(sample_df["label"].unique())

selected_label = st.selectbox("🔎 Choose a class to preview an example:", unique_labels)

example_row = sample_df[sample_df["label"] == selected_label].sample(1, random_state=42)

if not example_row.empty:
    preview_path = example_row.iloc[0]["filepath"]
    if os.path.exists(preview_path):
        st.image(preview_path, caption=f"Example of class: {selected_label}", use_container_width=True)
    else:
        st.warning("⚠️ Sample image file is missing.")



    

    
 

    
