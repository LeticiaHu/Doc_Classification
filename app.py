import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler

# --- Load and clean sample data ---
df_samples = pd.read_csv("sample_df_16class.csv")
df_samples["filepath"] = df_samples["filepath"].astype(str).str.strip().apply(os.path.normpath)
df_samples["exists"] = df_samples["filepath"].apply(os.path.exists)

if not df_samples["exists"].all():
    st.warning("⚠️ Some image files are missing.")
    st.dataframe(df_samples[df_samples["exists"] == False][["filepath", "label"]])

# --- Load pre-trained model and label encoder ---
model = joblib.load("best_model.pkl.gz")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# --- Helper function to extract simple features (grayscale pixel flattening) ---
def extract_features(img):
    img = img.convert("L").resize((64, 64))  # Resize & grayscale
    arr = np.array(img).flatten()            # Flatten to 1D array
    arr = arr / 255.0                        # Normalize
    arr = scaler.transform([arr])            # Scale using training scaler
    return arr

# --- UI ---
st.title("📄 Financial Document Classifier (Light Version)")
st.write("Upload a document and get the predicted class (no TensorFlow required).")

# --- Dropdown preview ---
st.subheader("🗂️ Explore Document Classes")
unique_classes = sorted(df_samples["label"].unique())
selected_label = st.selectbox("Choose a class to preview:", unique_classes)

sample = df_samples[df_samples["label"] == selected_label].iloc[0]
if os.path.exists(sample["filepath"]):
    st.image(sample["filepath"], caption=f"Example: {sample['label']}", width=300)
else:
    st.warning("❌ Example image not found.")

st.markdown("---")

# --- Upload and predict ---
uploaded = st.file_uploader("📤 Upload your document image", type=["jpg", "jpeg", "png"])
if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded)

    try:
        features = extract_features(img)
        prediction = model.predict(features)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Predicted Class: **{predicted_label}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
