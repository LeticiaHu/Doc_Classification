import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
import gzip

# --- Load model and encoder ---
@st.cache_resource
def load_model():
    with gzip.open("best_model.pkl.gz", "rb") as f:
        return joblib.load(f)

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
label_encoder = load_encoder()

# --- Load sample DataFrame ---
df_samples = pd.read_csv("sample_df_16class.csv")
df_samples["filepath"] = df_samples["filepath"].astype(str).str.strip()
df_samples["filepath"] = df_samples["filepath"].apply(os.path.normpath)
df_samples["exists"] = df_samples["filepath"].apply(os.path.exists)
df_samples = df_samples[df_samples["exists"]]

# --- UI ---
st.title("📄 Financial Document Classifier")
st.write("Upload an image to classify the document type and preview similar examples.")

# Explore known classes
class_names = sorted(df_samples["label"].unique())
selected_label = st.selectbox("📂 Explore Document Class", class_names)
sample = df_samples[df_samples["label"] == selected_label].iloc[0]
st.image(sample["filepath"], caption=sample["label"], width=250)

st.markdown("---")

# Upload and predict
uploaded = st.file_uploader("📤 Upload a document (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Your uploaded image", width=300)

    # Preprocess image (grayscale, resize, flatten)
    img = Image.open(uploaded).convert("L").resize((128, 128))
    img_array = np.array(img).flatten().reshape(1, -1)

    # Predict
    pred_encoded = model.predict(img_array)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"✅ Predicted Document Class: **{pred_label}**")

    # Show up to 5 example images
    examples = df_samples[df_samples["label"] == pred_label].head(5)
    st.subheader("📑 Example Documents in This Class:")
    for _, row in examples.iterrows():
        st.image(row["filepath"], caption=row["label"], width=150)
