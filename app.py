import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
import gzip

# --- Load data and models ---
st.title("📄 Financial Document Classifier")
st.write("Upload a document image and classify it into the correct category. Also explore known examples.")

# Load sample metadata
sample_df = pd.read_csv("sample_df_16class.csv")
sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip().apply(os.path.normpath)

# Drop duplicates by label for dropdown preview
unique_samples = sample_df.drop_duplicates(subset="label")

# Load trained model and encoders
def load_model():
    with gzip.open("best_model.pkl.gz", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# --- Class preview ---
st.subheader("🗂️ Explore Document Classes")
class_names = sorted(sample_df["label"].unique())
selected_label = st.selectbox("📂 Choose a document class to view an example:", class_names)

# Show example image
example_path = sample_df[sample_df["label"] == selected_label].iloc[0]["filepath"]
if os.path.exists(example_path):
    st.image(example_path, caption=f"Example of: {selected_label}", width=300)
else:
    st.warning(f"⚠️ Example image not found: {example_path}")

st.markdown("---")

# --- Upload for prediction ---
st.subheader("🔍 Upload a Document to Classify")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_flat = img_array.reshape(1, -1)  # Flatten image

    # Scale features
    features_scaled = scaler.transform(img_flat)

    # Predict
    prediction = model.predict(features_scaled)[0]
    decoded_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Document Class: **{decoded_label}**")

    # Show examples from that class
    matching = sample_df[sample_df["label"] == decoded_label]
    st.markdown("### 🖼 Example Documents from this Class")
    for _, row in matching.head(3).iterrows():
        path = os.path.normpath(row["filepath"])
        if os.path.exists(path):
            st.image(path, caption=row["label"], width=200)
        else:
            st.warning(f"❌ Missing: {path}")
