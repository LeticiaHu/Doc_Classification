import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
import gzip

# --- Load model and assets ---
@st.cache_resource
def load_model(path="best_model.pkl.gz"):
    with gzip.open(path, "rb") as f:
        model = joblib.load(f)
    return model

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
label_encoder = load_label_encoder()

# --- Load and clean the sample dataframe ---
df_samples = pd.read_csv("sample_df_16class.csv")
df_samples["filepath"] = df_samples["filepath"].astype(str).str.strip()
df_samples["filepath"] = df_samples["filepath"].apply(lambda x: os.path.normpath(x))
df_samples["exists"] = df_samples["filepath"].apply(os.path.exists)

# Drop duplicates by label for preview
unique_samples = df_samples[df_samples["exists"]].drop_duplicates(subset="label")

st.title("📄 Financial Document Classifier")
st.write("Upload a document image to classify it, or explore known classes.")

# --- Preview sample by class ---
st.subheader("🗂️ Explore Document Classes")
class_names = sorted(df_samples["label"].unique())
selected_label = st.selectbox("Choose a document class:", class_names)

sample_row = unique_samples[unique_samples["label"] == selected_label].iloc[0]
if os.path.exists(sample_row["filepath"]):
    st.image(sample_row["filepath"], caption=sample_row["label"], width=300)
else:
    st.warning(f"⚠️ Missing image: {sample_row['filepath']}")

st.markdown("---")

# --- Upload and predict ---
uploaded_file = st.file_uploader("📤 Upload a document to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale and resize
    img = Image.open(uploaded_file).convert("L").resize((128, 128))
    img_array = np.array(img).flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Document Class: **{predicted_label}**")

    # Show examples of this class
    st.subheader("📂 Example documents from this class")
    matching = df_samples[(df_samples["label"] == predicted_label) & (df_samples["exists"])]

    if not matching.empty:
        for _, row in matching.head(5).iterrows():
            st.image(row["filepath"], caption=row["label"], width=150)
    else:
        st.warning("⚠️ No example images available for this class.")
