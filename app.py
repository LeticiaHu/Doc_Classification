import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
import gzip

# --- Load model and assets ---
def load_model(path="best_model.pkl.gz"):
    with gzip.open(path, "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()
label_encoder = joblib.load("label_encoder.pkl")

# --- Load and clean the sample dataframe ---
df_samples = pd.read_csv("sample_df_16class.csv")
df_samples["filepath"] = df_samples["filepath"].apply(lambda x: os.path.normpath(x.strip()))
df_samples["exists"] = df_samples["filepath"].apply(os.path.exists)

st.title("📄 Financial Document Classifier")
st.write("Upload a document image to classify it, or explore examples from known classes.")

# --- Preview a class ---
st.subheader("🗂️ Explore Document Classes")
class_names = sorted(df_samples["label"].unique())
selected_label = st.selectbox("Choose a document class:", class_names)

# Show example image
sample_row = df_samples[df_samples["label"] == selected_label].iloc[0]
if os.path.exists(sample_row["filepath"]):
    st.image(sample_row["filepath"], caption=sample_row["label"], width=300)

st.markdown("---")

# --- Upload and classify ---
uploaded_file = st.file_uploader("📤 Upload a document", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_file).convert("L").resize((128, 128))  # adjust size to your model input
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(img_array)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Document Class: **{predicted_label}**")

    # Show more images from that class
    st.write("📂 Examples of similar documents:")
    for _, row in df_samples[df_samples["label"] == predicted_label].head(5).iterrows():
        if os.path.exists(row["filepath"]):
            st.image(row["filepath"], caption=row["label"], width=150)




## 
