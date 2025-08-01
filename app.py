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
sample_df = pd.read_csv("sample_df.csv")
sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip().apply(os.path.normpath)

# Drop duplicates by label for dropdown preview
unique_samples = sample_df.drop_duplicates(subset="label")

# Load trained model and encoders
def load_model():
    with open("best_model.pkl", "rb") as f:   
        model = joblib.load(f)
    return model

model = load_model()
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
X_features = np.load("X_features.npy")  


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

    # Load the MobileNetV2 feature extracted for this image
    try:
        # This expects that you already extracted the features offline using MobileNetV2
        feature_vector = np.load("X_features.npy")  # Replace with logic if using dynamic name
        if feature_vector.shape[0] != 1280:
            st.error("❌ Feature vector must be 1280-dimensional. Check preprocessing.")
        else:
            features_scaled = scaler.transform([feature_vector])
            prediction = model.predict(features_scaled)[0]
            decoded_label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Predicted Document Class: **{decoded_label}**")

            # Show examples from predicted class
            st.markdown("### 🖼 Example Documents from this Class")
            matching = sample_df[sample_df["label"] == decoded_label]
            for _, row in matching.head(3).iterrows():
                path = os.path.normpath(row["filepath"])
                if os.path.exists(path):
                    st.image(path, caption=row["label"], width=200)
                else:
                    st.warning(f"❌ Missing: {path}")
    except Exception as e:
        st.error(f"❌ Failed to load feature vector: {e}")

    

    
 

    
