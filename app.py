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

# --- Load and clean the sample dataframe ---
df_samples = pd.read_csv("sample_df.csv")

# Clean to ensure accurracy 
df_samples["filepath"] = df_samples["filepath"].astype(str).str.strip()
df_samples["filepath"] = df_samples["filepath"].apply(lambda x: x.replace("\r", "").replace("\n", "").strip())
df_samples["filepath"] = df_samples["filepath"].apply(os.path.normpath)

# Check which images are missing
df_samples["exists"] = df_samples["filepath"].apply(os.path.exists)
if not df_samples["exists"].all():
    st.warning("⚠️ Some image files are missing. Please ensure all `sample_images/` are present.")
    st.dataframe(df_samples[df_samples["exists"] == False][["filepath", "label"]])

# Drop duplicates by label for dropdown preview
unique_samples = df_samples.drop_duplicates(subset="label")

# --- Load model and feature extractor ---
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
def load_model(path="best_model.pkl.gz"):
    with gzip.open(path, "rb") as f:
        model = joblib.load(f)
    return model
model = load_model()

#model = joblib.load("best_model.pkl.gz")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# --- Streamlit UI ---
st.title("📄 Financial Document Classifier")
st.write("Explore known document classes and classify your own uploads!")

# --- Dropdown to preview classes ---
st.subheader("🗂️ Explore Document Classes")
class_names = sorted(df_samples["label"].unique())
selected_label = st.selectbox("📂 Choose a document class to view an example:", class_names)

# Show one sample image from selected label
selected_row = df_samples[df_samples["label"] == selected_label].iloc[0]
selected_path = os.path.normpath(selected_row["filepath"])

if os.path.exists(selected_path):
    st.image(selected_path, caption=f"{selected_row['label']} → {selected_row['merged_label']}", width=300)
else:
    st.warning(f"❌ Missing image: {selected_path}")

st.markdown("---")

# --- Upload + Predict ---
uploaded_file = st.file_uploader("📤 Upload a document to classify", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    features = mobilenet.predict(img_array)
    prediction = model.predict(features)[0]

    # Map encoded prediction to label
    label_map = dict(zip(df_samples["encoded_merged_label"], df_samples["merged_label"]))
    predicted_merged_label = label_map.get(prediction, "Unknown")

    st.success(f"✅ Predicted Document Group: **{predicted_merged_label}**")

    # Show examples of that group
    matching_samples = df_samples[df_samples["merged_label"] == predicted_merged_label]
    if not matching_samples.empty:
        st.write("📄 Examples of this document group:")
        for _, row in matching_samples.iterrows():
            row_path = os.path.normpath(row["filepath"].strip())
            if os.path.exists(row_path):
                st.image(row_path, caption=f"{row['label']} → {row['merged_label']}", width=200)
            else:
                st.warning(f"❌ Missing image: {row_path}")
    else:
        st.warning(f"⚠️ No examples available for: {predicted_merged_label}")

# # Load sample metadata
# sample_df = pd.read_csv("sample_df.csv")
# sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip().apply(os.path.normpath)

# # Drop duplicates by label for dropdown preview
# unique_samples = sample_df.drop_duplicates(subset="label")

# # Load trained model and encoders
# def load_model():
#     with open("best_model.pkl", "rb") as f:   
#         model = joblib.load(f)
#     return model

# model = load_model()
# label_encoder = joblib.load("label_encoder.pkl")
# scaler = joblib.load("scaler.pkl")
# X_features = np.load("X_features.npy")  


# # --- Class preview ---
# st.subheader("🗂️ Explore Document Classes")
# class_names = sorted(sample_df["label"].unique())
# selected_label = st.selectbox("📂 Choose a document class to view an example:", class_names)

# # Show example image
# example_path = sample_df[sample_df["label"] == selected_label].iloc[0]["filepath"]
# if os.path.exists(example_path):
#     st.image(example_path, caption=f"Example of: {selected_label}", width=300)
# else:
#     st.warning(f"⚠️ Example image not found: {example_path}")

# st.markdown("---")

# # --- Upload for prediction ---
# st.subheader("🔍 Upload a Document to Classify")
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)

#     # Load the MobileNetV2 feature extracted for this image
#     try:
#         # This expects that you already extracted the features offline using MobileNetV2
#         feature_vector = np.load("X_features.npy")  # Replace with logic if using dynamic name
#         if feature_vector.shape[0] != 1280:
#             st.error("❌ Feature vector must be 1280-dimensional. Check preprocessing.")
#         else:
#             features_scaled = scaler.transform([feature_vector])
#             prediction = model.predict(features_scaled)[0]
#             decoded_label = label_encoder.inverse_transform([prediction])[0]

#             st.success(f"✅ Predicted Document Class: **{decoded_label}**")

#             # Show examples from predicted class
#             st.markdown("### 🖼 Example Documents from this Class")
#             matching = sample_df[sample_df["label"] == decoded_label]
#             for _, row in matching.head(3).iterrows():
#                 path = os.path.normpath(row["filepath"])
#                 if os.path.exists(path):
#                     st.image(path, caption=row["label"], width=200)
#                 else:
#                     st.warning(f"❌ Missing: {path}")
#     except Exception as e:
#         st.error(f"❌ Failed to load feature vector: {e}")

    

    
 

    
