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
@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")  # or .gz depending on your file
    label_encoder = joblib.load("label_encoder.pkl")
    sample_df = pd.read_csv("sample_df.csv")  # this variable is named 'sample_df'
    return model, label_encoder, sample_df   # ❌ not sample_df vs df_sample

model, label_encoder, sample_df = load_model_and_data()

# And consistently use sample_df:
sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip()
sample_df["filepath"] = sample_df["filepath"].apply(lambda x: x.replace("\r", "").replace("\n", "").strip())
sample_df["filepath"] = sample_df["filepath"].apply(os.path.normpath)
sample_df["exists"] = sample_df["filepath"].apply(os.path.exists)

# Use sample_df instead of df_samples everywhere
if not sample_df["exists"].all():
    st.warning("⚠️ Some image files are missing. Please ensure all `sample_images/` are present.")
    st.dataframe(sample_df[~sample_df["exists"]][["filepath", "label"]])

unique_samples = sample_df.drop_duplicates(subset="label")

# Map: label → list of example image paths
label_to_images = sample_df.groupby("label")["filepath"].apply(list).to_dict()

# --- Function to preprocess and load features ---
def extract_features_manual(img: Image.Image):
    img = img.resize((64, 64)).convert("L")  # Fast resize
    return np.array(img).flatten() / 255.0


# --- Predict and find similar example ---
def predict_and_retrieve(img):
    # Step 1: Extract features
    features = extract_features_manual(img).reshape(1, -1)

    # Step 2: Predict
    pred_label_encoded = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_label_encoded])[0] if label_encoder else pred_label_encoded

    # Step 3: Pick a random sample from the predicted class
    example_path = random.choice(label_to_images[pred_label])
    return pred_label, example_path

# --- Streamlit UI ---
st.title("📄 Document Classifier")

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    pred_class, example_path = predict_and_retrieve(img)

    st.markdown(f"✅ **Predicted Document Class:** `{pred_class}`")
    st.markdown("🔁 **Example from this class:**")
    st.image(example_path, caption=pred_class, use_column_width=True)
st.markdown("---")
st.header("📚 Explore Known Examples")

# Extract unique labels for dropdown
unique_labels = sorted(sample_df["label"].unique())

selected_label = st.selectbox("Choose a class to preview an example:", unique_labels)

# Filter sample_df to get one example from that class
example_row = sample_df[sample_df["label"] == selected_label].sample(1, random_state=42)

if not example_row.empty:
    example_path = example_row.iloc[0]["filepath"]
    st.image(example_path, caption=f"Example of class: {selected_label}", use_column_width=True)
else:
    st.warning("No example available for that class.")

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

    

    
 

    
