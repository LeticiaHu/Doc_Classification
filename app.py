import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import random

# Load model and resources
@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    sample_df = pd.read_csv("sample_df.csv")
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    return model, label_encoder, scaler, sample_df, mobilenet

model, label_encoder, scaler, sample_df, mobilenet = load_model_and_data()

# 🧾 App Info
with st.expander("ℹ️ About this APP"):
    st.markdown("""
    This app uses **MobileNetV2** to extract features from financial document images and classify them using a trained **Random Forest** model.

    **How it works:**
    - Upload an image
    - MobileNetV2 extracts deep features (1280-dimensional)
    - A trained Random Forest model makes the prediction
    - See examples used in training for context

    Built with TensorFlow, Scikit-learn, and Streamlit.
    """)

st.title("📄 Financial Document Classifier (MobileNetV2)")
st.write("Upload a financial document image to classify it.")

# Normalize paths
sample_df["filepath"] = sample_df["filepath"].astype(str).str.strip().apply(os.path.normpath)
sample_df["exists"] = sample_df["filepath"].apply(os.path.exists)

if not sample_df["exists"].all():
    st.warning("⚠️ Some image files are missing. Please check your sample images.")
    st.dataframe(sample_df[~sample_df["exists"]][["filepath", "label"]])

label_to_images = sample_df.groupby("label")["filepath"].apply(list).to_dict()

# 📥 Upload Section
st.header("📤 Upload a Document")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def extract_features_mobilenet(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = mobilenet.predict(img_array, verbose=0)
    return scaler.transform(features)  # shape: (1, 1280)

def predict_and_retrieve_mobilenet(img):
    features = extract_features_mobilenet(img)
    st.markdown("🔍 **Feature Vector (first 100 values)**")
    st.line_chart(features[0][:100])
    if features.shape[1] != model.n_features_in_:
        st.error(f"❌ Feature mismatch: model expects {model.n_features_in_}, got {features.shape[1]}")
        return None, None
    pred_encoded = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    example_path = random.choice(label_to_images[pred_label]) if pred_label in label_to_images else None
    return pred_label, example_path

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    pred_class, example_path = predict_and_retrieve_mobilenet(img)
    if pred_class:
        st.success(f"✅ **Predicted Class:** `{pred_class}`")
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(extract_features_mobilenet(img))[0]
            confidence = np.max(probas) * 100
            st.markdown(f"📈 **Confidence:** `{confidence:.2f}%`")
        if example_path and os.path.exists(example_path):
            st.markdown("🔁 **Example from this class:**")
            st.image(example_path, caption=pred_class, use_column_width=True)
        else:
            st.warning("⚠️ No training example available for this predicted class.")

# 📚 Explore Tab
st.header("📚 Explore Known Examples")
unique_labels = sorted(sample_df["label"].unique())
selected_label = st.selectbox("🔎 Choose a class:", unique_labels)
if st.button("🎲 Show random examples"):
    example_row = sample_df[sample_df["label"] == selected_label].sample(1, random_state=42)
    if not example_row.empty:
        preview_path = example_row.iloc[0]["filepath"]
        if os.path.exists(preview_path):
            st.image(preview_path, caption=f"Example: {selected_label}", use_column_width=True)
        else:
            st.warning("⚠️ Example image missing.")

# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# from PIL import Image
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array

# @st.cache_resource
# def load_model_and_data():
#     model = joblib.load("best_model.pkl")
#     label_encoder = joblib.load("label_encoder.pkl")
#     scaler = joblib.load("scaler.pkl")
#     #pca = joblib.load("pca.pkl")
#     sample_df = pd.read_csv("sample_df.csv")
#     mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
#     return model, label_encoder, scaler, sample_df, mobilenet

# model, label_encoder, scaler,  sample_df, mobilenet = load_model_and_data()
# with st.expander("ℹ️ About this APP"):
#     st.markdown("""
#     This app uses machine learning to classify financial documents into categories like **Email**, **Memo**, **Letter**, etc.

#     **How it works:**
#     - Upload an image
#     - We extract grayscale, texture, and edge-based features
#     - A trained Random Forest model makes the prediction
#     - Look at examples of images used on training


# st.title("📄 Financial Document Classifier")

# st.write("Upload a document image to classify it.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # if uploaded_file:
# #     try:
# #         # Load and preprocess image
# #         image = Image.open(uploaded_file).convert("RGB")
# #         image = image.resize((224, 224))
# #         img_array = img_to_array(image)
# #         img_array = np.expand_dims(img_array, axis=0)
# #         img_array = preprocess_input(img_array)

# #         # Step 1: Extract features (shape: [1, 1280])
# #         features = mobilenet.predict(img_array)

# #         # Step 2: Scale features
# #         # Step 1: Extract features (shape: [1, 1280])
# #         features = mobilenet.predict(img_array)
        
# #         # Step 2: Scale features
# #         features_scaled = scaler.transform(features)
        
# #         # Step 3: Predict directly 
# #         prediction = model.predict(features_scaled)
# #         predicted_label = label_encoder.inverse_transform(prediction)[0]


# #         st.image(image, caption="Uploaded Image", use_column_width=True)
# #         st.success(f"✅ Predicted Document Class: **{predicted_label}**")

# #     except Exception as e:
# #         st.error(f"❌ Error processing image: {e}")

# #     st.write("Feature shape after MobileNet:", features.shape)
# #     st.write("After scaler:", features_scaled.shape)
   



    

    
 

    
