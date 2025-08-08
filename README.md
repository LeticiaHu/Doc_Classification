## 📖 Project Description

This project presents a document classification web app built using TensorFlow and Streamlit. It classifies financial documents into categories such as Email, Memo, Letter, Scientific, and more using deep features extracted from images via MobileNetV2.


🔍 How It Works
📤 Users upload an image of a financial document.

📦 A pre-trained MobileNetV2 model extracts a 1280-dimensional feature vector from the image.

📊 Features are scaled and passed into a Random Forest classifier trained on labeled document data.

✅ The app displays the predicted document class, confidence, and an example from the training set.

## 💻 Model Training
The model was trained using Google Colab for GPU-accelerated feature extraction and training.

The full training process is documented in the notebook:

📓 training_notebook.ipynb

This includes:

Loading and sampling the dataset

MobileNetV2 feature extraction

Feature scaling and PCA (optional)

Model training and evaluation

Saving the model pipeline with joblib

##  Why MobileNetV2?
Using MobileNetV2 for feature extraction allows us to:

Leverage a powerful CNN without training from scratch.

Generate robust, high-level visual features.

Reduce the model size and improve inference time — ideal for web apps.


## 📂 Dataset Access Note

Due to the large size of the dataset, it was downloaded and processed locally during development. To run the provided Python notebook or Streamlit app:

➡️ Please download the original dataset directly from Kaggle and place it in the appropriate local directory as specified in the notebook.

Source: https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg

Note: The dataset is not included in this repository.

Make sure to update file paths accordingly if your folder structure differs.

## 🌐 Streamlit App Description
This project includes a deployed interactive Streamlit web app that enables users to classify scanned financial documents into categories with just a single upload.

Key features of the app:

📤 Image Upload & Prediction: Users can upload a document image (e.g., memo, letter, form) and instantly receive the predicted document class.

🧠 Feature-Based Classification: The app uses grayscale, texture (LBP), and edge (Prewitt) features to generate machine learning-based predictions.

📈 Model Confidence & Insights: View prediction confidence and compare your image with known examples from the training dataset.

🖼️ Visual Example Gallery: Explore multiple samples from different categories to better understand the visual traits the model has learned.

ℹ️ Expandable Info Panel: Includes a detailed description of how the app works, the ML pipeline, and a disclaimer on model accuracy.

This app supports daily document triage, helping teams organize digital archives, automate workflows, and reduce manual classification effort in finance, HR, or administrative settings.

➡️ Live Demo: (https://docclassification-zwcnf5idih7a3j2qmufwzx.streamlit.app/)

## 📘 Credits
Code structure inspired by materials from Prof. Mr. Avinash Jairam, CIS 9660 - Data Mining for Business Analytics course. ChatGPT by OpenAI was used to clarify Python syntax, assist with implementation strategies, and explore alternatives for data preprocessing and modeling. All results, analysis, and business interpretations are original and completed independently.  

