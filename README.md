## 📖 Project Description

This project presents a lightweight yet effective AI-powered document classification system built using Python, Scikit-learn, and Streamlit. The app is designed to automatically classify financial and administrative document images—such as Emails, Letters, Memos, Reports, and Forms—into predefined categories based on their visual features.

Instead of relying on text or OCR, the model extracts meaningful patterns using a combination of:

Grayscale intensity features

Texture features (Local Binary Patterns)

Edge-based features (Prewitt filters)

A Random Forest classifier is trained on these features to identify document types with competitive accuracy, making it ideal for organizations seeking an efficient way to digitize and organize scanned paperwork.

The interactive Streamlit interface allows users to:

Upload a document image

Instantly receive a predicted document class

View a confidence score and similar examples from the dataset

This tool helps bridge the gap between paper-based documentation and digital classification workflows, enabling faster processing, better organization, and reduced manual workload.


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

