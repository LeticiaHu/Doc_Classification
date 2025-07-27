# Doc_Classification


## 📂 Dataset Access

This project uses the [Financial Document Classification dataset](https://www.kaggle.com/datasets/swatigupta555/financial-document-classification), originally hosted on Kaggle.

Due to its large size (~9.64 GB), the full dataset is not included in this repository. Instead, a **smaller sample dataset** is provided for quick testing and deployment.

---

### 🔎 Sample Dataset (Included)

A lightweight sample version is included in this repo:
- `sample_df.csv` – metadata for the sample images
- `sample_data.zip` – a compressed folder of example images (100–250 per class)

To use it:
```bash
unzip sample_data.zip -d sample_data/

----

🔗 Full Dataset (Optional)

The full dataset (9.64 GB) can be downloaded from Google Drive:

📁 [Download Full Dataset](https://drive.google.com/file/d/1j-XnBy7TbNUkvMFaZw4g81gFdkeC1FDL/view?usp=sharing)

**Or use this command to download in code (with `gdown`)**:

```python
gdown.download("https://drive.google.com/uc?id=1j-XnBy7TbNUkvMFaZw4g81gFdkeC1FDL", "financial_docs_full.zip", quiet=False)

