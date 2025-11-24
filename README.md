# 🧠 Human Stress Prediction

This repository contains an end-to-end NLP pipeline for predicting **human stress** from text.  
The project uses a modular Python architecture (data loading, cleaning, feature extraction, modeling) and can be run either as reusable Python modules or via the included Jupyter notebook.



## 📌 Project Goals

- Predict whether a given text indicates **stress** or **no stress**
- Build a **reproducible** pipeline for:
  - Data loading and preprocessing  
  - Text cleaning and feature extraction  
  - Model training, evaluation, and prediction
- Experiment with classic **machine learning models** (e.g. Logistic Regression) using TF-IDF and n-gram features.

> 💡 This project is mainly for learning and experimentation in **NLP**, **text classification**, and **ML pipelines**.

## 🧾 Dataset

This project uses **Reddit-sourced text data** collected from stress-related communities and discussions.
The dataset includes user-generated posts labeled as **stress** or **no stress**, making it suitable for supervised NLP classification.

Each row contains a **text** field and a label (e.g., ```0 = no stress```, ```1 = stress```).

Data is preprocessed to remove noise, normalize text, and prepare it for feature extraction.

> 🔧 Make sure to adjust the dataset path or column names in ```data_loader.py``` and ```data_preprocessor.py``` depending on your local folder structure.

## ✨ Features

- **Modular NLP pipeline**
- **Text cleaning** (punctuation removal, stopwords, tokenization)
- **TF-IDF feature extraction** with optional n-grams
- **Supervised ML models** for stress classification
- **Evaluation utilities** for classification metrics
- **Simple predictor interface** for new text inputs

## 🗂 Project Structure

```text
Human-Stress-Prediction/
├── data_loader.py        # Reads the dataset and returns raw data
├── data_preprocessor.py  # Handles preprocessing steps (train/test split, label handling, etc.)
├── text_cleaner.py       # Text normalization (lowercasing, stopwords, punctuation, etc.)
├── feature_extractor.py  # TF-IDF / n-gram feature extraction from cleaned text
├── stress_model.py       # Model training, evaluation, and persistence
├── stress_predictor.py   # High-level interface for loading a model and making predictions
├── test_notebook.ipynb   # Example notebook demonstrating the full pipeline
└── __pycache__/          # Python cache files (auto-generated)
```

## 🚀 Usage
**▶️ Run the Pipeline via Notebook**

Open the example notebook to run the full workflow:
```bash
jupyter notebook test_notebook.ipynb
```
The notebook demonstrates:

- Loading the dataset
- Cleaning text
- Extracting TF-IDF features
- Training and evaluating models
- Making predictions on custom text

## 🧭 Future Improvements

- Hyperparameter tuning
- Additional ML models (SVM, Random Forest, etc.)
- Word embeddings (Word2Vec, GloVe)
- Transformer-based methods
- Simple web UI/API for real-time predictions

## 👤 Author
This project was developed by **Nursena Taşkıran** as part of ongoing work in natural language processing and text classification.

