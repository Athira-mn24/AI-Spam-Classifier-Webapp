# 🛡️ Enterprise AI Spam Classification Dashboard

An end-to-end Natural Language Processing (NLP) and Machine Learning application that classifies text messages into **Spam** or **Ham (Legitimate)**. This project implements a decoupled architecture to evaluate the structural and runtime performance trade-offs between **Statistical Inference (Multinomial Naive Bayes)** and **Ensemble Learning (Random Forest)**.

---

## 🚀 Key Technical Features
* **Decoupled Architecture:** Separates the heavy model training phase (`train.py`) from the user interface rendering engine (`app.py`), dropping server initialization latency to sub-milliseconds.
* **Dual-Engine Live Comparison:** Evaluates custom text inputs through two completely distinct mathematical algorithms simultaneously, displaying live predictions, confidence percentages, and execution speed (inference latency in ms).
* **Class Imbalance Mitigation:** Combats the inherent class distribution skewness (~86% Ham, ~14% Spam) using stratified sampling and dynamic class-weight balancing configurations.
* **Production-Ready UI:** Implements a fully interactive dashboard built with the Streamlit framework.

---

## 📊 System Architecture Pipeline

1. **Data Ingestion:** Sourced from the verified Kaggle SMS Spam Dataset (compiled with `latin-1` structural encoding).
2. **Text Preprocessing:** * Case folding (lowercasing global string streams).
   * Regex parsing to strip URLs, HTML tags, and numeric anomalies.
   * Universal punctuation stripping.
   * Stopword filtering using the NLTK corpus.
   * Morphological normalization via the Porter Stemmer algorithm.
3. **Feature Engineering:** Vectorization using Term Frequency-Inverse Document Frequency (TF-IDF) with integrated unigram and bigram ranges (`ngram_range=(1, 2)`).
4. **Serialization:** Freezing vectorizer features and algorithmic decision weights into a unified binary payload bundle (`model_bundle.pkl`).

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **Data Scaffolding:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn, NLTK
* **Web Framework:** Streamlit
* **Deployment/Sourcing Track:** Google Colab, GitHub Command Workflow

---

## 📂 Project Directory Structure
```text
├── AI_Spam_Classifier.ipynb   # Interactive exploratory workflow from Google Colab
├── train.py                  # Standalone data cleaning and model training pipeline
├── app.py                    # Multi-column Streamlit web application layout
├── requirements.txt          # Explicit environmental dependency registry
└── README.md                 # System documentation portal
