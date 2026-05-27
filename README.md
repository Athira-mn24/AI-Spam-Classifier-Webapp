# AI-Spam-Classifier-Webapp
Beginner-friendly academic mini-project for detecting SMS spam using Python, NLP, Machine Learning, and Streamlit.

Features
Manual message prediction
CSV/TXT file upload prediction
Spam/Ham classification
Spam probability and risk meter
Suspicious keyword highlighting
Prediction history saved to history.csv
Model comparison for Naive Bayes, Logistic Regression, and SVM
Evaluation metrics and confusion matrix
Dataset
Use the SMS Spam Collection Dataset from Kaggle:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Place the dataset file as:

spam.csv
The Kaggle dataset usually contains v1 for labels and v2 for messages.

Folder Structure
spam_detection_project/
│
├── app.py
├── model_training.py
├── preprocess.py
├── requirements.txt
├── README.md
├── spam.csv
├── saved_model.pkl
├── vectorizer.pkl
└── history.csv
saved_model.pkl, vectorizer.pkl, evaluation_results.json, confusion_matrix.png, and history.csv are generated after training or using the app.

Setup
Install dependencies:

pip install -r requirements.txt
Train the machine learning models:

python model_training.py
Run the Streamlit app:

streamlit run app.py
How It Works
The text is converted to lowercase.
Punctuation is removed.
Text is tokenized into words.
Stopwords are removed using NLTK.
Words are stemmed using Porter Stemmer.
TF-IDF converts cleaned text into numerical features.
Naive Bayes, Logistic Regression, and SVM are trained and compared.
The best model is selected automatically using F1-score.
The Streamlit app loads the saved model and vectorizer for prediction.
Risk Meter
The app uses the model's spam probability:

0-40%: Safe
40-70%: Suspicious
70-100%: Spam
Notes
Keep spam.csv in the same folder as the Python files.
If model files are missing, run python model_training.py.
Uploaded CSV files should contain a message column such as message, text, sms, or v2
