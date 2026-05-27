import pandas as pd
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

print("⏳ Step 1: Initializing NLTK and loading Kaggle dataset...")
nltk.download('stopwords', quiet=True)

# Read the local Kaggle CSV file with correct encoding
df = pd.read_csv('spam.csv', encoding='latin-1')

# The classic Kaggle dataset has 3 unnecessary blank trailing columns. We drop them here.
df = df[['v1', 'v2']]
df.columns = ['target', 'text']
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove digits
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

print("🧹 Step 2: Cleaning and Preprocessing text rows...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Stratified split to maintain balanced data ratios
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
)

print("⚙️ Step 3: Vectorizing text via TF-IDF matrix...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)

print("🤖 Step 4: Training Engine A (Naive Bayes)...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)

print("🌲 Step 5: Training Engine B (Random Forest Ensemble)...")
rf_model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

print("💾 Step 6: Exporting binary models to disk...")
# Save everything into a structured pickle bundle file
with open('model_bundle.pkl', 'wb') as f:
    pickle.dump((tfidf, nb_model, rf_model), f)

print("✅ Success! 'model_bundle.pkl' has been generated in your folder.")