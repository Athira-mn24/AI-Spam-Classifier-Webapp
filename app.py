import streamlit as st
import pickle
import re
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1. PAGE LAYOUT SETUP
st.set_page_config(
    page_title="Enterprise AI Spam Detector", 
    page_icon="🛡️", 
    layout="wide"
)

# 2. SETUP TEXT TRANSFORMATION LAYERS
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove Digits
    text = re.sub(r'\d+', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# 3. LOAD FROZEN MODEL STRUCTURES INSTANTLY
@st.cache_resource
def load_saved_models():
    with open('model_bundle.pkl', 'rb') as f:
        tfidf, nb_model, rf_model = pickle.load(f)
    return tfidf, nb_model, rf_model

try:
    tfidf, nb_model, rf_model = load_saved_models()
except FileNotFoundError:
    st.error("❌ 'model_bundle.pkl' missing. Please run 'python train.py' in your Command Prompt first!")
    st.stop()

# 4. WEB PAGE USER INTERFACE
st.title("🛡️ Enterprise AI Spam Classification Dashboard")
st.markdown("An end-to-end NLP application evaluating structural tradeoffs between **Statistical Inference** and **Ensemble Learning**.")
st.write("---")

user_input = st.text_area(
    "Enter an SMS or Email message to analyze live:", 
    placeholder="Type or paste your text here... (e.g., 'Claim your free tickets now!' or 'Are we meeting at 5?')", 
    height=120
)

if st.button("Run Advanced Diagnostics", type="primary"):
    if user_input.strip() == "":
        st.warning("Please type a valid message first.")
    else:
        # Preprocess and vectorise the user input text
        processed_input = clean_text(user_input)
        vectorized_input = tfidf.transform([processed_input])
        
        # Create columns for real-time model comparison
        col1, col2 = st.columns(2)
        
        # --- ENGINE A: NAIVE BAYES ---
        with col1:
            st.subheader("Engine A: Multinomial Naive Bayes")
            t0 = time.time()
            pred = nb_model.predict(vectorized_input)
            
            # Extracting the 1D list from the 2D output matrix explicitly
            prob_matrix = nb_model.predict_proba(vectorized_input)
            prob = prob_matrix 
            
            latency = (time.time() - t0) * 1000
            
            if pred == 1:
                st.error(f"🚨 SPAM detected ({float(prob)*100:.2f}% Confidence)")
            else:
                st.success(f"✅ HAM verified ({float(prob)*100:.2f}% Confidence)")
            st.caption(f"Latency: {latency:.2f} ms")
            
        # --- ENGINE B: RANDOM FOREST ---
        with col2:
            st.subheader("Engine B: Random Forest Ensemble")
            t0 = time.time()
            pred_rf = rf_model.predict(vectorized_input)
            
            # Extracting the 1D list from the 2D output matrix explicitly
            prob_matrix_rf = rf_model.predict_proba(vectorized_input)
            prob_rf = prob_matrix_rf
            
            latency_rf = (time.time() - t0) * 1000
            
            if pred_rf == 1:
                st.error(f"🚨 SPAM detected ({float(prob_rf)*100:.2f}% Confidence)")
            else:
                st.success(f"✅ HAM verified ({float(prob_rf)*100:.2f}% Confidence)")
            st.caption(f"Latency: {latency_rf:.2f} ms")

        st.write("---")
        # Visual breakdown element for recruiters to look behind the curtain
        with st.expander("🔍 Show Text Transformation Breakdown"):
            st.write(f"**Original String Input:** `{user_input}`")
            st.write(f"**Processed Semantic Tokens:** `{processed_input}`")
            st.caption("Engine Pipeline Step: Normalization, Stopword Removal, and Case-Folding completed successfully.")