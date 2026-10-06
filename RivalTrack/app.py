import streamlit as st
import pandas as pd
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data quietly if not present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------- CONFIGURATION ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "ui" in os.path.dirname(__file__) else os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "cleaned_absa_corpus.csv")
VECTORIZER_FILE = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "linearsvc_model.pkl")

NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", 
    "couldn't", "shouldn't", "wouldn't", "isn't", "aren't", 
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", 
    "doesn't", "don't", "didn't", "won't"
}

DEFECT_OVERRIDE_TERMS = {"trash", "garbage", "waste", "horrible", "terrible", "useless", "broken", "defective"}

@st.cache_resource
def load_artifacts():
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).str.strip().str.title()
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(1).astype(int)
        return df
    return pd.DataFrame()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    standard_stops = set(stopwords.words('english'))
    custom_stops = standard_stops - NEGATIONS
    cleaned_tokens = [t for t in tokens if t not in custom_stops]
    return ' '.join(cleaned_tokens)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(page_title="RivalTrack", layout="wide")
st.title("⚡ RivalTrack: Baseline Sentiment & Defect Analysis")

try:
    vectorizer, model = load_artifacts()
    df = load_data()
except Exception as e:
    st.error("Model artifacts not found. Please ensure `python src/train_baseline.py` was executed successfully.")
    st.stop()

# ---------------- SIDEBAR: DATASET OVERVIEW ----------------
with st.sidebar:
    st.header("📊 Dataset Overview")
    if not df.empty:
        total_reviews = len(df)
        total_defects = len(df[df['sentiment'] == 0])
        total_positives = len(df[df['sentiment'] == 1])
        
        st.metric("Total Scraped Corpus", f"{total_reviews:,}")
        st.metric("Total Identified Defects (0)", f"{total_defects:,}")
        st.metric("Total Positive Reviews (1)", f"{total_positives:,}")
        
        st.divider()
        st.subheader("Category Breakdown")
        cat_df = df['category'].value_counts().reset_index()
        cat_df.columns = ['Category', 'Reviews']
        st.dataframe(cat_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No dataset loaded.")

# ---------------- MAIN DASHBOARD LAYOUT ----------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Single-Sentence Test Console")
    user_input = st.text_area(
        "Enter a sample competitor review:", 
        height=140, 
        value="The camera quality is trash and battery drains fast."
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            
            # Check for explicit slang/defect keywords
            has_defect_term = any(term in user_input.lower().split() for term in DEFECT_OVERRIDE_TERMS)
            
            if prediction == 0 or has_defect_term:
                st.error("🚨 Negative / Defect Sentiment Detected")
            else:
                st.success("🟢 Positive Sentiment Detected")
            
            with st.expander("View NLP Preprocessing Details"):
                st.write("**Original Text:**", user_input)
                st.write("**Cleaned (Negations Preserved):**", cleaned_input)
        else:
            st.warning("Please enter a review to analyze.")

with col2:
    st.subheader("2. Live Defect Pareto Chart")
    if not df.empty and 'sentiment' in df.columns and 'category' in df.columns:
        defects_df = df[df['sentiment'] == 0]
        
        if not defects_df.empty:
            st.write(f"Total Scraped Reviews: **{len(df):,}**")
            st.write(f"Total Identified Defects: **{len(defects_df):,}**")
            
            defect_counts = defects_df['category'].value_counts().reset_index()
            defect_counts.columns = ['Category', 'Defect Count']
            
            st.bar_chart(data=defect_counts.set_index('Category'))
        else:
            st.info("No defect data available to chart.")
    else:
        st.info("No dataset found to generate Pareto charts.")

st.divider()