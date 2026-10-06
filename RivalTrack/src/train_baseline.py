import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------- CONFIGURATION ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Reads directly from your EDA-processed ~14,000 row CSV file
DATA_FILE = os.path.join(BASE_DIR, "data", "cleaned_absa_corpus.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "linearsvc_model.pkl")

def train_baseline():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found! Please run clean_data.py and eda_aspects.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading EDA-processed dataset from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Check text column name (fall back to review_text if cleaned_text isn't present)
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'review_text'
    
    print(f"Dataset Size: {len(df):,} rows across categories: {df['category'].unique().tolist()}")

    # Ensure clean numeric types for sentiment
    df = df.dropna(subset=[text_col, 'sentiment']).copy()
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(1).astype(int)

    print("1. Fitting Sublinear TF-IDF Vectorization (N-Grams 1 to 2)...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, 
        ngram_range=(1, 2), 
        min_df=1,
        max_features=25000
    )
    
    X = vectorizer.fit_transform(df[text_col].astype(str))
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Training Balanced LinearSVC Classifier...")
    clf = LinearSVC(C=1.0, class_weight='balanced', random_state=42, dual=False)
    clf.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=["Defect/Negative (0)", "Positive (1)"]))

    # Save artifacts into /models/ directory
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"\nSaved model artifacts successfully to: {MODEL_DIR}")

if __name__ == "__main__":
    train_baseline()