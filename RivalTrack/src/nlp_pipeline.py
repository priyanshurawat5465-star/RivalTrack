import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required NLTK resources exist
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ["stopwords", "wordnet", "omw-1.4"] else f"taggers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Try loading SpaCy for POS & Noun-Chunk extraction (with graceful fallback)
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm", disable=["ner"])
    SPACY_AVAILABLE = True
except Exception:
    nlp_spacy = None
    SPACY_AVAILABLE = False

# 1. Custom E-Commerce Slang, Jargon & Typo Dictionary (Member 2 - 100% Target)
ECOMMERCE_JARGON_MAP = {
    r"\bbtry\b": "battery",
    r"\bbattry\b": "battery",
    r"\bbttry\b": "battery",
    r"\bbattr\b": "battery",
    r"\bmic\b": "microphone",
    r"\bmike\b": "microphone",
    r"\bcam\b": "camera",
    r"\bcmra\b": "camera",
    r"\bchrgr\b": "charger",
    r"\bchgr\b": "charger",
    r"\badptr\b": "adapter",
    r"\bdisp\b": "display",
    r"\bscrn\b": "screen",
    r"\bqlty\b": "quality",
    r"\bqulty\b": "quality",
    r"\bqualty\b": "quality",
    r"\bprfmc\b": "performance",
    r"\bperf\b": "performance",
    r"\bsnd\b": "sound",
    r"\bvol\b": "volume",
    r"\bspkr\b": "speaker",
    r"\b蓝牙\b": "bluetooth",
    r"\bbt\b": "bluetooth",
    r"\bconnc\b": "connection",
    r"\bwrk\b": "work",
    r"\bwrking\b": "working",
    r"\bdmg\b": "damaged",
    r"\bdamg\b": "damaged",
    r"\bwarrnty\b": "warranty",
    r"\bwrnty\b": "warranty",
    r"\bdlvry\b": "delivery",
    r"\bdelvry\b": "delivery",
    r"\bpkg\b": "packaging",
    r"\bpckg\b": "packaging",
    r"\bworthless\b": "useless",
    r"\bpathetic\b": "terrible",
    r"\btrash\b": "defective garbage",
    r"\bgarbage\b": "defective garbage"
}

# 2. Explicit Defect Override Lexicon (shared with UI and ABSA Engine)
DEFECT_OVERRIDE_TERMS = {
    "trash", "garbage", "useless", "pathetic", "worst", "horrible", "terrible",
    "defective", "broken", "damaged", "fake", "counterfeit", "waste", "fraud",
    "scam", "stopped", "dead", "overheating", "overheats", "heating", "drains",
    "drained", "lags", "lagging", "hanging", "flickering", "leaking", "torn",
    "ripped", "uncomfortable", "tight", "loose", "noisy", "sparking", "burnt"
}

# 3. Negation-Preserving Stopwords Set
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "doesn't", "isn't", "wasn't",
    "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't", "didn't",
    "doesnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt", "wont", "cant", "dont", "didnt"
}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()


def normalize_jargon(text: str) -> str:
    """Replaces e-commerce abbreviations, typos, and contractions prior to cleaning."""
    text = text.lower()
    # Expand common negative contractions explicitly so negation stays attached
    contractions = {
        "doesn't": "does not", "isn't": "is not", "wasn't": "was not",
        "won't": "will not", "can't": "can not", "didn't": "did not",
        "don't": "do not", "shouldn't": "should not", "wouldn't": "would not",
        "couldn't": "could not"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)

    for pattern, replacement in ECOMMERCE_JARGON_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_review_text(text: str) -> str:
    """
    Member 2 (100% Text Cleaner):
    Jargon normalization -> Regex cleaning -> Negation-preserving stopword removal -> WordNet Lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_jargon(text)
    text = re.sub(r"read more", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for word in text.split():
        if word not in STOP_WORDS and len(word) > 1:
            lemma = lemmatizer.lemmatize(word, pos="v")
            lemma = lemmatizer.lemmatize(lemma, pos="n")
            tokens.append(lemma)

    return " ".join(tokens)


def extract_pos_aspects(text: str) -> list:
    """
    Member 2 (100% POS / Noun-Chunk Extractor):
    Extracts candidate product aspect noun-phrases and opinion modifiers to feed into Member 3's ABSA engine.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    normalized = normalize_jargon(text)
    aspects = []

    if SPACY_AVAILABLE and nlp_spacy is not None:
        doc = nlp_spacy(normalized)
        # Extract noun chunks (e.g., 'battery life', 'sound quality', 'build material')
        for chunk in doc.noun_chunks:
            clean_chunk = " ".join([t.lemma_ for t in chunk if not t.is_stop and t.is_alpha])
            if 2 <= len(clean_chunk) <= 30:
                aspects.append(clean_chunk)
        # Also grab standalone nouns modified by adjectives (amod / acomp)
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha and not token.is_stop:
                aspects.append(token.lemma_)
    else:
        # Fallback: Pure regex/token heuristic if SpaCy model isn't loaded
        words = [w for w in re.sub(r"[^a-z\s]", " ", normalized).split() if w not in STOP_WORDS and len(w) > 2]
        aspects.extend(words[:5])

    # Deduplicate while preserving order
    seen = set()
    unique_aspects = []
    for asp in aspects:
        if asp not in seen:
            seen.add(asp)
            unique_aspects.append(asp)
    return unique_aspects[:6]


def build_tuned_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Member 2 (100% Tuned TF-IDF Vectorizer):
    Maximizes vocabulary efficiency using sublinear_tf, min_df=3, max_df=0.90, and (1,2) n-grams.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=3,           # Filters rare 1-2 occurrence spelling noise
        max_df=0.90,        # Ignores overly generic corpus terms
        sublinear_tf=True,  # Applies 1 + log(tf) dampening
        strip_accents="unicode"
    )


def run_member2_pipeline(
    input_csv: str = "data/cleaned_absa_corpus.csv",
    vectorizer_out: str = "models/tfidf_vectorizer.pkl"
):
    print("=" * 65)
    print("🚀 RUNNING MEMBER 2 (100% FINALIZED NLP & TF-IDF PIPELINE)")
    print("=" * 65)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Could not find dataset at {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"[*] Loaded corpus: {len(df):,} rows from {input_csv}")

    # Standardize category and sentiment columns
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(1).astype(int)

    # Apply upgraded jargon + negation + lemmatization pipeline
    source_col = "review_text" if "review_text" in df.columns else df.columns[0]
    print("[*] Applying E-Commerce Jargon Normalizer & Negation-Preserving Lemmatizer...")
    df["cleaned_text"] = df[source_col].astype(str).apply(clean_review_text)

    # Drop empty rows after cleaning
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)

    # Fit Tuned Sublinear TF-IDF Vectorizer
    print("[*] Fitting Tuned Sublinear TF-IDF Vectorizer (min_df=3, max_df=0.90, ngram=(1,2))...")
    vectorizer = build_tuned_tfidf_vectorizer()
    X_tfidf = vectorizer.fit_transform(df["cleaned_text"])
    print(f"[+] TF-IDF Matrix Shape: {X_tfidf.shape} (Vocabulary Size: {len(vectorizer.vocabulary_):,})")

    # Save updated dataset and vectorizer
    df.to_csv(input_csv, index=False)
    os.makedirs(os.path.dirname(vectorizer_out), exist_ok=True)
    with open(vectorizer_out, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[+] Saved upgraded corpus -> {input_csv}")
    print(f"[+] Saved tuned TF-IDF vectorizer -> {vectorizer_out}")

    # Quick POS & Jargon Demo for verification
    sample = "The btry drains super fast and mic qlty is not working at all!"
    print("\n--- Member 2 Quick Verification ---")
    print(f"Raw Input     : {sample}")
    print(f"Cleaned Output: {clean_review_text(sample)}")
    print(f"POS Aspects   : {extract_pos_aspects(sample)}")


if __name__ == "__main__":
    run_member2_pipeline()