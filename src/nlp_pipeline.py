import os
import re
import pickle
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure required NLTK resources exist
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ["stopwords", "wordnet", "omw-1.4"] else f"taggers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm", disable=["ner"])
    SPACY_AVAILABLE = True
except Exception:
    nlp_spacy = None
    SPACY_AVAILABLE = False

# Filter for non-review DOM artifacts (buyer districts, footer addresses, standalone names)
JUNK_REVIEW_REGEX = re.compile(
    r"(?:\b(?:district|taluk|tehsil|township|industrial area|municipality|panchayat|"
    r"outer ring road|devarabeesanahalli|embassy tech village|flipkart internet private|"
    r"certified buyer)\b|^,\s*[a-z\s]+$)",
    re.IGNORECASE
)

# 1. Custom E-Commerce Slang, Jargon & Typo Dictionary
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
    r"\bquility\b": "quality",
    r"\bmeterial\b": "material",
    r"\bstrich\b": "stitching",
    r"\bsmootg\b": "smooth",
    r"\bshose\b": "shoes",
    r"\blite\b": "light",
    r"\bosam\b": "awesome",
    r"\bpesa\s+wasol\b": "worth money",
    r"\bpaisa\s+vasool\b": "worth money",
    r"\bsabse\s+achcha\b": "very good",
    r"\bbakwas\b": "terrible",
    r"\bbekar\b": "useless",
    r"\bprfmc\b": "performance",
    r"\bperf\b": "performance",
    r"\bsnd\b": "sound",
    r"\bvol\b": "volume",
    r"\bspkr\b": "speaker",
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
    r"\bpckg\b": "packaging"
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "doesn't", "isn't", "wasn't",
    "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't", "didn't",
    "doesnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt", "wont", "cant", "dont", "didnt", "without"
}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()


def is_valid_review(text: str) -> bool:
    """Rejects buyer location strings, Flipkart footer addresses, and standalone reviewer names."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 12 or t.startswith(","):
        return False
    if JUNK_REVIEW_REGEX.search(t):
        return False
    words = re.findall(r"[a-zA-Z]+", t)
    if len(words) < 2:
        return False
    # Reject 2-3 word Title-Case reviewer names that contain no sentiment/product keywords
    if len(words) <= 3 and t.istitle():
        common_words = {
            "good", "nice", "best", "bad", "poor", "worst", "super", "awesome",
            "shoe", "shoes", "product", "quality", "worth", "money", "comfortable", "okay"
        }
        if not {w.lower() for w in words}.intersection(common_words):
            return False
    return True


def normalize_jargon(text: str) -> str:
    """Replaces e-commerce abbreviations, Hinglish phrases, and contractions."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
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
    """Jargon normalization -> Regex cleaning -> Negation-preserving stopword removal -> Lemmatization."""
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
    """Extracts candidate noun-chunks via SpaCy POS tagging."""
    if not isinstance(text, str) or not text.strip():
        return []

    normalized = normalize_jargon(text)
    aspects = []

    if SPACY_AVAILABLE and nlp_spacy is not None:
        doc = nlp_spacy(normalized)
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha and not token.is_stop:
                aspects.append(token.lemma_)
    else:
        words = [w for w in re.sub(r"[^a-z\s]", " ", normalized).split() if w not in STOP_WORDS and len(w) > 2]
        aspects.extend(words[:5])

    seen = set()
    unique_aspects = []
    for asp in aspects:
        if asp not in seen:
            seen.add(asp)
            unique_aspects.append(asp)
    return unique_aspects[:6]


def build_tuned_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode"
    )


def run_member2_pipeline():
    input_csv = PROJECT_ROOT / "data" / "cleaned_absa_corpus.csv"
    vectorizer_out = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

    print("=" * 65)
    print("🚀 RUNNING UPGRADED MEMBER 2 PIPELINE (JUNK FILTER + JARGON)")
    print("=" * 65)

    df = pd.read_csv(input_csv)
    before_len = len(df)
    source_col = "review_text" if "review_text" in df.columns else df.columns[0]

    # Purge district/address/name junk rows permanently
    df = df[df[source_col].apply(is_valid_review)].reset_index(drop=True)
    print(f"[*] Filtered Junk/Address Rows: {before_len:,} -> {len(df):,} valid reviews")

    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(1).astype(int)
    df["cleaned_text"] = df[source_col].astype(str).apply(clean_review_text)
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)

    vectorizer = build_tuned_tfidf_vectorizer()
    X_tfidf = vectorizer.fit_transform(df["cleaned_text"])
    print(f"[+] TF-IDF Matrix Shape: {X_tfidf.shape}")

    df.to_csv(input_csv, index=False)
    vectorizer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(vectorizer_out, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[+] Saved cleaned corpus -> {input_csv}")


if __name__ == "__main__":
    run_member2_pipeline()