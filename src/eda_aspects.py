import pandas as pd
import spacy
from collections import Counter
import os

CLEAN_DATA_PATH = "data/cleaned_absa_corpus.csv"

def extract_aspect_candidates():
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"Error: {CLEAN_DATA_PATH} not found.")
        return

    print("Loading SpaCy language model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    df = pd.read_csv(CLEAN_DATA_PATH)
    categories = df['category'].unique()

    print("\n--- Top Candidate Aspect Terms per Category ---")

    for cat in categories:
        cat_reviews = df[df['category'] == cat]['review_text'].sample(
            n=min(500, len(df[df['category'] == cat])), random_state=42
        )
        
        aspect_counter = Counter()

        # Batch process reviews for speed
        for doc in nlp.pipe(cat_reviews, batch_size=50):
            for token in doc:
                # Target Nouns and Proper Nouns that are not stop words or short symbols
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2:
                    aspect_counter[token.lemma_.lower()] += 1

        print(f"\n Category: [{cat.upper()}]")
        for rank, (aspect, count) in enumerate(aspect_counter.most_common(15), 1):
            print(f"  {rank:2d}. {aspect:<15} (frequency: {count})")

if __name__ == "__main__":
    extract_aspect_candidates()