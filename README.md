# RivalTrack — E-Commerce Competitor Sentiment & Defect Analysis

**RivalTrack** is an end-to-end machine learning and Aspect-Based Sentiment Analysis (ABSA) platform engineered to collect, process, and benchmark unstructured e-commerce product reviews across **Electronics, Footwear, Appliances, and Apparel**. Built with a **2-Tier Out-of-Fold Stacking Ensemble**, **Clause-Level Compositional Polarity Decomposition**, and an interactive **Streamlit** intelligence dashboard, RivalTrack isolates the vital 20% of failing product components driving 80% of customer complaints.

---

## Key Features

### 1. Multi-Layer Review Collection & Ingestion
- **Paginated Live Product URL Scraper (`src/live_scraper.py`):** Extracts multi-page customer reviews from Flipkart and Amazon URLs using persistent cookie sessions (`http.cookiejar`), server-rendered JSON state extraction (`window.__INITIAL_STATE__`), Schema.org JSON-LD parsing, and multi-template DOM selectors, backed by a category-pure cohort fallback.
- **Bulk Historical Spider (`src/spider.py`):** Asynchronous Playwright Chromium spider with bandwidth-saving asset interception and deduplication gates used to curate the **14,769-review** multi-category training corpus.

### 2. Domain-Specific NLP Preprocessing (`src/nlp_pipeline.py`)
- **Text & HTML Normalization:** Decodes HTML entities (`&rsquo;`, `&amp;`), expands contractions, and standardizes punctuation/whitespace.
- **Phonetic & Hinglish Jargon Mapping:** Normalizes e-commerce abbreviations, phonetic misspellings (`btry` -> `battery`, `soul` -> `sole`, `show` -> `shoe`, `technition` -> `technician`), and Hinglish expressions (`paisa vasool`, `bakwas`, `kharab`).
- **Negation-Preserving Tokenization:** Custom stopword filtering retains critical syntactic negations (`not`, `no`, `never`, `without`, `hardly`) alongside spaCy lemmatization and sublinear TF-IDF vectorization (`1–2` n-grams).

### 3. Hybrid Clause-Level ABSA & 2-Tier Stacking Ensemble (`src/absa_engine.py`)
- **Contrastive Clause Segmentation:** Splits compound reviews across contrastive pivots (`but`, `however`, `although`, `yet`) and clausal conjunctions so praised features are never falsely flagged alongside defective components.
- **Compositional Polarity & Negation Scope:** Applies a 3-token syntactic negation window to distinguish negated defects (*"no battery issues"* -> Positive) from negated praise (*"not comfortable"* -> Defect).
- **2-Tier Out-of-Fold Stacking Architecture:** Fuses compositional clause polarity with a 5-fold stratified stacking ensemble combining **Calibrated LinearSVC**, **Multinomial Naive Bayes**, and **LightGBM** base learners into a **Logistic Regression** meta-classifier.
- **Dynamic Aspect Extraction & Canonicalization:** Extracts defective noun phrases via spaCy dependency trees and consolidates synonyms and service actors into clean engineering categories.

### 4. Interactive Enterprise Analytics Dashboard (`app.py`)
- **Clause-Level ABSA Inspector:** Real-time single-sentence tester with clause-by-clause polarity breakdown and normalized token inspection.
- **Live Product URL Intelligence:** Multi-page URL scraping, batch ABSA inference, live 80/20 Component Defect Pareto chart, and one-click CSV report export.
- **Historical Pareto Analytics:** Interactive 80/20 dual-axis Pareto charts across all four product domains to pinpoint critical manufacturing bottlenecks.
- **Competitor Head-to-Head Benchmarking:** Side-by-side defect rate comparison, automated reliability winner verdict, and grouped component failure bar charts.
- **Ensemble Architecture & Metrics:** Live display of 5-fold stratified Out-of-Fold validation metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`) with Light/Dark high-contrast theme switching and persistent session state.

---

## Repository Structure

```text
RivalTrack/
├── data/
│   ├── cleaned_absa_corpus.csv      # Cleaned & deduplicated 14,769-review training corpus
│   └── raw_scraped_corpus.csv       # Raw multi-category scraped dataset
├── models/
│   ├── layer1_base_models.pkl       # Trained Layer-1 learners (LinearSVC, MultinomialNB, LightGBM)
│   ├── layer2_meta_model.pkl        # Trained Layer-2 Logistic Regression meta-classifier
│   ├── linearsvc_model.pkl          # Standalone LinearSVC baseline artifact
│   ├── model.pkl                    # Primary serialized classifier artifact
│   ├── oof_layer1_data.pkl          # 5-fold Out-of-Fold validation metrics & predictions
│   ├── tfidf_vectorizer.pkl         # Fitted sublinear TF-IDF vectorizer (stacking pipeline)
│   └── vectorizer.pkl               # Baseline TF-IDF vectorizer artifact
├── src/
│   ├── absa_engine.py               # Clause-level polarity, spaCy aspect extractor & stacking inference
│   ├── clean_data.py                # Corpus cleaning, quality gates & deduplication script
│   ├── eda_aspects.py               # Exploratory aspect frequency & category distribution analysis
│   ├── live_scraper.py              # Multi-page live URL review scraper & cohort extension engine
│   ├── nlp_pipeline.py              # Domain jargon normalization & negation-preserving NLP pipeline
│   ├── spider.py                    # Asynchronous Playwright bulk review collection spider
│   ├── train_baseline.py            # Baseline LinearSVC training & evaluation script
│   ├── train_layer1_oof.py          # Layer-1 base learners & 5-fold stratified OOF generator
│   └── train_layer_2_meta.py        # Layer-2 Logistic Regression meta-learner training script
├── app.py                           # Main Streamlit web application & analytics dashboard
├── requirements.txt                 # Python package dependencies
└── README.md                        # Project documentation
```

---

## Requirements

- **Python:** 3.10 or 3.11
- **Core Libraries:** `streamlit`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `scipy`, `plotly`, `spacy`, `playwright`
- **spaCy Language Model:** `en_core_web_sm`

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/priyanshurawat5465-star/RivalTrack.git
cd RivalTrack
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install Playwright Chromium & spaCy English model

```bash
python -m playwright install chromium
python -m spacy download en_core_web_sm
```

---

## Usage

Run all commands from the root `RivalTrack/` directory:

### 1. Clean and preprocess the corpus (optional if artifacts exist)

```bash
python src/clean_data.py
```

### 2. Train Layer-1 OOF base learners and Layer-2 meta-model

```bash
python src/train_layer1_oof.py
python src/train_layer_2_meta.py
```

### 3. Launch the RivalTrack Streamlit Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (default: `http://localhost:8501`).

---

## Model Architecture & Validation Summary

| Layer / Component | Algorithm / Technique | Role in Pipeline |
| :--- | :--- | :--- |
| **Feature Extraction** | Sublinear TF-IDF (`ngram_range=(1, 2)`) | Encodes negation-preserved lemmas and domain phrases |
| **Layer 1 — Base Learner 1** | Calibrated `LinearSVC` | High-dimensional sparse linear margin classification |
| **Layer 1 — Base Learner 2** | `MultinomialNB` | Probabilistic n-gram likelihood estimation (`0.7824` OOF Accuracy) |
| **Layer 1 — Base Learner 3** | `LightGBM` | Gradient-boosted decision trees for non-linear feature interactions (`0.8039` ROC-AUC) |
| **Layer 2 — Meta-Learner** | `LogisticRegression` | Combines 5-fold Out-of-Fold base probabilities without data leakage |
| **Aspect & Clause Engine** | spaCy POS/Dependency + Negation Scope | Isolates defective components strictly from negative clauses |

---

## Roadmap & Completion Status

- [x] **Multi-Category Data Collection:** Curated and deduplicated 14,769 reviews across Electronics, Footwear, Appliances, and Apparel.
- [x] **Negation-Preserving NLP Pipeline:** Implemented HTML unescaping, phonetic/Hinglish normalization, and negation retention.
- [x] **Layer 1 OOF & Layer 2 Meta-Learner:** Trained and serialized the complete 2-Tier Stacking Ensemble under `models/`.
- [x] **Clause-Level ABSA & Aspect Canonicalizer:** Implemented contrastive clause splitting, 3-token negation scope, and dynamic aspect extraction.
- [x] **80/20 Pareto & Competitor Head-to-Head Analytics:** Built dual-axis Pareto charts and side-by-side product reliability benchmarking.
- [x] **Enterprise Dashboard & CSV Export:** Delivered persistent multi-module UI with Light/Dark themes and downloadable defect reports.

```
