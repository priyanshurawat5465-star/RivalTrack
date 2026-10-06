# RivalTrack: E-Commerce Competitor Sentiment & Defect Analysis

RivalTrack is an end-to-end machine learning pipeline and interactive dashboard designed to extract, process, and analyze competitor product reviews. Built as a vertical slice for an engineering capstone project, it utilizes a custom asynchronous web spider to harvest authentic data, a negation-preserving NLP pipeline to clean text, and a baseline machine learning model to categorize product sentiment and component defects.

## 🌟 Key Features (Phase 1)
* **Low-Bandwidth Anti-Bot Spider:** A custom asynchronous Playwright scraper designed to bypass JS/bot walls natively. It employs route interception to block heavy media/CSS (conserving cellular data bandwidth) and includes a manual "Smart Pause" IP-rotation trigger.
* **Negation-Preserving NLP:** Custom NLTK preprocessing that strictly preserves critical negations (e.g., "not," "never," "didn't") during stopword removal to ensure accurate defect classification.
* **Sublinear TF-IDF & LinearSVC:** Text vectorization using `ngram_range=(1,2)` to prevent high-frequency terms from skewing weights, fed into a baseline Support Vector Classifier.
* **Interactive Streamlit Dashboard:** A live UI featuring a single-sentence test console and a Defect Pareto Chart for category-wide defect visualization.

## 📂 Repository Structure
```text
RivalTrack/
├── data/
│   ├── raw_scraped_corpus.csv        # Raw scraped reviews from target e-commerce URLs
│   └── cleaned_absa_corpus.csv       # Processed data prepared for Aspect-Based Sentiment Analysis
├── models/
│   ├── linearsvc_model.pkl           # Trained Baseline Classifier
│   ├── model.pkl                     # Upgraded/Ensemble Model Artifacts
│   ├── tfidf_vectorizer.pkl          # Sublinear TF-IDF Vectorizer
│   └── vectorizer.pkl                # Additional vectorizer artifacts
├── src/
│   ├── spider.py                     # Playwright data collection and IP rotation logic
│   ├── clean_data.py                 # NLTK preprocessing and negation preservation
│   ├── train_baseline.py             # LinearSVC training and evaluation pipeline
│   └── eda_aspects.py                # Exploratory Data Analysis for ABSA component mapping
├── app.py                            # Streamlit Dashboard UI
└── requirements.txt                  # Python dependencies
```

## ⚙️ Environment Setup & Installation

**Navigate to the RivalTrack directory in the terminal:**
```bash
cd RivalTrack
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Install Playwright Browsers (for the data collection spider):**
```bash
playwright install chromium
```

## 🚀 Usage Guide

### 1. Data Collection
Run the spider to scrape real-world competitor reviews. Ensure your hotspot/connection is active; the bot will chime if a CAPTCHA is detected requiring IP rotation.
```bash
python src/spider.py
```

### 2. Model Training
Execute the baseline training script to clean the data, generate the TF-IDF matrix, and train the LinearSVC model.
```bash
python src/train_baseline.py
```

### 3. Launch the Dashboard
Start the Streamlit application to access the single-sentence test console and view the Defect Pareto Charts.
```bash
streamlit run app.py
```

## 🗺️ Roadmap: Phase 2 (100% Finalization)

* **2-Tier Stacking Ensemble:** Upgrading from LinearSVC to a GPU-accelerated Stacking Ensemble (Base: LinearSVC, MultinomialNB, LightGBM | Meta: Logistic Regression).
* **Aspect-Based Sentiment Analysis (ABSA):** Implementing category-adaptive mapping to isolate exactly which components are defective (e.g., "sole" in Footwear).
* **Multi-Page Live Scraping:** Integrating the spider directly into the UI to fetch and score fresh competitor URLs on demand.
