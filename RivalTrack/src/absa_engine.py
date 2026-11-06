import pickle
import re
import numpy as np
from pathlib import Path
import scipy.sparse as sp

try:
    from src.nlp_pipeline import clean_review_text, normalize_jargon, nlp_spacy, SPACY_AVAILABLE
except ImportError:
    from nlp_pipeline import clean_review_text, normalize_jargon, nlp_spacy, SPACY_AVAILABLE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Phonetic & Hinglish normalization extensions
EXTRA_NORMALIZATION = {
    r"\bstich\b": "stitching",
    r"\bstiching\b": "stitching",
    r"\bstrich\b": "stitching",
    r"\bquility\b": "quality",
    r"\bmeterial\b": "material",
    r"\bpesa\s+wasol\b": "worth money",
    r"\bpaisa\s+vasool\b": "worth money",
    r"\bbadiya\b": "good",
    r"\bmast\b": "great",
    r"\bbakwas\b": "terrible",
    r"\bbekar\b": "useless",
    r"\bghatiya\b": "terrible",
    r"\bkharab\b": "damaged",
    r"\bbugget\b": "budget"
}

POSITIVE_LEXICON = {
    "good", "great", "nice", "best", "super", "superb", "awesome", "excellent", "perfect",
    "beautiful", "cute", "comfortable", "comfy", "happy", "satisfied", "satisfactory",
    "love", "loved", "amazing", "cool", "fine", "worth", "gorgeous", "smooth", "durable",
    "light", "weightless", "brilliant", "fantastic", "flawless", "sturdy", "reliable",
    "impressive", "clear", "fast", "value", "recommended", "okay", "ok", "decent", "genuine"
}

NEGATIVE_LEXICON = {
    "poor", "bad", "worst", "pathetic", "useless", "garbage", "trash", "defective", "broken",
    "broke", "break", "breaks", "breaking", "damage", "damaged", "waste", "worthless",
    "terrible", "horrible", "awful", "drain", "drains", "drained", "overheat", "overheating",
    "overheats", "lag", "lags", "lagging", "hang", "hanging", "fake", "fraud", "scam",
    "torn", "ripped", "uncomfortable", "tight", "loose", "noisy", "noise", "loud", "grinding",
    "rattling", "buzzing", "leaking", "leak", "burnt", "sparking", "stopped", "dead", "died",
    "die", "dies", "disappointed", "disappointing", "cheap", "inferior", "faulty", "slow",
    "faded", "fade", "peeling", "peel", "peeled", "cracked", "crack", "scratched", "scratch",
    "flimsy", "fragile", "painful", "smell", "stink", "rust", "rusted", "bent", "dented",
    "flimsy", "fragile", "painful", "heavy", "bulky", "stiff", "slippery", "smell", "stink",
    "missing", "wrong", "duplicate", "counterfeit", "substandard", "mediocre", "overpriced",
    "undone", "unravelling", "unstitched", "flicker", "flickering", "disconnect", "disconnecting",
    "shrink", "shrunk", "bleed", "bleeding", "separated", "unglued", "detached", "snapped", "jammed", "stuck"
}

PROBLEM_NOUNS = {
    "issue", "issues", "problem", "problems", "complaint", "complaints",
    "defect", "defects", "fault", "faults", "glitch", "glitches", "bug", "bugs",
    "error", "errors", "trouble", "pain", "regret", "drawback", "flaw", "flaws"
}

NEGATION_TOKENS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without", "zero", "none", "lack", "lacks", "lacking"
}

META_STOP_NOUNS = {
    "product", "products", "item", "items", "thing", "things", "one", "ones",
    "flipkart", "amazon", "brand", "company", "seller", "customer", "costumer", "buyer",
    "phone", "phones", "mobile", "mobiles", "smartphone", "handset", "device", "devices",
    "iphone", "samsung", "xiaomi", "redmi", "realme", "vivo", "oppo", "oneplus", "nokia", "motorola", "boat",
    "shoe", "shoes", "shose", "footwear", "sneaker", "sneakers", "apparel", "appliance", "electronics",
    "quality", "quility", "qlty", "qulty", "condition", "money", "price", "cost", "worth", "value",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "time", "times",
    "hour", "hours", "minute", "minutes", "mins", "sec", "seconds", "today", "yesterday",
    "way", "side", "sides", "part", "parts", "piece", "pieces", "pair", "pairs", "lot", "bit",
    "garbage", "trash", "waste", "rubbish", "junk", "nonsense", "scam", "fraud",
    "you", "your", "yours", "yourself", "we", "us", "they", "them", "he", "she", "him", "her",
    "slightly", "kinda", "sorta", "rather", "pretty", "quite", "fairly", "almost", "nearly",
    "better", "worse", "best", "good", "bad", "decent", "delightful", "terrific", "form", "purpose", "purposes",
    "shoe", "shoes", "shose", "sneaker", "sneakers", "footwear", "quality", "quility", "qlty", "qulty"
    "expectation", "expectations", "experience", "service", "range", "budget",
    "buy", "purchase", "order", "choice", "option", "person", "people", "guy", "guys", "all",
    "everything", "nothing", "anything", "something", "point", "star", "stars", "rating",
    "review", "reviews", "comment", "word", "words", "fact", "truth", "use", "usage", "user",
    "users", "need", "needs", "kind", "type", "types", "sort", "life",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "last", "next", "previous",
    "within", "without", "another", "other", "others", "total", "full", "half",
    "single", "double", "multiple", "several", "many", "few", "couple",
    "cash", "rupee", "rupees", "rs", "inr", "dollar", "penny",
    "wash", "wear", "spin", "cycle", "hold", "holds", "make", "makes", "made",
    "coming", "come", "comes", "came", "went", "gone", "going", "take", "takes", "took"
}

NON_ASPECT_WORDS = (
    POSITIVE_LEXICON | NEGATIVE_LEXICON | PROBLEM_NOUNS | NEGATION_TOKENS | META_STOP_NOUNS | {
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
        "is", "was", "are", "were", "be", "been", "being", "am", "has", "have", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "very", "too", "so", "quite", "really", "super", "extremely", "totally", "completely", "fully", "absolutely",
        "much", "many", "more", "most", "less", "least", "enough", "and", "or", "but", "if", "because",
        "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "any", "both", "each", "some", "such", "only", "own", "same", "than", "just", "now",
        "also", "even", "still", "already", "pls", "please", "plz", "go", "get", "got", "received", "receive",
        "bought", "buying", "using", "used", "working", "work", "works", "worked", "compare", "compared",
        "looking", "looks", "look", "feel", "feels", "felt", "seem", "seems"
    }
)

FAILURE_PATTERNS = [
    (re.compile(r"\b(?:stop|stopped|stops)\s+(?:working|work|charging|responding|functioning)\b"), "Functionality", -0.90),
    (re.compile(r"\b(?:came|come|comes|fell|fall)\s+(?:off|apart|out|loose|undone)\b"), "Build Integrity", -0.85),
    (re.compile(r"\b(?:not|never|hardly|barely)\s+(?:working|work|charging|charge|lasting|last|turning\s+on|connecting|pairing)\b"), "Functionality", -0.90),
    (re.compile(r"\b(?:not|never)\s+(?:to\s+my\s+expectation|up\s+to\s+the\s+mark|as\s+expected|as\s+shown|as\s+described|worth)\b"), "Value & Expectations", -0.85),
    (re.compile(r"\bwaste\s+of\s+(?:money|time|cash|hard\s+earned\s+money)\b"), "Value for Money", -0.95),
    (re.compile(r"\b(?:go\s+for\s+other|do\s+not\s+buy|don\s*t\s+buy|avoid\s+this|never\s+buy|return\s+it|returned\s+it)\b"), "Overall Quality", -0.85),
    (re.compile(r"\b(?:heat|heats|heating|overheat|overheats|overheating)\s+(?:up|a\s+lot|too\s+much|badly|quickly|fast)\b"), "Heating", -0.85),
    (re.compile(r"\b(?:drain|drains|drained|draining|die|dies|died)\s+(?:super\s+fast|very\s+fast|fast|quickly|within|in)\b"), "Battery", -0.90),
    (re.compile(r"\b(?:broke|broken|damaged|dead|died|cracked|tore|torn|peeled|faded)\s+(?:on|within|after|in)\s+(?:day|days|week|weeks|month|months|first|one|two|three|\d+)\b"), "Durability", -0.90)
]

SARCASM_PATTERNS = [
    re.compile(
        r"\b(?:great|awesome|amazing|best|wonderful|fantastic|superb|nice)\b.*?\b(?:broke|broken|died|dead|stopped|garbage|trash|useless|paperweight|brick|warmer|heater|waste|damaged|defective)\b",
        re.IGNORECASE
    )
]

HEDGED_DOUBT_PATTERNS = [
    re.compile(r"\bnot\s+sure\s+about\b", re.IGNORECASE),
    re.compile(r"\b(?:will\s+not|do\s+not|cannot|can\s+not)\s+expect\b", re.IGNORECASE)
]


def full_normalize(text: str) -> str:
    t = normalize_jargon(text)
    for pat, rep in EXTRA_NORMALIZATION.items():
        t = re.sub(pat, rep, t)
    return t


def segment_clauses(text: str) -> list:
    """Decomposes a compound review into atomic clauses across contrastive pivots and clausal conjunctions."""
    parts = re.split(r"([.!?;]+|\b(?:but|however|although|though|whereas|yet|except)\b)", text, flags=re.IGNORECASE)
    raw_clauses = []
    contrast_active = False

    for part in parts:
        p = part.strip()
        if not p:
            continue
        if re.match(r"^(?:but|however|although|though|whereas|yet|except)$", p, flags=re.IGNORECASE):
            contrast_active = True
            continue
        if re.match(r"^[.!?;]+$", p):
            contrast_active = False
            continue

        sub_parts = re.split(
            r"\band\b(?=\s+(?:the|my|its|this|that|another|one|left|right|also|[a-z]+\s+(?:is|was|are|were|has|have|does|did|drains|stops|broke|came|feels|looks)))",
            p,
            flags=re.IGNORECASE
        )
        for sp_item in sub_parts:
            if sp_item.strip():
                raw_clauses.append((sp_item.strip(), contrast_active))

    return raw_clauses if raw_clauses else [(text.strip(), False)]


def score_clause_polarity(clause: str) -> dict:
    """Computes compositional polarity of a single clause in [-1.0, +1.0] using syntactic negation windows."""
    norm = full_normalize(clause)

    if any(p.search(norm) for p in HEDGED_DOUBT_PATTERNS):
        tokens_check = set(re.findall(r"[a-z]+", norm))
        if not tokens_check.intersection(NEGATIVE_LEXICON):
            return {"score": 0.10, "is_negative": False, "is_positive": False, "implicit_aspect": None}

    score = 0.0
    implicit_aspect = None

    for sp_pat in SARCASM_PATTERNS:
        if sp_pat.search(norm):
            score -= 0.95

    for pat, asp, val in FAILURE_PATTERNS:
        if pat.search(norm):
            score += val
            if implicit_aspect is None:
                implicit_aspect = asp

    tokens = re.findall(r"[a-z]+", norm)
    n = len(tokens)

    for i, tok in enumerate(tokens):
        window_start = max(0, i - 3)
        preceding = tokens[window_start:i]
        is_negated = any(w in NEGATION_TOKENS for w in preceding)

        if tok in PROBLEM_NOUNS:
            if is_negated:
                score += 0.85  # Negated defect = Positive ("no battery issues")
            else:
                score -= 0.65  # Unnegated problem noun ("heating issue")
        elif tok in POSITIVE_LEXICON:
            if tok == "fast" and any(
                w in {"drain", "drains", "drained", "die", "dies", "heat", "heats", "heating"}
                for w in tokens[max(0, i - 3):min(n, i + 2)]
            ):
                continue
            if is_negated:
                score -= 0.85  # Negated positive = Defect ("not good", "not working")
            else:
                score += 0.60
        elif tok in NEGATIVE_LEXICON:
            if is_negated:
                score += 0.55  # Negated negative ("not bad", "never lags")
            else:
                score -= 0.80

    pos_emojis = sum(1 for ch in clause if ch in "👍👌😍❤️🥰💯🌟😄😁😊🔥")
    neg_emojis = sum(1 for ch in clause if ch in "👎😡🤬🤮💔😢😭😞")
    score += 0.35 * pos_emojis - 0.60 * neg_emojis

    score = float(np.clip(score, -1.0, 1.0))
    return {
        "score": score,
        "is_negative": score <= -0.35,
        "is_positive": score >= 0.30,
        "implicit_aspect": implicit_aspect
    }

# Universal Aspect Synonym & Service Canonicalizer (Groups typos, human actors & synonyms across all categories)
ASPECT_CANONICAL_MAP = {
    # Appliances & Installation / Service actors (including Flipkart's Jeeves service)
    r"\b(?:install|installation|instalation|installer|technician|technition|tecnician|engineer|partner|mechanic|demo|uninstallation|mounting|fitting|jeeves|jeeve)\b": "Installation & Service",
    r"\b(?:call|calls|calling|customer\s*care|support|helpline|complaint\s*number|response|service\s*center|warranty|guarantee)\b": "Customer Support & Warranty",
    r"\b(?:cool|cooling|chill|chilling|temperature|airflow|air\s*throw|throw|tonnage|capacity|room\s*cooling)\b": "Cooling Performance",
    r"\b(?:compressor|gas|refrigerant|leakage|water\s*leak|condenser|coil|copper|pipe|drain\s*pipe|outdoor\s*unit|indoor\s*unit|pcb|motor|blade|jar|drum|heating\s*element|stabilizer)\b": "Core Hardware & Motor",
    r"\b(?:noise|noisy|sound|vibration|vibrating|rattling|humming|grinding|fan\s*noise|loud)\b": "Noise & Vibration",
    r"\b(?:remote|wifi|wi\s*fi|smart\s*app|connectivity|bluetooth|pairing|sensor|swing|flap|display\s*panel|led)\b": "Smart Controls & Connectivity",
    r"\b(?:power|electricity|bill|consumption|voltage|inverter|energy)\b": "Power Efficiency",

    # Electronics
    r"\b(?:battery|btry|battry|backup|drain|draining|charging|charger|adapter|cable|port|type\s*c)\b": "Battery & Charging",
    r"\b(?:screen|display|touch|amoled|lcd|brightness|flicker|dead\s*pixel|glass)\b": "Display & Screen",
    r"\b(?:camera|cam|photo|picture|video|selfie|lens|focus|portrait)\b": "Camera Quality",
    r"\b(?:mic|microphone|mike|voice|earpiece|speaker|audio|bass|treble|volume|anc|earbud|left\s*earbud|right\s*earbud)\b": "Audio & Microphone",
    r"\b(?:heat|heating|overheat|overheating|warm|thermal|lag|lagging|hang|hanging|slow|processor|ram|performance)\b": "Heating & Performance",

    # Footwear & Apparel
    r"\b(?:sole|soul|insole|bottom|bottom\s*face|grip|heel|cushion|cushioning|arch|slip|slippery)\b": "Sole & Grip",
    r"\b(?:stitch|stitching|strich|stich|seam|glue|pasting|finishing|finish|thread|zipper|zip|button|buckle|lace|laces)\b": "Stitching & Finishing",
    r"\b(?:size|sizing|fit|tight|loose|narrow|wide|shoe\s*bite|comfort|comfortable|uncomfortable)\b": "Size, Fit & Comfort",
    r"\b(?:weight|heavy|bulky|lightweight)\b": "Weight & Bulkiness",
    r"\b(?:material|meterial|fabric|cloth|cotton|leather|mesh|synthetic|plastic|foam|build|durability|durable|quality|overall\s*build)\b": "Material & Build Quality",
    r"\b(?:colour|color|shade|fade|fading|print|dye|bleed|bleeding|look|design|appearance)\b": "Colour & Finish",
    r"\b(?:smell|smells|odor|odour|stink)\b": "Material Odor",

    # Universal E-Commerce Logistics & Pricing
    r"\b(?:package|packaging|pkg|box|seal|delivery|courier|shipping|damaged\s*box|missing\s*item|wrong\s*item|exchange|return|refund)\b": "Packaging & Delivery",
    r"\b(?:price|pricing|cost|money|worth|value|overpriced|expensive)\b": "Value for Money"
}


def canonicalize_aspect(raw_phrase: str) -> str:
    """Strictly maps extracted words to clean engineering categories, blocking random typos/names like 'Jeeve' or 'Soo'."""
    clean_p = raw_phrase.strip().lower()
    if not clean_p or clean_p in META_STOP_NOUNS:
        return ""
    for pattern, canonical_label in ASPECT_CANONICAL_MAP.items():
        if re.search(pattern, clean_p, re.IGNORECASE):
            return canonical_label
    # Return empty string for unrecognized out-of-vocabulary proper nouns/typos so charts remain 100% clean
    return ""

def extract_dynamic_aspects_from_clause(clause: str, implicit_aspect: str = None) -> list:
    norm = full_normalize(clause)
    raw_extracted = []

    # 1. Check direct domain canonical triggers inside the negative clause first
    for pattern, canonical_label in ASPECT_CANONICAL_MAP.items():
        if re.search(pattern, norm, re.IGNORECASE):
            if canonical_label not in raw_extracted:
                raw_extracted.append(canonical_label)

    # 2. Also run SpaCy POS/compound noun extraction for any unseen product-specific components
    if SPACY_AVAILABLE and nlp_spacy is not None:
        doc = nlp_spacy(norm)
        for token in doc:
            lemma = token.lemma_.lower()
            word = token.text.lower()
            if (
                token.pos_ == "NOUN"
                and lemma not in NON_ASPECT_WORDS
                and word not in NON_ASPECT_WORDS
                and len(lemma) > 2
            ):
                compounds = [
                    c.lemma_.lower() for c in token.lefts
                    if c.dep_ == "compound"
                    and c.lemma_.lower() not in NON_ASPECT_WORDS
                    and len(c.lemma_) > 2
                ]
                phrase = " ".join(compounds + [lemma])
                canon = canonicalize_aspect(phrase)
                if canon and canon not in raw_extracted:
                    raw_extracted.append(canon)
    else:
        tokens = re.findall(r"[a-z]+", norm)
        for w in tokens:
            if w not in NON_ASPECT_WORDS and len(w) > 2:
                canon = canonicalize_aspect(w)
                if canon and canon not in raw_extracted:
                    raw_extracted.append(canon)

    if not raw_extracted and implicit_aspect:
        canon_imp = canonicalize_aspect(implicit_aspect)
        raw_extracted.append(canon_imp if canon_imp else implicit_aspect)

    return raw_extracted[:3]

class ABSAEngine:
    def __init__(self):
        models_dir = PROJECT_ROOT / "models"
        with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(models_dir / "layer1_base_models.pkl", "rb") as f:
            self.layer1_models = pickle.load(f)
        with open(models_dir / "layer2_meta_model.pkl", "rb") as f:
            self.layer2_meta = pickle.load(f)

    def _sanitize_for_lgbm(self, csr_mat):
        csr = sp.csr_matrix(csr_mat, copy=True)
        csr.sort_indices()
        csr.data = np.ascontiguousarray(csr.data, dtype=np.float32)
        csr.indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
        csr.indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
        return csr

    def _ml_ensemble_probability(self, text: str) -> float:
        clean_txt = clean_review_text(text)
        if not clean_txt.strip():
            return 0.85

        X_vec = self.vectorizer.transform([clean_txt])
        l1_probs = []
        for name, model in self.layer1_models.items():
            if name == "LightGBM":
                X_safe = self._sanitize_for_lgbm(X_vec)
                try:
                    prob = model.predict_proba(X_safe)[:, 1][0]
                except OSError:
                    prob = model.predict_proba(np.ascontiguousarray(X_safe.toarray(), dtype=np.float32))[:, 1][0]
            else:
                prob = model.predict_proba(X_vec)[:, 1][0]
            l1_probs.append(prob)

        X_meta = np.array([l1_probs], dtype=np.float32)
        return float(self.layer2_meta.predict_proba(X_meta)[:, 1][0])

    def analyze(self, text: str) -> dict:
        """
        Dynamic Clause-Level ABSA + 2-Tier Stacking Ensemble Fusion.
        Evaluates each clause independently, weights post-contrastive clauses,
        and extracts defective component nouns strictly from negative clauses.
        """
        clauses = segment_clauses(text)
        clause_evals = []

        for clause_str, is_post_contrast in clauses:
            ling = score_clause_polarity(clause_str)
            ml_pos = self._ml_ensemble_probability(clause_str)
            clause_evals.append({
                "clause": clause_str,
                "post_contrast": is_post_contrast,
                "ling_score": ling["score"],
                "is_negative": ling["is_negative"],
                "is_positive": ling["is_positive"],
                "implicit_aspect": ling["implicit_aspect"],
                "ml_pos_prob": ml_pos
            })

        full_ml_pos = self._ml_ensemble_probability(text)
        has_neg_clause = any(c["is_negative"] for c in clause_evals)
        has_pos_clause = any(c["is_positive"] for c in clause_evals)

        # Fuse compositional clause polarity with 2-Tier Stacking ML probability
        if has_neg_clause:
            min_ling = min(c["ling_score"] for c in clause_evals if c["is_negative"])
            ling_defect_conf = min(0.96, 0.55 + abs(min_ling) * 0.40)
            ml_defect_conf = 1.0 - full_ml_pos
            confidence = 0.65 * ling_defect_conf + 0.35 * max(ml_defect_conf, 0.55)
            pred_label = "Defect"
        elif has_pos_clause and not has_neg_clause:
            max_ling = max(c["ling_score"] for c in clause_evals if c["is_positive"])
            ling_pos_conf = min(0.96, 0.55 + max_ling * 0.40)
            confidence = 0.60 * ling_pos_conf + 0.40 * max(full_ml_pos, 0.60)
            pred_label = "Positive"
        else:
            if full_ml_pos < 0.40:
                pred_label = "Defect"
                confidence = 1.0 - full_ml_pos
            else:
                pred_label = "Positive"
                confidence = full_ml_pos

        # Extract defective aspects strictly from negative clauses
        defects = []
        if pred_label == "Defect":
            neg_clauses = [c for c in clause_evals if c["is_negative"]]
            target_clauses = neg_clauses if neg_clauses else clause_evals
            for c in target_clauses:
                extracted = extract_dynamic_aspects_from_clause(c["clause"], c["implicit_aspect"])
                for asp in extracted:
                    if asp not in defects:
                        defects.append(asp)
            if not defects:
                defects.append("Overall Build / Quality")

        return {
            "text": text,
            "sentiment": pred_label,
            "confidence": float(np.clip(confidence, 0.55, 0.98)),
            "defective_aspects": defects,
            "clause_breakdown": clause_evals
        }