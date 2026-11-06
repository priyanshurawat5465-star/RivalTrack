import os
# MUST be set BEFORE importing numpy, scipy, sklearn, or lightgbm
# Prevents vcomp140.dll / OpenBLAS thread-local storage NULL pointer crash on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import pickle
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

# Detect project root directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import Member 2's components
from nlp_pipeline import build_tuned_tfidf_vectorizer, clean_review_text


def sanitize_csr_for_lgbm(csr_mat: sp.csr_matrix) -> sp.csr_matrix:
    """Ensures CSR buffers are C-contiguous float32/int32 for LightGBM's C++ API."""
    csr = sp.csr_matrix(csr_mat, copy=True)
    csr.sort_indices()
    csr.data = np.ascontiguousarray(csr.data, dtype=np.float32)
    csr.indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    csr.indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    return csr


def get_layer1_base_learners():
    """
    Member 3's Exact Layer 1 Architecture:
    1. Calibrated LinearSVC (Margin-based linear classifier)
    2. MultinomialNB (Probabilistic Naive Bayes classifier)
    3. LightGBM (Gradient Boosted Decision Trees)
    (LogisticRegression is strictly reserved for the Layer 2 Meta-Learner)
    """
    base_svc = LinearSVC(
        C=0.8,
        dual="auto",
        class_weight="balanced",
        random_state=42,
        max_iter=3000
    )
    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        cv=3,
        method="sigmoid",
        n_jobs=1
    )

    mnb = MultinomialNB(alpha=0.5)

    lgbm = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        force_col_wise=True,
        verbose=-1
    )

    return {
        "LinearSVC_Calibrated": calibrated_svc,
        "MultinomialNB": mnb,
        "LightGBM": lgbm
    }


def fit_and_predict_proba(model, name: str, X_tr, y_tr, X_eval):
    """Fits the base learner and returns positive class probabilities."""
    if name == "LightGBM":
        X_tr_safe = sanitize_csr_for_lgbm(X_tr)
        X_eval_safe = sanitize_csr_for_lgbm(X_eval)
        y_tr_safe = np.ascontiguousarray(y_tr, dtype=np.int32)
        model.fit(X_tr_safe, y_tr_safe)
        return model.predict_proba(X_eval_safe)[:, 1]
    else:
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_eval)[:, 1]


def run_layer1_oof_training(
    data_path: str = None,
    models_dir: str = None,
    n_splits: int = 5
):
    print("=" * 70)
    print("🧠 MEMBER 3: LAYER 1 (LinearSVC + MultinomialNB + LightGBM) + 5-FOLD OOF")
    print("=" * 70)

    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "cleaned_absa_corpus.csv"
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"

    data_path = Path(data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(1).astype(int)

    if "cleaned_text" not in df.columns:
        source_col = "review_text" if "review_text" in df.columns else df.columns[0]
        df["cleaned_text"] = df[source_col].astype(str).apply(clean_review_text)

    df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)

    X_text = df["cleaned_text"].values
    y = np.ascontiguousarray(df["sentiment"].values, dtype=np.int32)

    # Stratified 80/20 Train-Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[*] Training Samples: {len(X_train_text):,} | Holdout Test Samples: {len(X_test_text):,}")

    # Build and fit Tuned TF-IDF Vectorizer
    vectorizer = build_tuned_tfidf_vectorizer()
    X_train_tfidf = sanitize_csr_for_lgbm(vectorizer.fit_transform(X_train_text))
    X_test_tfidf = sanitize_csr_for_lgbm(vectorizer.transform(X_test_text))

    with open(models_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    learner_names = ["LinearSVC_Calibrated", "MultinomialNB", "LightGBM"]
    n_learners = len(learner_names)
    print(f"[*] Layer 1 Base Learners: {learner_names}")

    # Pre-allocate Out-of-Fold (OOF) Probability Matrices
    oof_train_probs = np.zeros((X_train_tfidf.shape[0], n_learners), dtype=np.float32)
    oof_test_probs = np.zeros((X_test_tfidf.shape[0], n_learners), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n[*] Starting {n_splits}-Fold Out-of-Fold (OOF) Cross-Validation...")
    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_tr = X_train_tfidf[train_idx]
        y_tr = np.ascontiguousarray(y_train[train_idx], dtype=np.int32)

        X_va = X_train_tfidf[val_idx]
        y_va = np.ascontiguousarray(y_train[val_idx], dtype=np.int32)

        learners = get_layer1_base_learners()

        for i, name in enumerate(learner_names):
            model = learners[name]
            t0 = time.time()

            val_probs = fit_and_predict_proba(model, name, X_tr, y_tr, X_va)
            oof_train_probs[val_idx, i] = val_probs

            test_probs = model.predict_proba(X_test_tfidf)[:, 1]
            oof_test_probs[:, i] += test_probs / n_splits

            val_preds = (val_probs >= 0.5).astype(int)
            acc = accuracy_score(y_va, val_preds)
            f1_defect = f1_score(y_va, val_preds, pos_label=0)
            auc = roc_auc_score(y_va, val_probs)

            print(
                f"  [{name:20s}] Acc: {acc:.4f} | Defect F1: {f1_defect:.4f} "
                f"| ROC-AUC: {auc:.4f} ({time.time() - t0:.1f}s)"
            )

    print("\n" + "=" * 70)
    print("📊 5-FOLD OUT-OF-FOLD (OOF) LAYER 1 SUMMARY METRICS")
    print("=" * 70)

    layer1_report = {}
    for i, name in enumerate(learner_names):
        full_oof_probs = oof_train_probs[:, i]
        full_oof_preds = (full_oof_probs >= 0.5).astype(int)

        acc = accuracy_score(y_train, full_oof_preds)
        prec_def = precision_score(y_train, full_oof_preds, pos_label=0)
        rec_def = recall_score(y_train, full_oof_preds, pos_label=0)
        f1_def = f1_score(y_train, full_oof_preds, pos_label=0)
        roc_auc = roc_auc_score(y_train, full_oof_probs)

        layer1_report[name] = {
            "oof_accuracy": round(acc, 4),
            "defect_precision": round(prec_def, 4),
            "defect_recall": round(rec_def, 4),
            "defect_f1": round(f1_def, 4),
            "roc_auc": round(roc_auc, 4)
        }
        print(
            f"👉 {name:20s} | OOF Acc: {acc:.4f} | Defect Precision: {prec_def:.4f} | "
            f"Defect Recall: {rec_def:.4f} | Defect F1: {f1_def:.4f} | ROC-AUC: {roc_auc:.4f}"
        )

    # Fit final Layer 1 base models on 100% of training data
    print("\n[*] Fitting final Layer 1 Base Learners on full training split...")
    final_layer1_models = get_layer1_base_learners()
    for name, model in final_layer1_models.items():
        _ = fit_and_predict_proba(model, name, X_train_tfidf, y_train, X_test_tfidf[:10])

    # Save Layer 1 models and OOF matrices for Layer 2 (LogisticRegression Meta-Learner)
    with open(models_dir / "layer1_base_models.pkl", "wb") as f:
        pickle.dump(final_layer1_models, f)

    oof_payload = {
        "learner_names": learner_names,
        "oof_train_probs": oof_train_probs,
        "oof_test_probs": oof_test_probs,
        "y_train": y_train,
        "y_test": y_test,
        "layer1_report": layer1_report
    }
    with open(models_dir / "oof_layer1_data.pkl", "wb") as f:
        pickle.dump(oof_payload, f)

    # Update linearsvc_model.pkl so app.py works out-of-the-box
    with open(models_dir / "linearsvc_model.pkl", "wb") as f:
        pickle.dump(final_layer1_models["LinearSVC_Calibrated"], f)

    print(f"\n[+] Saved Layer 1 Base Models -> {models_dir / 'layer1_base_models.pkl'}")
    print(f"[+] Saved 5-Fold OOF Matrices -> {models_dir / 'oof_layer1_data.pkl'}")
    print(f"[+] Total Layer 1 + OOF Execution Time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    run_layer1_oof_training()