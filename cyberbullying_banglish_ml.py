# =========================
# Bengali/Banglish Hate Speech Detection - SVM Only (Kaggle Output)
# Minimal pipeline + conference-ready artifacts
# =========================

import time
import json
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import string
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils import resample

# -----------------------------
# Config / output paths
# -----------------------------
INPUT_CSV = "/kaggle/input/banglish-hate-spech/Banglish Hate Speech Dataset.csv"  # change if needed
OUTPUT_DIR = "/kaggle/working/svm_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# NLTK stopwords (optional)
# -----------------------------
print("📦 Downloading NLTK stopwords (quiet)...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

sns.set_palette("husl")
plt.style.use('seaborn-v0_8')

print("🧭 Starting SVM-only pipeline (conference-ready outputs)")

# -----------------------------
# Load dataset
# -----------------------------
print("📥 Loading dataset...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"📊 Dataset shape: {df.shape}")
    print("📊 Class distribution (before cleaning):")
    # try common column names
    if 'Hate' in df.columns:
        print(df['Hate'].value_counts())
    elif 'hate' in df.columns:
        print(df['hate'].value_counts())
    else:
        print(df.iloc[:, -1].value_counts().head())
except FileNotFoundError as e:
    print(f"⚠️ Error: Dataset not found at {INPUT_CSV} — adjust INPUT_CSV path.")
    raise e

# -----------------------------
# Identify text & label columns (adjust if different)
# -----------------------------
# Adapt to common column names used in your dataset
if 'Comment' in df.columns:
    TEXT_COLUMN = 'Comment'
elif 'sentence' in df.columns:
    TEXT_COLUMN = 'sentence'
elif 'text' in df.columns:
    TEXT_COLUMN = 'text'
else:
    TEXT_COLUMN = df.columns[0]  # fallback to first column

if 'Hate' in df.columns:
    TARGET_COLUMN = 'Hate'
elif 'hate' in df.columns:
    TARGET_COLUMN = 'hate'
elif 'label' in df.columns:
    TARGET_COLUMN = 'label'
else:
    TARGET_COLUMN = df.columns[1] if df.shape[1] > 1 else df.columns[-1]  # fallback

print(f"ℹ️ Using text column: '{TEXT_COLUMN}', target column: '{TARGET_COLUMN}'")

# -----------------------------
# Basic cleaning
# -----------------------------
def clean_text(txt):
    if pd.isna(txt):
        return ""
    s = str(txt)
    # keep original script (Bangla/Banglish). Remove URLs/mentions and extra whitespace
    s = re.sub(r'http\S+|www\.\S+|@\w+|#\w+', ' ', s)
    # remove ASCII punctuation but keep Bangla characters and numbers
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

print("🧹 Cleaning text...")
df['clean_text'] = df[TEXT_COLUMN].apply(clean_text)
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"✅ After cleaning: {df.shape}")

# -----------------------------
# Encode labels
# -----------------------------
print("🔢 Encoding labels...")
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df[TARGET_COLUMN])
label_map = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
print("📋 Label mapping:", label_map)

# Save label encoder
joblib.dump(label_encoder, f"{OUTPUT_DIR}/label_encoder.pkl")

# -----------------------------
# Balance classes (oversample minority)
# -----------------------------
print("⚖️ Checking & balancing classes if needed...")
class_counts_before = df['label_enc'].value_counts().to_dict()
print("Before balancing:", class_counts_before)

max_count = max(class_counts_before.values())
balanced_parts = []
for lbl in sorted(df['label_enc'].unique()):
    chunk = df[df['label_enc'] == lbl]
    if len(chunk) < max_count:
        chunk = resample(chunk, replace=True, n_samples=max_count, random_state=42)
    balanced_parts.append(chunk)
df_bal = pd.concat(balanced_parts).reset_index(drop=True)
class_counts_after = df_bal['label_enc'].value_counts().to_dict()
print("After balancing:", class_counts_after)

# -----------------------------
# TF-IDF
# -----------------------------
print("🔤 Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95, lowercase=False, token_pattern=r'[\u0980-\u09FF\w]+')
X = tfidf.fit_transform(df_bal['clean_text'])
y = df_bal['label_enc'].values

# Save TF-IDF
joblib.dump(tfidf, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")

# -----------------------------
# Train-test split
# -----------------------------
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# -----------------------------
# Only SVM active (others commented)
# -----------------------------
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# ... (kept commented for future)

print("\n🚀 Training SVM (linear kernel, probability=True) ...")
svm_clf = SVC(kernel='linear', probability=True, random_state=42)  # linear to get coef_ as feature importance
t0 = time.time()
svm_clf.fit(X_train, y_train)
t1 = time.time()
train_time = t1 - t0
print(f"✅ SVM training time: {train_time:.2f} seconds")

# Save model
joblib.dump(svm_clf, f"{OUTPUT_DIR}/svm_model.pkl")

# -----------------------------
# Evaluation: predictions + metrics
# -----------------------------
print("\n📏 Evaluating on test set...")
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Classification report -> save as CSV & TXT
clf_report_dict = classification_report(y_test, y_pred, output_dict=True)
clf_report_txt = classification_report(y_test, y_pred)
pd.DataFrame(clf_report_dict).transpose().to_csv(f"{OUTPUT_DIR}/classification_report.csv", index=True)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(clf_report_txt)

print("🧾 Classification report saved.")

# Confusion matrix -> PNG + CSV
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv", index=True)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()
print("📉 Confusion matrix saved.")

# ROC & AUC (binary only) + ROC PNG
auc_score = None
avg_precision = None
if len(np.unique(y_test)) == 2:
    y_proba = svm_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.4f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SVM")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve_svm.png", dpi=300)
    plt.close()
    print(f"🔥 ROC AUC: {auc_score:.4f}")

    # Precision recall curve + AP
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"SVM (AP = {avg_precision:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - SVM")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_curve_svm.png", dpi=300)
    plt.close()
    print(f"📈 Average Precision (AP): {avg_precision:.4f}")
else:
    print("ℹ️ Multi-class detected — ROC/PR per-class not generated in this minimal script.")

# -----------------------------
# Feature importance for linear SVM (coef_)
# -----------------------------
try:
    if hasattr(svm_clf, "coef_"):
        coefs = svm_clf.coef_.toarray() if hasattr(svm_clf.coef_, "toarray") else svm_clf.coef_
        # For binary: shape (1, n_features); For multi-class: (n_classes, n_features)
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)
        # We'll compute absolute mean importance across class coefs
        importance = np.mean(np.abs(coefs), axis=0)
        feature_names = tfidf.get_feature_names_out()
        top_n = 30
        top_idx = np.argsort(importance)[::-1][:top_n]
        top_feats = feature_names[top_idx]
        top_vals = importance[top_idx]

        feat_df = pd.DataFrame({'feature': top_feats, 'importance': top_vals})
        feat_df.to_csv(f"{OUTPUT_DIR}/svm_top_features.csv", index=False)

        plt.figure(figsize=(8, max(4, top_n//2)))
        plt.barh(range(len(top_feats))[::-1], top_vals[::-1], align='center')
        plt.yticks(range(len(top_feats)), top_feats[::-1])
        plt.xlabel("Importance (|coef| average)")
        plt.title("Top features - Linear SVM")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/svm_top_features.png", dpi=300)
        plt.close()
        print("🔎 Top features saved.")
    else:
        print("ℹ️ SVM does not expose coef_, skipping feature importance.")
except Exception as e:
    print("⚠️ Error computing feature importance:", str(e))

# -----------------------------
# Learning curve
# -----------------------------
try:
    print("📈 Computing learning curve (may take some time)...")
    train_sizes, train_scores, test_scores = learning_curve(svm_clf, X, y, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1,1.0,5), scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - SVM")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/svm_learning_curve.png", dpi=300)
    plt.close()
    print("📚 Learning curve saved.")
except Exception as e:
    print("⚠️ Learning curve failed:", e)

# -----------------------------
# Cross-validation scores
# -----------------------------
try:
    print("🔁 Running 5-fold cross-validation (accuracy)...")
    cv_scores = cross_val_score(svm_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_summary = {
        'cv_scores': [float(s) for s in cv_scores.tolist()],
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores))
    }
    with open(f"{OUTPUT_DIR}/svm_cv_summary.json", "w", encoding='utf-8') as f:
        json.dump(cv_summary, f, indent=4, ensure_ascii=False)
    print("🔁 Cross-validation summary saved.")
except Exception as e:
    print("⚠️ CV scoring failed:", e)

# -----------------------------
# Sample inference & timing for 3 samples
# -----------------------------
sample_texts = [
    "ami tomake bhalobashi",       # likely non-hate
    "tui ekta kharap lok",        # possibly hate
    "aaj khabar kemon choleche"   # neutral
]

cleaned_samples = [clean_text(s) for s in sample_texts]
sample_vec = tfidf.transform(cleaned_samples)

# measure predict once for batch, and loop per-sample for more stable measure
# predict (batch)
t0 = time.perf_counter()
preds_batch = svm_clf.predict(sample_vec)
t1 = time.perf_counter()
total_predict_batch = t1 - t0
avg_predict_batch = total_predict_batch / len(sample_texts)

# predict loop (per sample)
t0 = time.perf_counter()
for i in range(len(sample_texts)):
    _ = svm_clf.predict(sample_vec[i])
t1 = time.perf_counter()
avg_predict_loop = (t1 - t0) / len(sample_texts)

# predict_proba (batch) if available
total_proba_batch = None
avg_proba_batch = None
if hasattr(svm_clf, 'predict_proba'):
    t0 = time.perf_counter()
    probas = svm_clf.predict_proba(sample_vec)
    t1 = time.perf_counter()
    total_proba_batch = t1 - t0
    avg_proba_batch = total_proba_batch / len(sample_texts)

inference_info = {
    'samples': sample_texts,
    'num_samples': len(sample_texts),
    'total_predict_batch_sec': float(total_predict_batch),
    'avg_predict_batch_sec_per_sample': float(avg_predict_batch),
    'avg_predict_loop_sec_per_sample': float(avg_predict_loop),
    'total_proba_batch_sec': float(total_proba_batch) if total_proba_batch is not None else None,
    'avg_proba_batch_sec_per_sample': float(avg_proba_batch) if avg_proba_batch is not None else None
}

with open(f"{OUTPUT_DIR}/svm_inference_time.json", "w", encoding='utf-8') as f:
    json.dump(inference_info, f, indent=4, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/svm_inference_time.txt", "w", encoding='utf-8') as f:
    f.write(f"Total predict (batch): {total_predict_batch:.6f} s\n")
    f.write(f"Avg predict per sample (batch): {avg_predict_batch:.6f} s\n")
    f.write(f"Avg predict per sample (loop): {avg_predict_loop:.6f} s\n")
    if total_proba_batch is not None:
        f.write(f"Total predict_proba (batch): {total_proba_batch:.6f} s\n")
        f.write(f"Avg predict_proba per sample: {avg_proba_batch:.6f} s\n")

# Save sample predictions
predicted_labels = label_encoder.inverse_transform(preds_batch)
sample_df = pd.DataFrame({
    'text': sample_texts,
    'cleaned': cleaned_samples,
    'predicted_label': predicted_labels,
})
sample_df.to_csv(f"{OUTPUT_DIR}/svm_sample_predictions.csv", index=False)

print("🧪 Sample predictions + inference time saved.")

# -----------------------------
# Dataset info (JSON) - safe keys
# -----------------------------
dataset_info = {
    "original_shape": [int(df.shape[0]), int(df.shape[1])],
    "cleaned_shape": [int(df_bal.shape[0]), int(df_bal.shape[1])],
    "text_column": str(TEXT_COLUMN),
    "target_column": str(TARGET_COLUMN),
    "class_distribution_before": {str(k): int(v) for k, v in class_counts_before.items()},
    "class_distribution_after": {str(k): int(v) for k, v in class_counts_after.items()},
    "num_features_tfidf": int(X.shape[1])
}
with open(f"{OUTPUT_DIR}/dataset_info.json", "w", encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=4, ensure_ascii=False)
print("🗂 Dataset info saved.")

# -----------------------------
# Final summary printed
# -----------------------------
print("\n" + "="*60)
print("🏁 Pipeline complete — SVM-only artifacts saved to:")
print(f"    {OUTPUT_DIR}")
print("="*60)
print(f"Accuracy (test): {acc:.4f}")
if auc_score is not None:
    print(f"ROC AUC: {auc_score:.4f}    Avg Precision (AP): {avg_precision:.4f}")
print("Files saved (examples):")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    print("  -", fname)
