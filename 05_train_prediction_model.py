from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

INPUT_FILE = Path("data_processed/link_prediction_dataset_with_embeddings.csv")
OUT_DIR = Path("data_processed/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

FEATURES = [
    "degree_a",
    "degree_b",
    "weighted_degree_a",
    "weighted_degree_b",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "dprev",
    "embedding_cosine_similarity",
]

print("Loading dataset...", flush=True)

df = pd.read_csv(INPUT_FILE)

print(df.shape)
print(df["label"].value_counts())

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# dprev 999 = no path; behalten, aber als Feature interpretierbar
X = df[FEATURES].copy()
y = df["label"].astype(int)

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y
)

logreg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED
    ))
])

print("\nTraining Logistic Regression...", flush=True)
logreg.fit(X_train, y_train)

p_log = logreg.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print("ROC AUC:", roc_auc_score(y_test, p_log))
print("PR AUC:", average_precision_score(y_test, p_log))

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_SEED
)

print("\nTraining Random Forest...", flush=True)
rf.fit(X_train, y_train)

p_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print("ROC AUC:", roc_auc_score(y_test, p_rf))
print("PR AUC:", average_precision_score(y_test, p_rf))

def precision_at_k(y_true, probs, k):
    order = np.argsort(probs)[::-1][:k]
    return y_true.iloc[order].mean()

print("\n=== Precision@K Random Forest ===")
for k in [10, 50, 100, 500, 1000]:
    if k <= len(y_test):
        print(f"Precision@{k}: {precision_at_k(y_test.reset_index(drop=True), p_rf, k):.4f}")

df_test_out = df_test.copy()
df_test_out["pred_logreg"] = p_log
df_test_out["pred_rf"] = p_rf

df_test_out = df_test_out.sort_values("pred_rf", ascending=False)

out_pred = OUT_DIR / "test_predictions_2023_2024.csv"
df_test_out.to_csv(out_pred, index=False)

print(f"\nSaved test predictions: {out_pred}")

fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

out_fi = OUT_DIR / "random_forest_feature_importance.csv"
fi.to_csv(out_fi, index=False)

print(f"Saved feature importance: {out_fi}")
print(fi)

df_all = df.copy()
df_all["pred_rf_full"] = rf.predict_proba(X)[:, 1]
df_all["pred_logreg_full"] = logreg.predict_proba(X)[:, 1]

out_all = OUT_DIR / "all_link_prediction_scores_2023_2024.csv"
df_all.to_csv(out_all, index=False)

print(f"Saved all scores: {out_all}")
print("\nDone.")
