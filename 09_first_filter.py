from pathlib import Path
import pandas as pd

INPUT = Path("data_processed/future_predictions/future_link_predictions_2025_2026_style.csv")
OUTPUT = Path("data_processed/future_predictions/publishable_hypotheses.csv")

df = pd.read_csv(INPUT)

print("Initial:", len(df))

TRIVIAL = [
    "surgery",
    "minimally invasive surgery",
    "laparoscopic surgery",
    "postoperative complications",
    "outcomes",
    "comparison",
    "analysis",
    "therapy",
    "treatment"
]

def is_trivial(x):
    return any(t in x.lower() for t in TRIVIAL)

df = df[
    (~df["concept_a"].apply(is_trivial)) &
    (~df["concept_b"].apply(is_trivial))
]

print("After trivial filter:", len(df))


def similar_string(a, b):
    return len(set(a.split()) & set(b.split())) > 1

df = df[~df.apply(lambda r: similar_string(r["concept_a"], r["concept_b"]), axis=1)]

print("After synonym filter:", len(df))


df = df[df["dprev"] >= 3]

print("After distance filter:", len(df))


df = df[
    (df["embedding_cosine_similarity"] >= 0.25) &
    (df["embedding_cosine_similarity"] <= 0.6)
]

print("After semantic filter:", len(df))

df = df[df["pred_mean"] > 0.7]

print("After score filter:", len(df))


df = df.sort_values("pred_mean", ascending=False).head(200)

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(df[["concept_a", "concept_b", "pred_mean"]].head(20))
