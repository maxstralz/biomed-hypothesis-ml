from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data_processed/future_predictions/publishable_hypotheses_filtered.csv")

OUTPUT_FULL = Path("data_processed/final_novelty_ranking_full.csv")
OUTPUT_CLEAN = Path("data_processed/final_novelty_ranking_clean.csv")

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())


required = [
    "concept_a",
    "concept_b",
    "embedding_cosine_similarity",
    "jaccard",
    "common_neighbors",
    "dprev"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["novelty_score"] = 1 - df["embedding_cosine_similarity"]

df["redundancy_score"] = (
    df["jaccard"].fillna(0)
    + df["common_neighbors"].fillna(0) * 0.1
)

df["time_boost"] = df["dprev"].apply(
    lambda x: 0.3 if x >= 4 else (0.15 if x == 3 else 0)
)

df["final_score"] = (
    df["novelty_score"]
    - df["redundancy_score"]
    + df["time_boost"]
)

df = df.sort_values(by="final_score", ascending=False)

df.to_csv(
    OUTPUT_FULL,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"Saved FULL ranking: {OUTPUT_FULL}")


df_clean = df[[
    "concept_a",
    "concept_b",
    "final_score",
    "novelty_score",
    "redundancy_score",
    "embedding_cosine_similarity",
    "dprev"
]].copy()

df_clean.to_csv(
    OUTPUT_CLEAN,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"Saved CLEAN ranking: {OUTPUT_CLEAN}")


print("\nTOP 20 NOVEL IDEAS:\n")
print(df_clean.head(20))
