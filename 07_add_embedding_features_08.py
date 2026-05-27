from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_DATASET = Path("data_processed/link_prediction_dataset_2020_2022_to_2023_2024.csv")
INPUT_EMBEDDINGS = Path("data_processed/concept_embeddings.csv")
OUTPUT_FILE = Path("data_processed/link_prediction_dataset_with_embeddings.csv")

def parse_embedding(x):
    return np.array([float(v) for v in str(x).split(",")], dtype=np.float32)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

print("Loading dataset...")
df = pd.read_csv(INPUT_DATASET)

print("Loading embeddings...")
emb_df = pd.read_csv(INPUT_EMBEDDINGS)

embeddings = {
    row["concept"]: parse_embedding(row["embedding"])
    for _, row in emb_df.iterrows()
}

print(f"Embeddings loaded: {len(embeddings)}")

sims = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    a = row["concept_a"]
    b = row["concept_b"]

    if a in embeddings and b in embeddings:
        sims.append(cosine_similarity(embeddings[a], embeddings[b]))
    else:
        sims.append(np.nan)

df["embedding_cosine_similarity"] = sims
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(df["embedding_cosine_similarity"].describe())
