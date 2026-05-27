from pathlib import Path
import random
import math
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

EDGE_FILE = Path("data_processed/concept_edges_summary.csv")
NODE_FILE = Path("data_processed/concept_nodes_summary.csv")
EMBEDDING_FILE = Path("data_processed/concept_embeddings.csv")
TRAIN_DATASET = Path("data_processed/link_prediction_dataset_with_embeddings.csv")

OUT_DIR = Path("data_processed/future_predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "future_link_predictions_2025_2026_style.csv"

TRAIN_GRAPH_END_YEAR = 2024

N_CANDIDATES = 300000
TOP_OUTPUT = 5000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def parse_embedding(x):
    return np.array([float(v) for v in str(x).split(",")], dtype=np.float32)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def shortest_path_or_999(G, a, b):
    try:
        return nx.shortest_path_length(G, a, b)
    except nx.NetworkXNoPath:
        return 999

def pair_features(G, degree, weighted_degree, embeddings, pair):
    a, b = pair

    neigh_a = set(G.neighbors(a)) if a in G else set()
    neigh_b = set(G.neighbors(b)) if b in G else set()

    common = neigh_a & neigh_b
    union = neigh_a | neigh_b

    common_neighbors = len(common)
    jaccard = common_neighbors / len(union) if len(union) > 0 else 0

    adamic_adar = 0
    for z in common:
        dz = degree.get(z, 0)
        if dz > 1:
            adamic_adar += 1 / math.log(dz)

    preferential_attachment = degree.get(a, 0) * degree.get(b, 0)
    dprev = shortest_path_or_999(G, a, b)

    if a in embeddings and b in embeddings:
        emb_sim = cosine_similarity(embeddings[a], embeddings[b])
    else:
        emb_sim = np.nan

    return {
        "concept_a": a,
        "concept_b": b,
        "degree_a": degree.get(a, 0),
        "degree_b": degree.get(b, 0),
        "weighted_degree_a": weighted_degree.get(a, 0),
        "weighted_degree_b": weighted_degree.get(b, 0),
        "common_neighbors": common_neighbors,
        "jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "preferential_attachment": preferential_attachment,
        "dprev": dprev,
        "embedding_cosine_similarity": emb_sim,
    }

print("Loading training dataset...", flush=True)

train_df = pd.read_csv(TRAIN_DATASET)

for col in FEATURES:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

X = train_df[FEATURES]
y = train_df["label"].astype(int)

print(train_df.shape)
print(train_df["label"].value_counts())

print("Training logistic regression...", flush=True)

logreg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED
    ))
])

logreg.fit(X, y)

print("Training random forest...", flush=True)

rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_SEED
)

rf.fit(X, y)

print("Loading graph data...", flush=True)

edges = pd.read_csv(EDGE_FILE, dtype=str)
nodes = pd.read_csv(NODE_FILE, dtype=str)

for col in ["weight", "first_year", "last_year"]:
    edges[col] = pd.to_numeric(edges[col], errors="coerce")

for col in ["occurrence", "n_pmids", "first_year", "last_year"]:
    nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

graph_edges = edges[edges["first_year"] <= TRAIN_GRAPH_END_YEAR].copy()

G = nx.Graph()

for _, row in nodes.iterrows():
    G.add_node(
        row["concept"],
        occurrence=int(row["occurrence"]),
        first_year=int(row["first_year"]),
        last_year=int(row["last_year"]),
    )

for _, row in graph_edges.iterrows():
    G.add_edge(
        row["concept_a"],
        row["concept_b"],
        weight=int(row["weight"]),
        first_year=int(row["first_year"]),
    )

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges ≤ {TRAIN_GRAPH_END_YEAR}: {G.number_of_edges()}")

all_nodes = list(G.nodes())
existing_pairs = set(tuple(sorted(e)) for e in G.edges())

degree = dict(G.degree())
weighted_degree = dict(G.degree(weight="weight"))

print("Loading embeddings...", flush=True)

emb_df = pd.read_csv(EMBEDDING_FILE)
embeddings = {
    row["concept"]: parse_embedding(row["embedding"])
    for _, row in emb_df.iterrows()
}

print(f"Embeddings: {len(embeddings)}")

print(f"Sampling {N_CANDIDATES} candidate non-edges...", flush=True)

candidate_pairs = set()

while len(candidate_pairs) < N_CANDIDATES:
    a, b = random.sample(all_nodes, 2)
    pair = tuple(sorted([a, b]))

    if pair in existing_pairs:
        continue

    candidate_pairs.add(pair)

print(f"Candidate pairs: {len(candidate_pairs)}")

rows = []

print("Building candidate features...", flush=True)

for pair in tqdm(candidate_pairs):
    rows.append(pair_features(G, degree, weighted_degree, embeddings, pair))

cand_df = pd.DataFrame(rows)

for col in FEATURES:
    cand_df[col] = pd.to_numeric(cand_df[col], errors="coerce")

print("Predicting...", flush=True)

cand_df["pred_logreg"] = logreg.predict_proba(cand_df[FEATURES])[:, 1]
cand_df["pred_rf"] = rf.predict_proba(cand_df[FEATURES])[:, 1]

cand_df["pred_mean"] = (cand_df["pred_logreg"] + cand_df["pred_rf"]) / 2

cand_df["semantic_band"] = pd.cut(
    cand_df["embedding_cosine_similarity"],
    bins=[-1, 0.20, 0.35, 0.50, 1.0],
    labels=["low", "medium", "high", "very_high"]
)

cand_df = cand_df.sort_values("pred_mean", ascending=False).head(TOP_OUTPUT)

cand_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(cand_df.head(25)[[
    "concept_a",
    "concept_b",
    "pred_mean",
    "pred_logreg",
    "pred_rf",
    "embedding_cosine_similarity",
    "dprev",
    "semantic_band"
]])