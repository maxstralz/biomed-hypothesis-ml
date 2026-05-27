from pathlib import Path
import random
import pandas as pd
import networkx as nx
from tqdm import tqdm

EDGE_FILE = Path("data_processed/concept_edges_summary.csv")
NODE_FILE = Path("data_processed/concept_nodes_summary.csv")

OUTPUT_FILE = Path("data_processed/link_prediction_dataset_2020_2022_to_2023_2024.csv")

TRAIN_END_YEAR = 2022
LABEL_START_YEAR = 2023
LABEL_END_YEAR = 2024

NEGATIVE_PER_POSITIVE = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

edges = pd.read_csv(EDGE_FILE, dtype=str)
nodes = pd.read_csv(NODE_FILE, dtype=str)

for col in ["weight", "first_year", "last_year"]:
    edges[col] = pd.to_numeric(edges[col], errors="coerce")

for col in ["occurrence", "n_pmids", "first_year", "last_year"]:
    nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

# canonical pair
edges["pair"] = edges.apply(
    lambda r: tuple(sorted([r["concept_a"], r["concept_b"]])),
    axis=1
)

train_edges = edges[edges["first_year"] <= TRAIN_END_YEAR].copy()

G = nx.Graph()

for _, row in nodes.iterrows():
    G.add_node(
        row["concept"],
        occurrence=int(row["occurrence"]),
        first_year=int(row["first_year"]),
        last_year=int(row["last_year"]),
    )

for _, row in train_edges.iterrows():
    G.add_edge(
        row["concept_a"],
        row["concept_b"],
        weight=int(row["weight"]),
        first_year=int(row["first_year"]),
    )

print(f"Train graph nodes: {G.number_of_nodes()}")
print(f"Train graph edges: {G.number_of_edges()}")

all_nodes = list(G.nodes())
train_pairs = set(tuple(sorted(e)) for e in G.edges())


future_edges = edges[
    (edges["first_year"] >= LABEL_START_YEAR) &
    (edges["first_year"] <= LABEL_END_YEAR)
].copy()

positive_pairs = set(future_edges["pair"])
positive_pairs = positive_pairs - train_pairs

print(f"Positive new edges {LABEL_START_YEAR}-{LABEL_END_YEAR}: {len(positive_pairs)}")


target_negatives = len(positive_pairs) * NEGATIVE_PER_POSITIVE
negative_pairs = set()

print(f"Sampling negatives: {target_negatives}")

while len(negative_pairs) < target_negatives:
    a, b = random.sample(all_nodes, 2)
    pair = tuple(sorted([a, b]))

    if pair in train_pairs:
        continue

    if pair in positive_pairs:
        continue

    negative_pairs.add(pair)

print(f"Negative pairs: {len(negative_pairs)}")


degree = dict(G.degree())
weighted_degree = dict(G.degree(weight="weight"))

def safe_shortest_path_length(G, a, b, cutoff=6):
    try:
        return nx.shortest_path_length(G, a, b)
    except nx.NetworkXNoPath:
        return 999

def pair_features(pair, label):
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
            import math
            adamic_adar += 1 / math.log(dz)

    preferential_attachment = degree.get(a, 0) * degree.get(b, 0)

    dprev = safe_shortest_path_length(G, a, b)

    return {
        "concept_a": a,
        "concept_b": b,
        "label": label,
        "degree_a": degree.get(a, 0),
        "degree_b": degree.get(b, 0),
        "weighted_degree_a": weighted_degree.get(a, 0),
        "weighted_degree_b": weighted_degree.get(b, 0),
        "common_neighbors": common_neighbors,
        "jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "preferential_attachment": preferential_attachment,
        "dprev": dprev,
    }

rows = []

print("Building positive rows...")
for pair in tqdm(positive_pairs):
    rows.append(pair_features(pair, 1))

print("Building negative rows...")
for pair in tqdm(negative_pairs):
    rows.append(pair_features(pair, 0))

dataset = pd.DataFrame(rows)

dataset.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(dataset["label"].value_counts())
print(dataset.head())
