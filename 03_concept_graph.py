from pathlib import Path
from itertools import combinations
import pandas as pd
from collections import Counter, defaultdict

INPUT_FILE = Path("data_processed/concepts_2_per_abstract.csv")

OUTPUT_DIR = Path("data_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = OUTPUT_DIR / "concept_edges_by_abstract.csv"
EDGE_SUMMARY_FILE = OUTPUT_DIR / "concept_edges_summary.csv"
NODE_FILE = OUTPUT_DIR / "concept_nodes_summary.csv"

MIN_CONCEPT_OCCURRENCE = 3


print("Loading concepts...", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str)

required_cols = ["pmid", "year", "query_name", "title", "journal", "concept"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["pmid", "year", "concept"]).copy()
df["pmid"] = df["pmid"].astype(str)
df["concept"] = df["concept"].astype(str).str.lower().str.strip()
df["concept"] = df["concept"].str.replace(r"\s+", " ", regex=True)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

df = df[(df["year"] >= 2020) & (df["year"] <= 2026)].copy()

print(f"Concept rows loaded: {len(df)}", flush=True)
print(f"Unique PMIDs: {df['pmid'].nunique()}", flush=True)
print(f"Unique concepts before filtering: {df['concept'].nunique()}", flush=True)


concept_counts = df["concept"].value_counts()

keep_concepts = set(
    concept_counts[concept_counts >= MIN_CONCEPT_OCCURRENCE].index
)

df = df[df["concept"].isin(keep_concepts)].copy()

print(f"Unique concepts after occurrence filter ≥{MIN_CONCEPT_OCCURRENCE}: {df['concept'].nunique()}", flush=True)
print(f"Concept rows after filtering: {len(df)}", flush=True)



node_summary = (
    df.groupby("concept")
    .agg(
        occurrence=("pmid", "count"),
        n_pmids=("pmid", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    .reset_index()
    .sort_values(["occurrence", "concept"], ascending=[False, True])
)

node_summary.to_csv(NODE_FILE, index=False)
print(f"Saved node summary: {NODE_FILE}", flush=True)


print("Building abstract-level edges...", flush=True)

edge_rows = []

group_cols = ["pmid", "year", "query_name", "title", "journal"]

for keys, g in df.groupby(group_cols, dropna=False):
    pmid, year, query_name, title, journal = keys

    concepts = sorted(set(g["concept"].dropna().astype(str)))

    if len(concepts) < 2:
        continue

    for a, b in combinations(concepts, 2):
        if a == b:
            continue

        edge_rows.append({
            "concept_a": a,
            "concept_b": b,
            "pmid": pmid,
            "year": int(year),
            "query_name": query_name,
            "title": title,
            "journal": journal,
        })

edges = pd.DataFrame(edge_rows)

if edges.empty:
    raise RuntimeError("No edges generated. Check whether each PMID has at least 2 retained concepts.")

edges.to_csv(EDGE_FILE, index=False)
print(f"Saved abstract-level edges: {EDGE_FILE}", flush=True)
print(f"Edges by abstract: {len(edges)}", flush=True)


print("Building edge summary...", flush=True)

edge_summary = (
    edges.groupby(["concept_a", "concept_b"])
    .agg(
        weight=("pmid", "count"),
        n_pmids=("pmid", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        pmids=("pmid", lambda x: "; ".join(sorted(set(map(str, x)))[:20])),
        example_titles=("title", lambda x: " || ".join(list(dict.fromkeys(map(str, x)))[:3])),
    )
    .reset_index()
)

edge_summary = edge_summary.sort_values(
    ["weight", "first_year", "concept_a", "concept_b"],
    ascending=[False, True, True, True]
)

edge_summary.to_csv(EDGE_SUMMARY_FILE, index=False)

print(f"Saved edge summary: {EDGE_SUMMARY_FILE}", flush=True)
print(f"Unique concept edges: {len(edge_summary)}", flush=True)


print("\nTemporal edge summary:")
for y in range(2020, 2027):
    n_edges_year = edges[edges["year"] == y][["concept_a", "concept_b"]].drop_duplicates().shape[0]
    print(f"{y}: {n_edges_year} unique edges")

print("\nDone.", flush=True)
