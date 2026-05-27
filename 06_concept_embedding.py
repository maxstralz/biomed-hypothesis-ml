from pathlib import Path
import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

INPUT_FILE = Path("data_processed/concept_nodes_summary.csv")
OUTPUT_FILE = Path("data_processed/concept_embeddings.csv")

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
SLEEP = 0.2

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

nodes = pd.read_csv(INPUT_FILE)
concepts = nodes["concept"].dropna().astype(str).str.lower().str.strip().unique().tolist()

if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    done = set(existing["concept"].astype(str))
    results = existing.to_dict("records")
else:
    done = set()
    results = []

concepts_todo = [c for c in concepts if c not in done]

print(f"Total concepts: {len(concepts)}")
print(f"Already done: {len(done)}")
print(f"Remaining: {len(concepts_todo)}")

for i in tqdm(range(0, len(concepts_todo), BATCH_SIZE)):
    batch = concepts_todo[i:i+BATCH_SIZE]

    response = client.embeddings.create(
        model=MODEL,
        input=batch
    )

    for concept, emb in zip(batch, response.data):
        results.append({
            "concept": concept,
            "embedding": ",".join(map(str, emb.embedding))
        })

    if (i // BATCH_SIZE + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    time.sleep(SLEEP)

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")
