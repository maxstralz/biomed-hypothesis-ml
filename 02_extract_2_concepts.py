from pathlib import Path
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


INPUT_FILE = Path("data_raw/all_combined_deduplicated.csv")
OUTPUT_FILE = Path("data_processed/concepts_2_per_abstract.csv")

MODEL = "gpt-4o-mini"

BATCH_SIZE = 5                 
SAVE_EVERY_BATCHES = 5         
SLEEP_BETWEEN_BATCHES = 0.3
TIMEOUT_SECONDS = 60

TEST_MODE = True               
TEST_N = 20                    


print("SCRIPT STARTED", flush=True)

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY FOUND:", api_key is not None, flush=True)

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. In PowerShell set it with:\n"
        '$env:OPENAI_API_KEY="your_new_key"'
    )

client = OpenAI(api_key=api_key)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


print(f"Loading input file: {INPUT_FILE}", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str)

print(f"Loaded file rows: {len(df)}", flush=True)
print("Columns:", df.columns.tolist(), flush=True)

required_cols = ["pmid", "year", "title", "abstract", "query_name"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["pmid", "year", "title", "abstract"])
df["pmid"] = df["pmid"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

df = df[(df["year"] >= 2020) & (df["year"] <= 2026)].copy()

print(f"Abstracts 2020-2026: {len(df)}", flush=True)

if TEST_MODE:
    df = df.head(TEST_N).copy()
    print(f"TEST MODE ACTIVE: only first {len(df)} abstracts", flush=True)


if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE, dtype=str)

    if "pmid" in existing.columns:
        done_pmids = set(existing["pmid"].astype(str).unique())
        results = existing.to_dict("records")
        print(f"Resuming. Already processed PMIDs: {len(done_pmids)}", flush=True)
    else:
        done_pmids = set()
        results = []
else:
    done_pmids = set()
    
results = []

df_todo = df[~df["pmid"].astype(str).isin(done_pmids)].copy()

print(f"Remaining abstracts: {len(df_todo)}", flush=True)


def extract_batch(batch_df: pd.DataFrame):
    items = []

    for _, row in batch_df.iterrows():
        items.append({
            "pmid": str(row["pmid"]),
            "title": str(row["title"])[:1000],
            "abstract": str(row["abstract"])[:5000],
        })

    prompt = f"""
You are extracting scientific concepts from surgical and biomedical abstracts.

For each abstract, extract EXACTLY the 2 most novel or high-value concepts.

Rules:
- Each concept must be 2 to 6 words.
- Use lowercase.
- Prefer specific biomedical, surgical, mechanistic, biomarker, complication, technique, diagnostic, perioperative, or outcome concepts.
- Prefer concepts that could become nodes in a biomedical hypothesis graph.
- Prefer highly specific descriptions of diagnostic or surgical approaches.
- Prefer concepts described as novel, emerging, new, innovative, or underexplored.
- Prefer mechanistic or clinically testable concepts over broad disease labels.
- Avoid entire sentences.
- Normalize singular/plural.
- Avoid abbreviations unless the abbreviation is standard and meaningful.
- Avoid generic or commonly repeated concepts such as:
  patient, patients, study, treatment, therapy, surgery, cancer, disease, outcome,
  risk factor, complication, mortality, morbidity, survival, overall survival,
  colorectal surgery, abdominal surgery, abdominal pain, adverse effects,
  analgesic, antimicrobial, postoperative complication.

Examples:

Abstract:
Postoperative pancreatic fistula remains a major complication after pancreaticoduodenectomy. Drain amylase levels and gland texture are key predictors.
Output:
["postoperative pancreatic fistula", "drain amylase levels"]

Abstract:
Alterations in gut microbiota are associated with chronic pouchitis. Fecal microbiota transplantation shows therapeutic potential.
Output:
["chronic pouchitis microbiota", "fecal microbiota transplantation"]

Abstract:
Intraoperative fluorescence imaging improves perfusion assessment and may reduce anastomotic leakage risk in colorectal surgery.
Output:
["intraoperative perfusion imaging", "anastomotic leakage risk"]

Return valid JSON only.

Return exactly this JSON structure:
{{
  "items": [
    {{
      "pmid": "123",
      "concepts": ["concept one", "concept two"]
    }}
  ]
}}

Abstracts:
{json.dumps(items, ensure_ascii=False)}
"""

    print(f"Calling OpenAI for {len(items)} abstracts...", flush=True)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        
response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract normalized scientific concepts and return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("JSON parse failed. Raw output:", flush=True)
        print(text[:2000], flush=True)
        return []

    return parsed.get("items", [])


batches = [
    df_todo.iloc[i:i + BATCH_SIZE]
    for i in range(0, len(df_todo), BATCH_SIZE)
]

print(f"Total batches: {len(batches)}", flush=True)

for batch_idx, batch_df in enumerate(tqdm(batches, desc="Extracting concepts")):
    try:
        extracted = extract_batch(batch_df)

        meta_cols = ["year", "query_name", "title"]
        if "journal" in batch_df.columns:
            meta_cols.append("journal")

        meta = batch_df.set_index("pmid")[meta_cols].to_dict("index")

        for item in extracted:
            pmid = str(item.get("pmid", "")).strip()
            concepts = item.get("concepts", [])

            if pmid not in meta:
                continue

            if not isinstance(concepts, list):
                continue

            for concept in concepts[:2]:
                concept = str(concept).strip().lower()
                concept = " ".join(concept.split())

                if len(concept.split()) < 2:
                    continue

                if len(concept.split()) > 8:
                    continue

                results.append({
                    "pmid": pmid,
                    "year": meta[pmid].get("year", ""),
                    "query_name": meta[pmid].get("query_name", ""),
                    "title": meta[pmid].get("title", ""),
                    "journal": meta[pmid].get("journal", ""),
                    "concept": concept,
                })

        print(
            f"Batch {batch_idx + 1}/{len(batches)} done. Total concept rows: {len(results)}",
            flush=True
        )

    except Exception as e:
        print(f"Error in batch {batch_idx + 1}: {repr(e)}", flush=True)

    if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
        pd.DataFrame(results).drop_duplicates().to_csv(OUTPUT_FILE, index=False)
        print(f"Saved checkpoint: {OUTPUT_FILE} | rows: {len(results)}", flush=True)

    time.sleep(SLEEP_BETWEEN_BATCHES)

pd.DataFrame(results).drop_duplicates().to_csv(OUTPUT_FILE, index=False)

print(f"Done. Saved: {OUTPUT_FILE}", flush=True)