from pathlib import Path
import pandas as pd
import re

INPUT = Path("data_processed/future_predictions/future_link_predictions_2025_2026_style.csv")
OUTPUT = Path("data_processed/future_predictions/publishable_hypotheses_filtered.csv")

df = pd.read_csv(INPUT)

print("Initial:", len(df))


GENERIC_PHRASES = [
    "postoperative complications",
    "postoperative complication",
    "complications reduction",
    "complication reduction",
    "complications prediction",
    "complication prediction",
    "complication rates",
    "postoperative morbidity",
    "surgical outcomes",
    "surgery outcomes",
    "perioperative outcomes",
    "short-term outcomes",
    "long-term outcomes",
    "oncological outcomes",
    "outcomes comparison",
    "outcome comparison",
    "therapy impact",
    "treatment impact",
    "risk factors",
    "risk factor",
    "success rate",
    "survival rate",
    "overall survival",
    "disease-free survival",
    "recurrence-free survival",
    "local recurrence rate",
    "learning curve analysis",
    "patient-reported outcomes",
    "health-related quality of life",
    "major adverse cardiac events",
    "major adverse cardiovascular events",
]

OVERLY_BROAD_TERMS = [
    "minimally invasive surgery",
    "laparoscopic surgery",
    "robot-assisted surgery",
    "video-assisted thoracic surgery",
    "endovascular treatment",
    "endovascular therapy",
    "neoadjuvant therapy",
    "neoadjuvant treatment",
    "adjuvant chemotherapy",
    "preoperative chemoradiotherapy",
    "postoperative chemotherapy",
]

BAD_SINGLE_TOKENS = [
    "comparison",
    "impact",
    "prediction",
    "outcomes",
    "outcome",
    "rates",
    "rate",
    "reduction",
    "incidence",
    "prognosis",
    "morbidity",
    "mortality",
]


def normalize_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x

def contains_generic_phrase(x):
    x = normalize_text(x)
    return any(p in x for p in GENERIC_PHRASES)

def is_overly_broad_exact_or_near(x):
    x_norm = normalize_text(x)
    for term in OVERLY_BROAD_TERMS:
        t = normalize_text(term)
        if x_norm == t:
            return True
    return False

def has_bad_ending(x):
    x_norm = normalize_text(x)
    return any(x_norm.endswith(" " + tok) or x_norm == tok for tok in BAD_SINGLE_TOKENS)

def token_overlap_too_high(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return True

    overlap = len(a_tokens & b_tokens)
    min_len = min(len(a_tokens), len(b_tokens))

    # entfernt Synonyme / fast gleiche Begriffe
    return overlap / min_len >= 0.60

def same_core_phrase(a, b):
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    # einer ist fast im anderen enthalten
    if a_norm in b_norm or b_norm in a_norm:
        return True

    return False


df["a_norm"] = df["concept_a"].apply(normalize_text)
df["b_norm"] = df["concept_b"].apply(normalize_text)

# 1) Generische Outcome-/Complication-Phrasen entfernen
df = df[
    ~df["concept_a"].apply(contains_generic_phrase) &
    ~df["concept_b"].apply(contains_generic_phrase)
]
print("After generic phrase filter:", len(df))

# 2) Zu breite Konzepte entfernen
df = df[
    ~df["concept_a"].apply(is_overly_broad_exact_or_near) &
    ~df["concept_b"].apply(is_overly_broad_exact_or_near)
]
print("After broad term filter:", len(df))

# 3) Schlechte Endungen entfernen
df = df[
    ~df["concept_a"].apply(has_bad_ending) &
    ~df["concept_b"].apply(has_bad_ending)
]
print("After bad ending filter:", len(df))
df = df[
    ~df.apply(lambda r: token_overlap_too_high(r["concept_a"], r["concept_b"]), axis=1)
]
print("After token overlap filter:", len(df))

df = df[
    ~df.apply(lambda r: same_core_phrase(r["concept_a"], r["concept_b"]), axis=1)
]
print("After core phrase filter:", len(df))


df = df[df["embedding_cosine_similarity"].between(0.20, 0.60)]
print("After semantic window filter:", len(df))


df = df.sort_values("pred_mean", ascending=False)

df = df.drop(columns=["a_norm", "b_norm"], errors="ignore")

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(df[["concept_a", "concept_b", "pred_mean", "embedding_cosine_similarity", "dprev"]].head(50	))
