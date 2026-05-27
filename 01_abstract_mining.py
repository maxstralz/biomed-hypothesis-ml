import requests
from xml.etree import ElementTree
import csv
import time
import os
MAX_RESULTS_PER_MODULE = 50000
SEARCH_BATCH_SIZE = 3000
FETCH_BATCH_SIZE = 200
EMAIL = "your_email@example.com"

OUTPUT_DIR = "data_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_FILTER = '("2020"[Date - Publication] : "2026"[Date - Publication])'

MODULES = {
    "core_surgery": '''
        ("General Surgery"[Mesh] OR "Surgical Procedures, Operative"[Mesh]
        OR "Digestive System Surgical Procedures"[Mesh]
        OR "Minimally Invasive Surgical Procedures"[Mesh]
        OR "Robotic Surgical Procedures"[Mesh]
        OR laparotomy[Title/Abstract]
        OR laparoscopy[Title/Abstract]
        OR "robotic surgery"[Title/Abstract]
        OR "minimally invasive surgery"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "hpb_surgery": '''
        ("Hepatectomy"[Mesh] OR "Pancreatectomy"[Mesh]
        OR "Pancreaticoduodenectomy"[Mesh]
        OR "Cholecystectomy"[Mesh]
        OR "Biliary Tract Surgical Procedures"[Mesh]
        OR hepatectomy[Title/Abstract]
        OR "liver resection"[Title/Abstract]
        OR pancreatectomy[Title/Abstract]
        OR pancreaticoduodenectomy[Title/Abstract]
        OR whipple[Title/Abstract]
        OR "bile duct surgery"[Title/Abstract]
        OR "biliary surgery"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "colorectal_surgery": '''
        ("Colectomy"[Mesh] OR "Proctectomy"[Mesh]
        OR colectomy[Title/Abstract]
        OR proctectomy[Title/Abstract]
        OR "colorectal surgery"[Title/Abstract]
        OR "rectal resection"[Title/Abstract]
        OR "low anterior resection"[Title/Abstract]
        OR "total mesorectal excision"[Title/Abstract]
        OR "right hemicolectomy"[Title/Abstract]
        OR "left hemicolectomy"[Title/Abstract]
        OR "colorectal cancer surgery"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "upper_gi_bariatric": '''
        ("Gastrectomy"[Mesh] OR "Esophagectomy"[Mesh]
        OR "Fundoplication"[Mesh] OR "Bariatric Surgery"[Mesh]
        OR "Gastric Bypass"[Mesh]
        OR gastrectomy[Title/Abstract]
        OR esophagectomy[Title/Abstract]
        OR fundoplication[Title/Abstract]
        OR "bariatric surgery"[Title/Abstract]
        OR "sleeve gastrectomy"[Title/Abstract]
        OR "gastric bypass"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "visceral_oncology": '''
        ("Digestive System Neoplasms"[Mesh]
        OR "Gastrointestinal Neoplasms"[Mesh]
        OR "Colorectal Neoplasms"[Mesh]
        OR "Rectal Neoplasms"[Mesh]
        OR "Pancreatic Neoplasms"[Mesh]
        OR "Liver Neoplasms"[Mesh]
        OR "Cholangiocarcinoma"[Mesh]
        OR "Bile Duct Neoplasms"[Mesh]
        OR "Gallbladder Neoplasms"[Mesh]
        OR "Esophageal Neoplasms"[Mesh]
        OR "Stomach Neoplasms"[Mesh]
        OR "Peritoneal Neoplasms"[Mesh])
        AND
        ("Surgical Procedures, Operative"[Mesh]
        OR surgery[Title/Abstract]
        OR resection[Title/Abstract]
        OR operative[Title/Abstract])
        AND hasabstract[text]
    ''',

    "vascular_surgery": '''
        ("Vascular Surgical Procedures"[Mesh]
        OR "Endovascular Procedures"[Mesh]
        OR "Angioplasty"[Mesh]
        OR "Stents"[Mesh]
        OR "Vascular Grafting"[Mesh]
        OR "Endarterectomy"[Mesh]
        OR "Endarterectomy, Carotid"[Mesh]
        OR "Aortic Aneurysm, Abdominal"[Mesh]
        OR "Peripheral Arterial Disease"[Mesh]
        OR "Peripheral Vascular Diseases"[Mesh]
        OR "Amputation"[Mesh]
        OR "endovascular aneurysm repair"[Title/Abstract]
        OR EVAR[Title/Abstract]
        OR TEVAR[Title/Abstract]
        OR "carotid endarterectomy"[Title/Abstract]
        OR "vascular surgery"[Title/Abstract]
        OR "limb salvage"[Title/Abstract]
        OR "critical limb ischemia"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "thoracic_surgery": '''
        ("Thoracic Surgery"[Mesh]
        OR "Thoracic Surgical Procedures"[Mesh]
        OR "Thoracoscopy"[Mesh]
        OR "Pneumonectomy"[Mesh]
        OR "Lobectomy"[Mesh]
        OR "Pulmonary Surgical Procedures"[Mesh]
        OR "Lung Neoplasms"[Mesh]
        OR "Mediastinal Neoplasms"[Mesh]
        OR "Esophageal Neoplasms"[Mesh]
        OR "video-assisted thoracic surgery"[Title/Abstract]
        OR VATS[Title/Abstract]
        OR thoracoscopy[Title/Abstract]
        OR lobectomy[Title/Abstract]
        OR pneumonectomy[Title/Abstract]
        OR "lung resection"[Title/Abstract]
        OR "thoracic surgery"[Title/Abstract])
        AND hasabstract[text]
    ''',

    "complications_outcomes_prediction": '''
        ("Postoperative Complications"[Mesh]
        OR "Intraoperative Complications"[Mesh]
        OR "Treatment Outcome"[Mesh]
        OR "Risk Assessment"[Mesh]
        OR "Risk Factors"[Mesh]
        OR "Predictive Value of Tests"[Mesh]
        OR "Prognosis"[Mesh]
        OR "Length of Stay"[Mesh]
        OR "Patient Readmission"[Mesh]
        OR "Mortality"[Mesh]
        OR "Morbidity"[Mesh]
        OR "Surgical Wound Infection"[Mesh]
        OR "Anastomosis, Surgical"[Mesh]
        OR "anastomotic leak"[Title/Abstract]
        OR "anastomotic leakage"[Title/Abstract]
        OR "surgical site infection"[Title/Abstract]
        OR SSI[Title/Abstract]
        OR "Clavien-Dindo"[Title/Abstract]
        OR "comprehensive complication index"[Title/Abstract]
        OR "failure to rescue"[Title/Abstract]
        OR "postoperative morbidity"[Title/Abstract]
        OR "risk prediction"[Title/Abstract]
        OR nomogram[Title/Abstract])
        AND
        ("Surgical Procedures, Operative"[Mesh]
        OR surgery[Title/Abstract]
        OR surgical[Title/Abstract]
        OR operation[Title/Abstract]
        OR operative[Title/Abstract])
        AND hasabstract[text]
    ''',

    "perioperative_biology_microbiome": '''
        ("Microbiota"[Mesh]
        OR "Gastrointestinal Microbiome"[Mesh]
        OR "Inflammation"[Mesh]
        OR "C-Reactive Protein"[Mesh]
        OR "Cytokines"[Mesh]
        OR "Interleukins"[Mesh]
        OR "Neutrophils"[Mesh]
        OR "Macrophages"[Mesh]
        OR microbiome[Title/Abstract]
        OR microbiota[Title/Abstract]
        OR inflammation[Title/Abstract]
        OR cytokine[Title/Abstract]
        OR "immune response"[Title/Abstract]
        OR "bile acid"[Title/Abstract]
        OR metabolomics[Title/Abstract])
        AND
        ("Surgical Procedures, Operative"[Mesh]
        OR surgery[Title/Abstract]
        OR surgical[Title/Abstract]
        OR perioperative[Title/Abstract]
        OR postoperative[Title/Abstract])
        AND hasabstract[text]
    '''
}

def clean_query(q):
    q = " ".join(q.split())
    return f"({q}) AND {DATE_FILTER}"

def safe_get_json(url, params, retries=3):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  Warnung: JSON/Request-Fehler Versuch {attempt}/{retries}: {e}")
            time.sleep(2 * attempt)

    raise RuntimeError("PubMed JSON konnte nach mehreren Versuchen nicht geladen werden.")

def search_pubmed_all(query, max_results=None):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    query_clean = clean_query(query)

    params_count = {
        "db": "pubmed",
        "term": query_clean,
        "retmax": 0,
        "retmode": "json",
        "email": EMAIL
    }

    data = safe_get_json(search_url, params_count)
    total_count = int(data["esearchresult"]["count"])

    target_count = total_count if max_results is None else min(total_count, max_results)

    print(f"PubMed Treffer insgesamt: {total_count}")
    print(f"Ziel für Download: {target_count}")

    all_pmids = []

    for start in range(0, target_count, SEARCH_BATCH_SIZE):
        retmax = min(SEARCH_BATCH_SIZE, target_count - start)

        print(f"  Suche PMIDs {start + 1}-{start + retmax} von {target_count}")

        params = {
            "db": "pubmed",
            "term": query_clean,
            "retstart": start,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "email": EMAIL
        }

        data = safe_get_json(search_url, params)
        pmids = data["esearchresult"]["idlist"]
        all_pmids.extend(pmids)

        time.sleep(0.7)

    all_pmids = list(dict.fromkeys(all_pmids))
    print(f"PMIDs tatsächlich gesammelt: {len(all_pmids)}")

    return all_pmids, total_count

def fetch_pubmed_details(pmids):
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    rows = []

    for start in range(0, len(pmids), FETCH_BATCH_SIZE):
        batch = pmids[start:start + FETCH_BATCH_SIZE]

        print(f"  Lade Details {start + 1}-{start + len(batch)} von {len(pmids)}")

        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "email": EMAIL
        }

        try:
            response = requests.get(fetch_url, params=params, timeout=120)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except Exception as e:
            print(f"  Fehler beim Laden dieses Detail-Batches: {e}")
            continue

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")

            abstract_parts = []
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                if abstract_text.text:
                    label = abstract_text.attrib.get("Label")
                    if label:
                        abstract_parts.append(f"{label}: {abstract_text.text}")
                    else:
                        abstract_parts.append(abstract_text.text)

            abstract = " ".join(abstract_parts)
            journal = article.findtext(".//Journal/Title")
            year = article.findtext(".//PubDate/Year")

            if year is None:
                year = article.findtext(".//PubDate/MedlineDate")

            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            publication_types = []
            for pubtype in article.findall(".//PublicationType"):
                if pubtype.text:
                    publication_types.append(pubtype.text)

            doi = None
            for aid in article.findall(".//ArticleId"):
                if aid.attrib.get("IdType") == "doi":
                    doi = aid.text

            if abstract:
                rows.append({
                    "pmid": pmid,
                    "year": year,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "mesh_terms": "; ".join(mesh_terms),
                    "publication_types": "; ".join(publication_types),
                    "doi": doi
                })

        time.sleep(0.7)

    return rows

def save_csv(rows, filename):
    fieldnames = [
        "module",
        "pmid",
        "year",
        "title",
        "abstract",
        "journal",
        "mesh_terms",
        "publication_types",
        "doi"
    ]

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gespeichert: {filepath} ({len(rows)} Abstracts)")

all_rows = []
seen_pmids = set()

for module_name, query in MODULES.items():
    print("\n" + "=" * 70)
    print(f"MODUL: {module_name}")
    print("=" * 70)

    pmids, total_count = search_pubmed_all(
        query=query,
        max_results=MAX_RESULTS_PER_MODULE
    )

    if not pmids:
        print("Keine Treffer.")
        continue

    rows = fetch_pubmed_details(pmids)

    for row in rows:
        row["module"] = module_name

    save_csv(rows, f"{module_name}.csv")

    for row in rows:
        if row["pmid"] not in seen_pmids:
            all_rows.append(row)
            seen_pmids.add(row["pmid"])

    save_csv(all_rows, "all_modules_combined_PROGRESS.csv")

print("\n" + "=" * 70)
print("GESAMTDATEI")
print("=" * 70)

save_csv(all_rows, "all_modules_combined.csv")

print("\nFertig.")
print(f"Einzigartige Abstracts gesamt: {len(all_rows)}")
