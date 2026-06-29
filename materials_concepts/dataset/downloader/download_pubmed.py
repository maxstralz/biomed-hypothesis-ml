"""Download and normalize a surgical PubMed corpus in one resumable command.

The module queries are intentionally kept beside the retrieval logic so a run
has one public entry point.  The command writes the project's common works-table
schema directly; no intermediate PubMed-to-OpenAlex adapter is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import click
import pandas as pd
import requests


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SUMMARY_BATCH_SIZE = 500
FETCH_BATCH_SIZE = 200
# PubMed's ESearch results are limited to the first 10,000 matches. Keep every
# History-server query within that boundary by splitting broad searches by date.
MAX_PUBMED_HISTORY_RESULTS = 10_000
DEFAULT_OUTPUT = "data/table/surgery.pubmed.works.csv"
# Match the supplied 01_abstract_mining.py script.  The core-surgery search
# alone spans millions of records, which is not a practical default for an
# abstract download followed by LLM extraction.
DEFAULT_MAX_RESULTS_PER_MODULE = 120000
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025
MIN_ABSTRACT_CHARS = 300
MAX_ABSTRACT_CHARS = 4_000
MIN_TITLE_CHARS = 10

EXCLUDED_PUBLICATION_TYPES = {
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Editorial",
    "Letter",
    "Comment",
    "Case Reports",
    "Retracted Publication",
}
_EXCLUDED_PUBLICATION_TYPES_NORMALIZED = {
    publication_type.casefold() for publication_type in EXCLUDED_PUBLICATION_TYPES
}
_ENGLISH_LANGUAGE_VALUES = {"en", "eng", "english"}
EXCLUSION_REASONS = (
    "missing_pmid",
    "missing_title",
    "missing_abstract",
    "missing_publication_date",
    "title_too_short",
    "abstract_too_short",
    "abstract_too_long",
    "non_english",
    "excluded_publication_type",
    "unparseable_xml",
)

# Applied once to every module query. Keep the eligibility criteria centralized
# so each module only expresses its subject-matter search terms.
GLOBAL_FILTER = '''
    AND hasabstract[text]
    AND english[lang]
    AND humans[Mesh]
    NOT (review[Publication Type]
         OR systematic review[Publication Type]
         OR meta-analysis[Publication Type]
         OR editorial[Publication Type]
         OR letter[Publication Type]
         OR comment[Publication Type]
         OR case reports[Publication Type])
'''


# These queries are based on the supplied 01_abstract_mining.py workflow.
MODULES = {
    "core_surgery": '''
        ("General Surgery"[Mesh]
        OR "Digestive System Surgical Procedures"[Mesh]
        OR laparotomy[Title/Abstract]
        OR laparoscopy[Title/Abstract]
        OR "robotic surgery"[Title/Abstract]
        OR "minimally invasive surgery"[Title/Abstract])
        AND
        (
            abdominal[Title/Abstract]
            OR gastrointestinal[Title/Abstract]
            OR digestive[Title/Abstract]
            OR colorectal[Title/Abstract]
            OR hepatobiliary[Title/Abstract]
            OR pancreatic[Title/Abstract]
            OR gastric[Title/Abstract]
            OR esophageal[Title/Abstract]
            OR bowel[Title/Abstract]
            OR intestinal[Title/Abstract]
            OR visceral[Title/Abstract]
        )
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
    ''',
    "vascular_surgery": '''
    (
        "Vascular Surgical Procedures"[Mesh]
        OR "Endovascular Procedures"[Mesh]
        OR "Angioplasty"[Mesh]
        OR "Vascular Grafting"[Mesh]
        OR "Endarterectomy"[Mesh]
        OR "Endarterectomy, Carotid"[Mesh]
        OR "endovascular aneurysm repair"[Title/Abstract]
        OR EVAR[Title/Abstract]
        OR TEVAR[Title/Abstract]
        OR "carotid endarterectomy"[Title/Abstract]
        OR "vascular surgery"[Title/Abstract]
        OR "limb salvage"[Title/Abstract]
        OR "critical limb ischemia"[Title/Abstract]
    )
    AND
    (
        surgery[Title/Abstract]
        OR surgical[Title/Abstract]
        OR endovascular[Title/Abstract]
        OR repair[Title/Abstract]
        OR revascularization[Title/Abstract]
        OR bypass[Title/Abstract]
        OR graft[Title/Abstract]
        OR stent[Title/Abstract]
        OR angioplasty[Title/Abstract]
        OR endarterectomy[Title/Abstract]
        OR amputation[Title/Abstract]
        )
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
        AND
        (
            surgery[Title/Abstract]
            OR surgical[Title/Abstract]
            OR resection[Title/Abstract]
            OR operative[Title/Abstract]
            OR thoracoscopic[Title/Abstract]
            OR VATS[Title/Abstract]
            OR lobectomy[Title/Abstract]
            OR segmentectomy[Title/Abstract]
            OR pneumonectomy[Title/Abstract]
        )
    ''',
    "complications_outcomes_prediction": '''
        (
            "anastomotic leak"[Title/Abstract]
            OR "anastomotic leakage"[Title/Abstract]
            OR "surgical site infection"[Title/Abstract]
            OR "wound infection"[Title/Abstract]
            OR "postoperative pancreatic fistula"[Title/Abstract]
            OR "pancreatic fistula"[Title/Abstract]
            OR "bile leak"[Title/Abstract]
            OR "biliary leak"[Title/Abstract]
            OR "post-hepatectomy liver failure"[Title/Abstract]
            OR "posthepatectomy liver failure"[Title/Abstract]
            OR "delayed gastric emptying"[Title/Abstract]
            OR "ileus"[Title/Abstract]
            OR "postoperative ileus"[Title/Abstract]
            OR "intraoperative bleeding"[Title/Abstract]
            OR "postoperative hemorrhage"[Title/Abstract]
            OR "pulmonary complication"[Title/Abstract]
            OR "venous thromboembolism"[Title/Abstract]
            OR "deep vein thrombosis"[Title/Abstract]
            OR "pulmonary embolism"[Title/Abstract]
            OR "Clavien-Dindo"[Title/Abstract]
            OR "comprehensive complication index"[Title/Abstract]
            OR "failure to rescue"[Title/Abstract]
            OR "unplanned readmission"[Title/Abstract]
            OR "30-day readmission"[Title/Abstract]
            OR "length of stay"[Title/Abstract]
        )
        AND
        (
            surgery[Title/Abstract]
            OR surgical[Title/Abstract]
            OR postoperative[Title/Abstract]
            OR perioperative[Title/Abstract]
            OR operation[Title/Abstract]
            OR operative[Title/Abstract]
            OR resection[Title/Abstract]
            OR colectomy[Title/Abstract]
            OR proctectomy[Title/Abstract]
            OR hepatectomy[Title/Abstract]
            OR pancreatectomy[Title/Abstract]
            OR gastrectomy[Title/Abstract]
            OR esophagectomy[Title/Abstract]
            OR lobectomy[Title/Abstract]
        )
        AND
        (
            predict*[Title/Abstract]
            OR risk[Title/Abstract]
            OR model[Title/Abstract]
            OR nomogram[Title/Abstract]
            OR score[Title/Abstract]
            OR calculator[Title/Abstract]
            OR machine learning[Title/Abstract]
            OR artificial intelligence[Title/Abstract]
            OR biomarker[Title/Abstract]
        )
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
    ''',
}


def compact_query(query: str) -> str:
    return " ".join(query.split())


def _publication_date_filter(start_date: date, end_date: date) -> str:
    return (
        f'("{start_date:%Y/%m/%d}"[Date - Publication] : '
        f'"{end_date:%Y/%m/%d}"[Date - Publication])'
    )


def effective_date_bounds(start_year: int, end_year: int) -> tuple[date, date]:
    """Return the requested date range, excluding future publication dates."""
    if start_year > end_year:
        raise ValueError("start_year must not be later than end_year")

    start_date = date(start_year, 1, 1)
    end_date = min(date(end_year, 12, 31), date.today())
    if start_date > end_date:
        raise ValueError(
            "The requested publication-date range contains no dates up to today."
        )
    return start_date, end_date


def build_query(query: str, start_year: int, end_year: int) -> str:
    if start_year > end_year:
        raise ValueError("start_year must not be later than end_year")
    date_filter = f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'
    return f"({compact_query(query)}) {compact_query(GLOBAL_FILTER)} AND {date_filter}"


def build_date_range_query(query: str, start_date: date, end_date: date) -> str:
    """Add global eligibility and a day-level slice to one module query."""
    if start_date > end_date:
        raise ValueError("start_date must not be later than end_date")
    return (
        f"({compact_query(query)}) {compact_query(GLOBAL_FILTER)} "
        f"AND {_publication_date_filter(start_date, end_date)}"
    )


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _publication_date(date_element: ElementTree.Element | None) -> str | None:
    """Return an ISO date from an ArticleDate or PubDate XML element."""
    year = _parse_int(_element_text(date_element.find("Year")) if date_element is not None else "")
    if year is None:
        medline_date = _element_text(
            date_element.find("MedlineDate") if date_element is not None else None
        )
        for token in medline_date.split():
            year = _parse_int(token[:4])
            if year is not None:
                break
    if year is None:
        return None

    month_text = _element_text(date_element.find("Month")) if date_element is not None else ""
    month = _parse_int(month_text)
    if month is None:
        month = MONTHS.get(month_text.lower().rstrip("."), 1)
    month = month if month and 1 <= month <= 12 else 1

    day = _parse_int(_element_text(date_element.find("Day")) if date_element is not None else "")
    day = day if day and 1 <= day <= 31 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _abstract_length_exclusion_reason(abstract: str) -> str | None:
    if len(abstract) <= MIN_ABSTRACT_CHARS:
        return "abstract_too_short"
    if len(abstract) >= MAX_ABSTRACT_CHARS:
        return "abstract_too_long"
    return None


def parse_pubmed_article(
    article: ElementTree.Element,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one PubMed XML article and return a post-fetch exclusion reason."""
    pmid = _normalize_whitespace(
        _element_text(article.find("./MedlineCitation/PMID"))
    )
    title = _normalize_whitespace(
        _element_text(article.find("./MedlineCitation/Article/ArticleTitle"))
    )
    abstract_parts: list[str] = []
    for abstract_text in article.findall(
        "./MedlineCitation/Article/Abstract/AbstractText"
    ):
        text = _element_text(abstract_text)
        if text:
            label = abstract_text.attrib.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = _normalize_whitespace(" ".join(abstract_parts))

    article_date = article.find("./MedlineCitation/Article/ArticleDate")
    journal_date = article.find(
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
    )
    publication_date = _publication_date(article_date) or _publication_date(journal_date)
    publication_types = [
        _element_text(publication_type)
        for publication_type in article.findall(
            "./MedlineCitation/Article/PublicationTypeList/PublicationType"
        )
        if _element_text(publication_type)
    ]
    languages = [
        _element_text(language)
        for language in article.findall("./MedlineCitation/Article/Language")
        if _element_text(language)
    ]

    if not pmid:
        return None, "missing_pmid"
    if not title:
        return None, "missing_title"
    if not abstract:
        return None, "missing_abstract"
    if not publication_date:
        return None, "missing_publication_date"
    if len(title) < MIN_TITLE_CHARS:
        return None, "title_too_short"
    abstract_length_reason = _abstract_length_exclusion_reason(abstract)
    if abstract_length_reason is not None:
        return None, abstract_length_reason
    if not any(language.casefold() in _ENGLISH_LANGUAGE_VALUES for language in languages):
        return None, "non_english"
    if any(
        publication_type.casefold() in _EXCLUDED_PUBLICATION_TYPES_NORMALIZED
        for publication_type in publication_types
    ):
        return None, "excluded_publication_type"

    mesh_terms = [
        _element_text(mesh)
        for mesh in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")
        if _element_text(mesh)
    ]
    doi = ""
    for article_id in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType", "").lower() == "doi":
            doi = _element_text(article_id)
            break

    return {
        "id": f"PMID:{pmid}",
        "pmid": pmid,
        "doi": doi,
        "display_name": title,
        "publication_date": publication_date,
        "publication_year": int(publication_date[:4]),
        "abstract": abstract,
        "source_id": _element_text(article.find("./MedlineCitation/Article/Journal/Title")),
        "mesh_terms": "; ".join(dict.fromkeys(mesh_terms)),
        "publication_types": "; ".join(dict.fromkeys(publication_types)),
        "languages": "; ".join(dict.fromkeys(languages)),
        "is_retracted": "Retracted Publication" in publication_types,
        # Compatibility fields: the biomedical graph should be built with
        # --include_elements False and later adds llm_concepts.
        "elements": "",
        "concepts": "[]",
    }, None


def abstract_length_is_eligible(abstract: str) -> bool:
    """Keep raw abstracts strictly longer than 300 and shorter than 4,000 chars."""
    return _abstract_length_exclusion_reason(_normalize_whitespace(abstract)) is None


def module_mapping(module_pmids: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert module result lists while retaining every module membership."""
    pmid_modules: dict[str, list[str]] = defaultdict(list)
    for module, pmids in module_pmids.items():
        for pmid in pmids:
            if module not in pmid_modules[pmid]:
                pmid_modules[pmid].append(module)
    return dict(pmid_modules)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _write_state(
    state_path: Path,
    *,
    configuration_hash: str,
    module_pmids: dict[str, list[str]],
    module_searches: dict[str, dict[str, Any]],
    excluded_pmids_by_reason: dict[str, set[str]],
    exclusion_counts: Counter[str],
) -> None:
    state = {
        "configuration_hash": configuration_hash,
        "module_pmids": module_pmids,
        "module_searches": module_searches,
        "excluded_pmids_by_reason": {
            reason: sorted(pmids)
            for reason, pmids in sorted(excluded_pmids_by_reason.items())
            if pmids
        },
        "exclusion_counts": {
            reason: int(exclusion_counts[reason])
            for reason in EXCLUSION_REASONS
            if exclusion_counts[reason]
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")


def _configuration_hash(
    module_queries: dict[str, str], start_year: int, end_year: int, max_results: int
) -> str:
    config = {
        "module_queries": {key: compact_query(value) for key, value in module_queries.items()},
        "global_filter": compact_query(GLOBAL_FILTER),
        "start_year": start_year,
        "end_year": end_year,
        "max_results_per_module": max_results,
        "post_fetch_filter": {
            "min_abstract_chars_exclusive": MIN_ABSTRACT_CHARS,
            "max_abstract_chars_exclusive": MAX_ABSTRACT_CHARS,
            "min_title_chars_inclusive": MIN_TITLE_CHARS,
            "english_language_values": sorted(_ENGLISH_LANGUAGE_VALUES),
            "excluded_publication_types": sorted(EXCLUDED_PUBLICATION_TYPES),
        },
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class PubMedClient:
    """Small rate-limited client for the E-utilities endpoints used here."""

    def __init__(
        self,
        *,
        email: str,
        tool: str,
        api_key: str | None,
        request_delay_seconds: float,
        timeout_seconds: float,
        max_retries: int,
    ):
        self.email = email
        self.tool = tool
        self.api_key = api_key
        self.request_delay_seconds = request_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{tool} ({email})"})
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        remaining = self.request_delay_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def get(self, endpoint: str, params: dict[str, Any]) -> requests.Response:
        request_params = {
            **params,
            "email": self.email,
            "tool": self.tool,
        }
        if self.api_key:
            request_params["api_key"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(
                    f"{EUTILS_BASE_URL}/{endpoint}",
                    params=request_params,
                    timeout=self.timeout_seconds,
                )
                self._last_request_at = time.monotonic()
                if response.status_code == 429:
                    last_error = RuntimeError("HTTP 429 Too Many Requests")
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else attempt * 2
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(attempt * 2)

        raise RuntimeError(
            f"PubMed request to {endpoint} failed after {self.max_retries} attempts: {last_error}"
        )

    def get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.get(endpoint, params).json()
        except ValueError as error:
            raise RuntimeError(f"PubMed returned invalid JSON from {endpoint}") from error

    def close(self) -> None:
        self.session.close()


def search_history(client: PubMedClient, query: str) -> tuple[int, str, str]:
    result = client.get_json(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retmode": "json",
            "usehistory": "y",
            "sort": "pub date",
        },
    ).get("esearchresult", {})
    try:
        return int(result["count"]), str(result["webenv"]), str(result["querykey"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("PubMed ESearch response omitted count, WebEnv, or query key.") from error


def history_pmids(
    client: PubMedClient,
    *,
    count: int,
    webenv: str,
    query_key: str,
    limit: int,
) -> list[str]:
    """Read up to 10,000 PMIDs from one ESearch History result."""
    target = count if limit == 0 else min(count, limit)
    if target > MAX_PUBMED_HISTORY_RESULTS:
        raise ValueError(
            "A single PubMed History query cannot retrieve more than "
            f"{MAX_PUBMED_HISTORY_RESULTS:,} PMIDs; split the search by date."
        )
    pmids: list[str] = []
    for start in range(0, target, SUMMARY_BATCH_SIZE):
        expected_count = min(SUMMARY_BATCH_SIZE, target - start)
        # ESummary returns article titles and other metadata that this step does
        # not need. A malformed character in one of those fields can make its
        # JSON invalid. EFetch's uilist/text response contains only PMIDs.
        response = client.get(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "WebEnv": webenv,
                "query_key": query_key,
                "retstart": start,
                "retmax": expected_count,
                "rettype": "uilist",
                "retmode": "text",
            },
        )
        response_lines = [line.strip() for line in response.text.splitlines()]
        unexpected_lines = [line for line in response_lines if line and not line.isdigit()]
        if unexpected_lines:
            raise RuntimeError(
                "PubMed returned an unexpected UID-list response at "
                f"offset {start}: {unexpected_lines[0][:120]!r}"
            )
        batch_pmids = [line for line in response_lines if line]
        if len(batch_pmids) > expected_count:
            raise RuntimeError(
                "PubMed UID-list endpoint returned more PMIDs than requested at "
                f"offset {start}: expected at most {expected_count}, "
                f"received {len(batch_pmids)}."
            )
        if len(batch_pmids) < expected_count:
            click.echo(
                "Warning: PubMed UID-list endpoint returned "
                f"{len(batch_pmids)}/{expected_count} PMIDs at offset {start}. "
                "Continuing; the missing record will be recorded in the manifest.",
                err=True,
            )
        pmids.extend(batch_pmids)
    return list(dict.fromkeys(pmids))


def collect_date_sliced_pmids(
    client: PubMedClient,
    *,
    module_query: str,
    start_date: date,
    end_date: date,
    reported_count: int,
    initial_webenv: str,
    initial_query_key: str,
    limit: int,
    progress_label: str,
) -> tuple[list[str], list[dict[str, Any]], int, int]:
    """Select PMIDs newest-first while respecting PubMed's 10k result limit."""
    target = reported_count if limit == 0 else min(reported_count, limit)
    pmids: list[str] = []
    retrieval_slices: list[dict[str, Any]] = []
    uid_list_missing_count = 0
    oversized_day_slices = 0

    def collect_slice(
        slice_start: date,
        slice_end: date,
        search_result: tuple[int, str, str] | None = None,
    ) -> None:
        nonlocal uid_list_missing_count, oversized_day_slices
        if len(pmids) >= target:
            return

        query = build_date_range_query(module_query, slice_start, slice_end)
        count, webenv, query_key = search_result or search_history(client, query)
        retrieval_slices.append(
            {
                "start_date": slice_start.isoformat(),
                "end_date": slice_end.isoformat(),
                "reported_count": count,
            }
        )
        if count == 0:
            return

        if count > MAX_PUBMED_HISTORY_RESULTS and slice_start < slice_end:
            midpoint = slice_start + timedelta(
                days=(slice_end - slice_start).days // 2
            )
            # Descending publication-date selection: exhaust the newer range
            # before moving to the older one.
            collect_slice(midpoint + timedelta(days=1), slice_end)
            collect_slice(slice_start, midpoint)
            return

        remaining = target - len(pmids)
        requested_count = min(count, MAX_PUBMED_HISTORY_RESULTS)
        if count > MAX_PUBMED_HISTORY_RESULTS:
            if limit == 0:
                raise RuntimeError(
                    "A single publication date has more than 10,000 matching "
                    "PubMed records. Narrow the query or use a local PubMed "
                    "data source for an uncapped collection."
                )
            oversized_day_slices += 1
            click.echo(
                "Warning: PubMed returned more than 10,000 matching records for "
                f"{slice_start.isoformat()}. Keeping its first 10,000 PMIDs and "
                "continuing with older dates.",
                err=True,
            )

        retrieved = history_pmids(
            client,
            count=count,
            webenv=webenv,
            query_key=query_key,
            limit=requested_count,
        )
        uid_list_missing_count += requested_count - len(retrieved)
        selected = retrieved[:remaining]
        pmids.extend(selected)
        progress.update(len(selected))

    with click.progressbar(
        length=target,
        label=progress_label,
        show_percent=True,
        show_pos=True,
    ) as progress:
        initial_result = (reported_count, initial_webenv, initial_query_key)
        collect_slice(start_date, end_date, initial_result)

    return pmids, retrieval_slices, uid_list_missing_count, oversized_day_slices


def fetch_articles(
    client: PubMedClient, pmids: list[str]
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, set[str]], Counter[str]]:
    """Fetch records and report unresolved PMIDs and post-fetch exclusions."""
    records: dict[str, dict[str, Any]] = {}
    unresolved: set[str] = set()
    excluded_pmids_by_reason: dict[str, set[str]] = defaultdict(set)
    exclusion_counts: Counter[str] = Counter()

    def request_xml(batch: list[str]) -> ElementTree.Element | None:
        """Retry parse failures because PubMed can return truncated HTTP-200 XML."""
        max_parse_retries = getattr(client, "max_retries", 1)
        for attempt in range(1, max_parse_retries + 1):
            response = client.get(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            )
            try:
                return ElementTree.fromstring(response.content)
            except ElementTree.ParseError:
                if attempt < max_parse_retries:
                    time.sleep(attempt * 2)
        return None

    def process_batch(batch: list[str]) -> None:
        root = request_xml(batch)
        if root is None:
            if len(batch) == 1:
                pmid = batch[0]
                click.echo(
                    f"Warning: PubMed returned unparseable XML for PMID {pmid}; "
                    "excluding it from this run.",
                    err=True,
                )
                excluded_pmids_by_reason["unparseable_xml"].add(pmid)
                exclusion_counts["unparseable_xml"] += 1
                return
            midpoint = len(batch) // 2
            process_batch(batch[:midpoint])
            process_batch(batch[midpoint:])
            return

        returned_pmids: set[str] = set()
        for article in root.findall("./PubmedArticle"):
            parsed, exclusion_reason = parse_pubmed_article(article)
            pmid = _element_text(article.find("./MedlineCitation/PMID"))
            if pmid:
                returned_pmids.add(pmid)
            if parsed is not None:
                records[parsed["pmid"]] = parsed
            elif exclusion_reason is not None:
                exclusion_counts[exclusion_reason] += 1
                if pmid:
                    excluded_pmids_by_reason[exclusion_reason].add(pmid)
        unresolved.update(set(batch) - returned_pmids)

    for start in range(0, len(pmids), FETCH_BATCH_SIZE):
        process_batch(pmids[start : start + FETCH_BATCH_SIZE])
    return records, unresolved, dict(excluded_pmids_by_reason), exclusion_counts


def _read_existing_output(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"pmid": str})
    if "pmid" not in df or df["pmid"].duplicated().any():
        raise RuntimeError(f"Cannot resume from {path}: PMIDs must be present and unique.")
    return {str(row["pmid"]): row.to_dict() for _, row in df.iterrows()}


def _write_output(records: dict[str, dict[str, Any]], output_path: Path) -> None:
    columns = [
        "id",
        "pmid",
        "doi",
        "display_name",
        "publication_date",
        "publication_year",
        "abstract",
        "source_id",
        "modules",
        "mesh_terms",
        "publication_types",
        "languages",
        "is_retracted",
        "elements",
        "concepts",
    ]
    df = pd.DataFrame(records.values(), columns=columns)
    if not df.empty:
        df = df.sort_values(["publication_date", "pmid"], kind="stable")
    _atomic_write_csv(df, output_path)


def _load_or_discover_module_pmids(
    *,
    client: PubMedClient,
    module_queries: dict[str, str],
    start_year: int,
    end_year: int,
    max_results_per_module: int,
    state_path: Path,
    resume: bool,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    str,
    dict[str, set[str]],
    Counter[str],
]:
    config_hash = _configuration_hash(
        module_queries, start_year, end_year, max_results_per_module
    )
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_hash") != config_hash:
            raise RuntimeError(
                "The existing PubMed state belongs to a different query configuration. "
                "Choose a new output/state path or rerun with --no-resume."
        )
        module_pmids = state.get("module_pmids", {})
        module_searches = state.get("module_searches", {})
        excluded_pmids_by_reason = {
            reason: set(pmids)
            for reason, pmids in state.get("excluded_pmids_by_reason", {}).items()
        }
        exclusion_counts = Counter(state.get("exclusion_counts", {}))
    else:
        module_pmids = {}
        module_searches = {}
        excluded_pmids_by_reason = {}
        exclusion_counts = Counter()

    start_date, end_date = effective_date_bounds(start_year, end_year)

    for module, module_query in module_queries.items():
        if module in module_pmids and module in module_searches:
            click.echo(f"{module}: reusing {len(module_pmids[module]):,} checkpointed PMIDs")
            continue
        query = build_date_range_query(module_query, start_date, end_date)
        click.echo(f"{module}: searching PubMed…")
        count, webenv, query_key = search_history(client, query)
        target = count if max_results_per_module == 0 else min(
            count, max_results_per_module
        )
        click.echo(f"{module}: found {count:,} records; collecting {target:,} PMIDs.")
        if count > MAX_PUBMED_HISTORY_RESULTS:
            click.echo(
                f"{module}: splitting the date range to respect PubMed's "
                f"{MAX_PUBMED_HISTORY_RESULTS:,}-record retrieval limit."
            )
        pmids, retrieval_slices, uid_list_missing_count, oversized_day_slices = (
            collect_date_sliced_pmids(
                client,
                module_query=module_query,
                start_date=start_date,
                end_date=end_date,
                reported_count=count,
                initial_webenv=webenv,
                initial_query_key=query_key,
                limit=max_results_per_module,
                progress_label=f"{module}: collecting PMIDs",
            )
        )
        module_pmids[module] = pmids
        module_searches[module] = {
            "query": query,
            "effective_start_date": start_date.isoformat(),
            "effective_end_date": end_date.isoformat(),
            "reported_count": count,
            "retrieved_count": len(pmids),
            "uid_list_missing_count": uid_list_missing_count,
            "oversized_day_slices": oversized_day_slices,
            "retrieval_slices": retrieval_slices,
            "selection_order": "publication_date_descending",
            "truncated": target < count,
        }
        _write_state(
            state_path,
            configuration_hash=config_hash,
            module_pmids=module_pmids,
            module_searches=module_searches,
            excluded_pmids_by_reason=excluded_pmids_by_reason,
            exclusion_counts=exclusion_counts,
        )
        click.echo(f"{module}: {len(pmids):,}/{count:,} PMIDs")
    return (
        module_pmids,
        module_searches,
        config_hash,
        excluded_pmids_by_reason,
        exclusion_counts,
    )


@click.command()
@click.option("--output", "output_path", default=DEFAULT_OUTPUT, show_default=True, type=click.Path(path_type=Path))
@click.option("--module", "selected_modules", multiple=True, help="Run only these module names; repeat this option as needed.")
@click.option(
    "--start-year",
    default=DEFAULT_START_YEAR,
    show_default=True,
    type=click.IntRange(1700, 2100),
)
@click.option(
    "--end-year",
    default=DEFAULT_END_YEAR,
    show_default=True,
    type=click.IntRange(1700, 2100),
)
@click.option(
    "--max-results-per-module",
    default=DEFAULT_MAX_RESULTS_PER_MODULE,
    show_default=True,
    type=click.IntRange(0),
    help="Maximum records per module; pass 0 to request all records (very large).",
)
@click.option("--email", envvar="NCBI_EMAIL", required=True, help="Contact email required by NCBI; NCBI_EMAIL is also accepted.")
@click.option("--tool", default="materials_concepts", show_default=True)
@click.option("--api-key-env", default="NCBI_API_KEY", show_default=True, help="Optional environment variable containing an NCBI API key.")
@click.option("--request-delay-seconds", default=0.34, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--timeout-seconds", default=60.0, show_default=True, type=click.FloatRange(min=1.0))
@click.option("--max-retries", default=4, show_default=True, type=click.IntRange(1, 20))
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--state-path", type=click.Path(path_type=Path), default=None, help="Defaults to <output>.state.json and stores PMID/module checkpoints.")
@click.option("--manifest-path", type=click.Path(path_type=Path), default=None, help="Defaults to <output>.manifest.json.")
def main(
    output_path: Path,
    selected_modules: tuple[str, ...],
    start_year: int,
    end_year: int,
    max_results_per_module: int,
    email: str,
    tool: str,
    api_key_env: str,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    resume: bool,
    state_path: Path | None,
    manifest_path: Path | None,
) -> None:
    """Search PubMed modules, fetch unique abstracts, and write a works table."""
    if start_year > end_year:
        raise click.ClickException("--start-year must not be later than --end-year.")
    try:
        effective_start_date, effective_end_date = effective_date_bounds(
            start_year, end_year
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    unknown_modules = set(selected_modules) - set(MODULES)
    if unknown_modules:
        raise click.ClickException("Unknown modules: " + ", ".join(sorted(unknown_modules)))
    module_queries = {
        module: MODULES[module]
        for module in (selected_modules or tuple(MODULES))
    }
    state_path = state_path or output_path.with_suffix(output_path.suffix + ".state.json")
    manifest_path = manifest_path or output_path.with_suffix(output_path.suffix + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(
        f"Searching {len(module_queries)} PubMed module(s), "
        f"effective publication dates {effective_start_date.isoformat()} to "
        f"{effective_end_date.isoformat()}."
    )

    client = PubMedClient(
        email=email,
        tool=tool,
        api_key=os.environ.get(api_key_env),
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    try:
        (
            module_pmids,
            module_searches,
            config_hash,
            excluded_pmids_by_reason,
            exclusion_counts,
        ) = _load_or_discover_module_pmids(
            client=client,
            module_queries=module_queries,
            start_year=start_year,
            end_year=end_year,
            max_results_per_module=max_results_per_module,
            state_path=state_path,
            resume=resume,
        )
        pmid_modules = module_mapping(module_pmids)
        click.echo(
            f"PMID discovery complete: found {len(pmid_modules):,} unique PMIDs "
            f"across {len(module_pmids):,} module(s)."
        )
        existing_records = _read_existing_output(output_path) if resume else {}
        excluded_pmids = set().union(*excluded_pmids_by_reason.values())
        pending_pmids = [
            pmid
            for pmid in pmid_modules
            if pmid not in existing_records and pmid not in excluded_pmids
        ]
        click.echo(
            f"Fetching {len(pending_pmids):,} unique PubMed records "
            f"({len(existing_records):,} already written; "
            f"{len(excluded_pmids):,} previously excluded)."
        )

        unresolved_pmids: set[str] = set()
        with click.progressbar(
            length=len(pending_pmids),
            label="Fetching and parsing abstracts",
            show_percent=True,
            show_pos=True,
        ) as progress:
            for start in range(0, len(pending_pmids), FETCH_BATCH_SIZE):
                batch = pending_pmids[start : start + FETCH_BATCH_SIZE]
                (
                    fetched,
                    unresolved,
                    batch_excluded_pmids_by_reason,
                    batch_exclusion_counts,
                ) = fetch_articles(
                    client, batch
                )
                unresolved_pmids.update(unresolved)
                for reason, excluded_pmids_for_reason in (
                    batch_excluded_pmids_by_reason.items()
                ):
                    excluded_pmids_by_reason.setdefault(reason, set()).update(
                        excluded_pmids_for_reason
                    )
                exclusion_counts.update(batch_exclusion_counts)
                for pmid, record in fetched.items():
                    record["modules"] = "; ".join(pmid_modules[pmid])
                    existing_records[pmid] = record
                _write_output(existing_records, output_path)
                _write_state(
                    state_path,
                    configuration_hash=config_hash,
                    module_pmids=module_pmids,
                    module_searches=module_searches,
                    excluded_pmids_by_reason=excluded_pmids_by_reason,
                    exclusion_counts=exclusion_counts,
                )
                progress.update(len(batch))

        # Also create a valid header-only CSV when a query has no results.
        _write_output(existing_records, output_path)
        click.echo(
            f"Post-fetch eligibility filter excluded "
            f"{sum(exclusion_counts.values()):,} record(s)."
        )

        manifest = {
            "configuration_hash": config_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output_path),
            "state_path": str(state_path),
            "start_year": start_year,
            "end_year": end_year,
            "global_filter": compact_query(GLOBAL_FILTER),
            "effective_start_date": effective_start_date.isoformat(),
            "effective_end_date": effective_end_date.isoformat(),
            "post_fetch_filter": {
                "min_abstract_chars_exclusive": MIN_ABSTRACT_CHARS,
                "max_abstract_chars_exclusive": MAX_ABSTRACT_CHARS,
                "min_title_chars_inclusive": MIN_TITLE_CHARS,
                "english_language_values": sorted(_ENGLISH_LANGUAGE_VALUES),
                "excluded_publication_types": sorted(EXCLUDED_PUBLICATION_TYPES),
            },
            "max_results_per_module": max_results_per_module,
            "module_searches": module_searches,
            "unique_pmids_requested": len(pmid_modules),
            "works_written": len(existing_records),
            "unresolved_pmids": sorted(unresolved_pmids),
            "exclusion_counts": {
                reason: int(exclusion_counts[reason]) for reason in EXCLUSION_REASONS
            },
            "excluded_pmids_by_reason": {
                reason: sorted(excluded_pmids_by_reason.get(reason, set()))
                for reason in EXCLUSION_REASONS
                if excluded_pmids_by_reason.get(reason)
            },
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
        click.echo(f"Wrote {len(existing_records):,} works to {output_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
