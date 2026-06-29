# Biomedical PubMed workflow

This is an additive workflow for building a temporal biomedical hypothesis graph
from the included PubMed surgical-search modules. It does not alter the published
materials-science data or reproduction path.

## 1. Collect PubMed records

```bash
export NCBI_EMAIL='you@example.org'

python -m materials_concepts.dataset.downloader.download_pubmed \
  --output data/table/surgery.pubmed.works.csv \
  --start-year 2000 \
  --end-year 2025
```

The command holds the surgical `MODULES` search definitions, stores all matching
modules for each PMID, fetches each unique PMID once, and writes the
source-neutral fields used by the rest of the project: `id`, `pmid`, `doi`,
`display_name`, `publication_date`, `abstract`, `source_id`, `modules`, MeSH
terms, publication types and languages. It saves an atomic output checkpoint,
query/PMID state file, and a run manifest alongside the output.

Every module shares one global eligibility filter: it requires an abstract,
English language and human studies, and excludes reviews, systematic reviews,
meta-analyses, editorials, letters, comments and case reports. The
`--start-year` and `--end-year` options directly control the publication-date
range (the defaults are 2000–2025). Use `--module name` one or more times to
retrieve only selected search modules. An optional `NCBI_API_KEY` can be provided
through the environment; it is never written to output files.

After retrieval, the downloader keeps only records with a PMID, title,
publication date and English-language metadata; titles of at least 10
characters; and raw abstracts strictly longer than 300 and strictly shorter than
4,000 characters. It rejects the publication types Review, Systematic Review,
Meta-Analysis, Editorial, Letter, Comment, Case Reports and Retracted
Publication. Excluded PMIDs and counts by reason are checkpointed and reported
in the run manifest.

For safety, the default cap is **50,000 records per module**, matching the
original `01_abstract_mining.py` script. The broad `core_surgery` query otherwise
matches more than two million records, which is usually unsuitable for local CSV
handling and downstream LLM extraction. Pass `--max-results-per-module 0` only
for a deliberately large, separately planned collection.

The cap selects the newest matching records first, using PubMed publication-date
order. When a module has more than 10,000 matches, the downloader automatically
splits its publication-date range into smaller searches before collecting PMIDs;
this is required by PubMed's retrieval limit and avoids dropping later result
pages. The manifest records every date slice used for each module.

## 2. Clean titles and abstracts

```bash
python -m materials_concepts.dataset.preparation.clean_abstracts \
  --input data/table/surgery.pubmed.works.csv \
  --output data/table/surgery.pubmed.cleaned.csv
```

The cleaning step prepends the title to the abstract. This is intentional: title
terms often supply the most useful concise procedure or disease-subtype labels.

## 3. Extract biomedical concepts with an OpenAI-compatible server

```bash
export OPENAI_API_KEY='...'

python -m materials_concepts.dataset.preparation.extract_llm_concepts \
  --input data/table/surgery.pubmed.cleaned.csv \
  --output data/table/surgery.pubmed.llm.csv \
  --model YOUR_MODEL_NAME \
  --base-url https://YOUR_SERVER/v1 \
  --batch-size 6
```

The extractor sends only the PMID and abstract text in each request. It writes a
JSON list to `llm_concepts`, resumes by default, and checkpoints after every 100
handled abstracts. By default it uses batches of 6, three concurrent workers,
and a maximum of 12 concepts per paper.
Use `--workers 1` for a constrained local Ollama server, or raise
`--max-concepts` to 20 to restore the earlier cap.

It writes `<output>.metadata.json` with the effective settings and current-run
failures, and appends every skipped batch to `<output>.failures.jsonl`. Neither
contains the API key. Case and whitespace are normalized locally; duplicate,
generic, malformed, one-word, and excess labels are simply removed. A
structurally malformed answer (invalid JSON or wrong PMIDs) is logged, left blank
in the CSV, and **not** repaired, split, or sent again. Run the same command later
and `--resume` retries only those blank rows. Transient network and HTTP 429/5xx
failures receive at most one retry by default: `--max-attempts 2` means one
initial request plus one retry.

`--response-format json_schema` is the default and should be used whenever the
server supports OpenAI-style structured output. For compatible servers that do
not support it, try `--response-format json_object`, then `none` only as a final
fallback; local response validation remains enabled in all modes.

The extractor uses the OpenAI-compatible Chat Completions endpoint
`<base-url>/chat/completions`. A server that exposes only the Responses API
needs a small provider adapter before it can be used with this command.

For a low-cost smoke test, use `--test`. It reads and processes only the first
configured test-limit records (pass `--test-limit 20` for a 20-paper run) and
writes a normal, resumable test output; use a separate output filename so it
cannot be confused with a complete corpus run.

```bash
python -m materials_concepts.dataset.preparation.extract_llm_concepts \
  --input data/table/surgery.pubmed.cleaned.csv \
  --output data/table/surgery.pubmed.gpt-5-nano.test.llm.csv \
  --model gpt-5-nano \
  --test \
  --test-limit 20 \
  --batch-size 6 \
  --workers 3 \
  --checkpoint-every 20
```

Do a manual quality check on a stratified pilot before running the entire corpus.
In particular, inspect concepts across every search module and check that specific
clinical terms are not being replaced by generic labels.

### Large OpenAI runs: Batch API

For an offline corpus-scale OpenAI run, use the Batch API companion instead of
the synchronous extractor. It has a 24-hour completion window and cannot be used
with Ollama. `submit` creates remote jobs only for blank `llm_concepts` rows; it
does **not** rewrite the output CSV. `collect` validates remote responses and
fills blank cells only, preserving every already-computed concept list.

By default, each `submit` creates one conservatively sized remote job (estimated
below 1.5 million input tokens). This fits beneath the 2-million-token Batch
queue limit of a Tier-1 `gpt-4o-mini` account. After collecting that job, run
`submit` again to queue the next blank rows. If the dashboard shows a larger
Batch queue limit, increase `--max-estimated-input-tokens` and/or
`--max-jobs-per-submit`.

```bash
# Autonomous mode: submit, poll, collect, and submit the next chunk until done.
python -m materials_concepts.dataset.preparation.extract_llm_concepts_batch watch \
  --input data/table/surgery.pubmed.works.csv \
  --output data/table/surgery.pubmed.gpt-4o-mini.llm.csv \
  --model gpt-4o-mini \
  --batch-size 6 \
  --poll-seconds 300

# Optional: inspect the current remote job in a second terminal.
python -m materials_concepts.dataset.preparation.extract_llm_concepts_batch status \
  --output data/table/surgery.pubmed.gpt-4o-mini.llm.csv
```

It writes `<output>.batch.state.json`, which tracks remote job IDs and request to
PMID mappings, and `<output>.batch/`, which stores submitted JSONL input and
downloaded result files. Keep both until all jobs have been collected. Do not run
the synchronous extractor and Batch `collect` against the same output file at
the same time. `watch` is safe to stop with Ctrl-C and resume using the exact
same command.

## 4. Build a concept-only biomedical graph

The materials formula extractor is intentionally skipped. The biomedical concept
extractor is responsible for biomarkers, interventions and clinically meaningful
entities, while `--include_elements False` prevents the materials-specific formula
logic from adding nodes.

```bash
python -m materials_concepts.graph.build \
  --input_path data/table/surgery.pubmed.llm.csv \
  --output_path data/graph/surgery.edges.pkl \
  --output_lookup_path data/table/lookup/surgery.lookup.csv \
  --colname llm_concepts \
  --min_occurence 3 \
  --min_words 2 \
  --max_words 6 \
  --min_length 2 \
  --include_elements False
```

For `--colname llm_concepts`, the graph builder uses biomedical-aware label
filtering: it keeps numeric and Unicode labels (for example, `type 2 diabetes`
and `α-fetoprotein level`), enforces the chosen word range, and additionally
permits the canonical one-token abbreviations `ct`, `crispr`, `ercp`, `icu`,
`mri`, `pcr`, and `tme`. It does not apply the material-science LLaMA or
ASCII-only filters. The original `llama_concepts` workflow is unchanged.

The semantic embedding generator is resumable. It skips CSV rows with no
retained graph concepts and, by default, scans `embeddings_*.pkl.gz` in its
output directory to skip already embedded work IDs. After LLM extraction adds
concepts to formerly blank rows, rerun the same command and only those new
concept-bearing records are embedded. Use a fresh output directory if you
change the embedding model or want to refresh already embedded records.

The resulting graph is compatible with the existing temporal data generation and
topological baseline. For semantic features, pass `--concept_column llm_concepts`
to the word-embedding scripts. `generate.py` automatically selects
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (the current
name for PubMedBERT) for that column; the original `llama_concepts` workflow
continues to default to MatSciBERT. You can still override this with
`--embedding_model`.

## Temporal evaluation

Use only fully observed publication years to create labels. For example, if the
latest complete year is 2024, train on a historical three-year formation window
and reserve a later complete three-year window for final testing. Do not treat a
partially observed current year as a negative-label period.
