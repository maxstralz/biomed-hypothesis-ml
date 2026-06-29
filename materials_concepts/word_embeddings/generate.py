import gzip
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from materials_concepts.utils.utils import prepare_dataframe

tqdm.pandas()

DIM_EMBEDDING = 768
MAX_TOKENS = 510  # 512 - 2 (CLS and SEP)

# The current canonical Hugging Face name for PubMedBERT. Microsoft renamed the
# model family to BiomedBERT; the abstract-plus-full-text checkpoint is the
# direct replacement for the former PubMedBERT checkpoint.
MATERIALS_EMBEDDING_MODEL = "m3rg-iitd/matscibert"
BIOMEDICAL_EMBEDDING_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
EMBEDDING_FILE_PATTERN = re.compile(r"embeddings_(\d+)\.pkl\.gz$")


def setup_logger(level=logging.INFO, log_to_stdout=True):
    """Configure only this module's logger, not every dependency's logger."""
    logger = logging.getLogger("materials_concepts.word_embeddings.generate")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler("logs.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def quiet_dependency_logs():
    """Keep model-download and HTTP implementation details out of normal runs."""
    for name in ("huggingface_hub", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def setup_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return tokenizer, model


def select_device(device_name="auto"):
    """Choose CUDA, then Apple MPS, then CPU unless explicitly requested."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available.")
    return device


def init_get_embeddings(model, tokenizer, device):
    def func(text):
        # Work only with content-token IDs here. The tokenizer is responsible
        # for adding its own model-specific special tokens to every chunk.
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        max_content_tokens = min(
            MAX_TOKENS,
            tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(
                pair=False
            ),
        )
        if max_content_tokens <= 0:
            raise ValueError("Tokenizer has no usable content-token capacity.")

        chunks = [
            tokens[i : i + max_content_tokens]
            for i in range(0, len(tokens), max_content_tokens)
        ]

        embedded_chunks = []
        with torch.no_grad():
            for chunk in chunks:
                # Both supported checkpoints are BERT encoders.  Transformers
                # 5 removed BertTokenizer.prepare_for_model(), so construct
                # the one-sequence BERT input explicitly using the tokenizer's
                # own special-token IDs rather than hard-coded numeric IDs.
                # This preserves the one output vector per original content
                # token needed for matching concepts against abstracts.
                if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
                    raise ValueError(
                        "The selected embedding model must provide BERT-style "
                        "[CLS] and [SEP] token IDs."
                    )
                input_ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
                encoded = {
                    "input_ids": torch.tensor([input_ids], device=device),
                    "attention_mask": torch.ones(
                        (1, len(input_ids)), dtype=torch.long, device=device
                    ),
                    "token_type_ids": torch.zeros(
                        (1, len(input_ids)), dtype=torch.long, device=device
                    ),
                }
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state.squeeze(0)
                content_embeddings = embeddings[1:-1]
                if len(content_embeddings) != len(chunk):
                    raise RuntimeError(
                        "BERT special-token handling did not preserve the "
                        "expected number of content-token embeddings."
                    )
                embedded_chunks.append(content_embeddings)

        return torch.cat(embedded_chunks)

    return func


def find_sequence_in_list(lst, seq):
    seq_len = len(seq)
    indices = []
    for i in range(len(lst)):
        if lst[i : i + seq_len] == seq:
            indices.append(i)
    return indices


def init_get_token_ids(tokenizer):
    def func(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return func


def get_concept_embedding(
    abstract_embedding, concept_tokens, indices, aggregation=torch.mean
):
    concept_embeddings = [
        abstract_embedding[i : i + len(concept_tokens)] for i in indices
    ]

    averaged_concept_embeddings = torch.stack(
        [aggregation(cs, dim=0) for cs in concept_embeddings]
    )

    return aggregation(averaged_concept_embeddings, dim=0)


def extract_embeddings_for_abstract(
    abstract_embedding, abstract_tokens, concepts, aggregation=torch.mean
):
    avg_embedding = torch.mean(abstract_embedding, dim=0)

    concept_embeddings = {}
    for concept in concepts:
        concept_tokens = get_token_ids(concept)
        indices = find_sequence_in_list(abstract_tokens, concept_tokens)
        logger.debug(f"Concept: {concept} - Occurences: {len(indices)}")

        concept_embedding = (
            get_concept_embedding(
                abstract_embedding, concept_tokens, indices, aggregation
            )
            if indices
            else avg_embedding  # use average embedding of abstract
        )

        concept_embeddings[concept] = concept_embedding

    return concept_embeddings


def save_compressed(obj, path):
    compressed = gzip.compress(pickle.dumps(obj))
    with open(path, "wb") as f:
        f.write(compressed)


def load_compressed(path):
    with open(path, "rb") as f:
        return pickle.loads(gzip.decompress(f.read()))


def existing_embedding_chunks(output_path):
    """Return existing chunk paths and their numeric suffixes in write order."""
    chunks = []
    for path in Path(output_path).glob("embeddings_*.pkl.gz"):
        match = EMBEDDING_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            chunks.append((int(match.group(1)), path))
    return sorted(chunks)


def existing_embedding_ids(chunks):
    """Read only work IDs from existing output chunks for resumable extraction."""
    ids = set()
    if not chunks:
        return ids
    for _, path in tqdm(chunks, desc="Reading existing embedding chunks"):
        chunk = load_compressed(path)
        if not isinstance(chunk, dict):
            raise ValueError(f"Embedding chunk is not a dictionary: {path}")
        ids.update(str(work_id) for work_id in chunk)
    return ids


def process_works(df, desc):
    store = {}

    for id, abstract, concepts in tqdm(
        zip(df.id, df.abstract, df.concepts, strict=False), total=len(df), desc=desc
    ):
        logger.debug(
            f"Process {id}: abstract len of {len(abstract)} with {len(concepts)} concepts"
        )

        abstract_tokens = get_token_ids(abstract)

        abstract_embedding = get_embeddings(abstract)

        logger.debug(
            f"Tokenized abstract length: {len(abstract_tokens)}, Abstract embedding shape: {abstract_embedding.shape}"
        )

        embeddings = extract_embeddings_for_abstract(
            abstract_embedding, abstract_tokens, concepts, aggregation=torch.mean
        )

        assert len(embeddings.values()) == len(concepts)

        store[str(id)] = {k: v.cpu() for k, v in embeddings.items()}

    return store


def default_embedding_model(concept_column):
    """Select the domain model without changing the materials workflow."""
    if concept_column == "llm_concepts":
        return BIOMEDICAL_EMBEDDING_MODEL
    return MATERIALS_EMBEDDING_MODEL


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large.csv",
    output_path="data/embeddings/large/",
    embedding_model=None,
    concept_column="llama_concepts",
    log_to_stdout=False,
    step_size=500,
    start=0,
    end=None,
    resume=True,
    device_name="auto",
):
    global logger, get_embeddings, get_token_ids
    logger = setup_logger(logging.INFO, log_to_stdout=log_to_stdout)
    quiet_dependency_logs()
    embedding_model = embedding_model or default_embedding_model(concept_column)

    device = select_device(device_name)
    logger.info(f"Using device: {device}")
    logger.info(f"Embedding model: {embedding_model}")

    logger.info("Prepare dataframe")

    df = prepare_dataframe(
        # Avoid pandas' chunked type inference warning for serialized concept
        # lists, which legitimately contain blanks alongside JSON strings.
        df=pd.read_csv(concepts_path, low_memory=False),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "abstract", "concepts"],
        concept_column=concept_column,
    )
    df["id"] = df["id"].astype(str)
    input_records = len(df)
    df = df[df["concepts"].map(bool)].reset_index(drop=True)
    logger.info(
        "Keeping %d of %d records with at least one retained graph concept",
        len(df),
        input_records,
    )

    if end is None:
        end = len(df)
    if not 0 <= start <= end:
        raise ValueError("Require 0 <= start <= end.")
    if end > len(df):
        raise ValueError(
            f"Requested end={end:,}, but only {len(df):,} records have retained concepts."
        )
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    os.makedirs(output_path, exist_ok=True)
    chunks = existing_embedding_chunks(output_path)
    if resume:
        completed_ids = existing_embedding_ids(chunks)
        pending_df = df.iloc[start:end]
        pending_df = pending_df[~pending_df["id"].isin(completed_ids)].reset_index(
            drop=True
        )
        next_chunk_index = chunks[-1][0] + 1 if chunks else 0
        logger.info(
            "Resume: %d completed work(s) in %d chunk(s); %d of %d selected "
            "work(s) remain.",
            len(completed_ids),
            len(chunks),
            len(pending_df),
            end - start,
        )
    else:
        if chunks:
            raise ValueError(
                "Output directory already contains embedding chunks. Use "
                "--resume True or choose a new output_path."
            )
        pending_df = df.iloc[start:end].reset_index(drop=True)
        next_chunk_index = 0

    logger.info("Setup model")
    tokenizer, model = setup_model(embedding_model, device)
    get_embeddings = init_get_embeddings(model, tokenizer, device)
    get_token_ids = init_get_token_ids(tokenizer)

    logger.info("Generate word embeddings")

    for offset in range(0, len(pending_df), step_size):
        chunk_end = min(offset + step_size, len(pending_df))
        chunk_index = next_chunk_index + offset // step_size
        logger.info(f"Process pending records {offset} to {chunk_end}...")
        partial_df = pending_df.iloc[offset:chunk_end]
        store = process_works(
            partial_df,
            desc=f"Generate embeddings ({chunk_index})",
        )
        logger.info("Save embeddings")
        save_path = os.path.join(output_path, f"embeddings_{chunk_index:06d}.pkl.gz")
        save_compressed(store, save_path)


if __name__ == "__main__":
    fire.Fire(main)
