"""Run biomedical concept extraction through OpenAI's asynchronous Batch API.

This is deliberately separate from ``extract_llm_concepts.py``.  The
synchronous extractor remains useful for local models and small tests, while
this command handles large OpenAI jobs in three explicit stages:

``submit``
    Create JSONL input files for only the rows whose ``llm_concepts`` value is
    blank, upload them, and create Batch API jobs.
``status``
    Show the remote state of every submitted job.
``collect``
    Download completed responses, validate them, and fill only blank cells in
    the output CSV.  Existing concepts are never overwritten or deleted.

The state file and JSONL input files are durable.  Do not delete them until all
jobs are collected.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import click
import pandas as pd
import requests

from .extract_llm_concepts import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCEPT_COLUMN,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_MODEL,
    DEFAULT_PMID_COLUMN,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_TEST_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_CONCEPTS,
    Batch,
    InvalidResponseError,
    _atomic_write_csv,
    _atomic_write_json,
    _clean_text,
    _guard_test_output,
    iter_batches,
    make_messages,
    response_format_payload,
    validate_response,
)


# ---------------------------------------------------------------------------
# Defaults. Every CLI option below refers to one of these constants.
# ---------------------------------------------------------------------------

STATE_VERSION = 1
DEFAULT_MAX_REQUESTS_PER_FILE = 50_000
DEFAULT_MAX_FILE_MIB = 190
DEFAULT_MAX_ESTIMATED_INPUT_TOKENS = 1_500_000
DEFAULT_MAX_JOBS_PER_SUBMIT = 1
DEFAULT_POLL_SECONDS = 300
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_BATCH_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled", "not_found"}


class BatchApiError(RuntimeError):
    """An OpenAI Batch API request failed."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(output_path: Path, configured_path: Path | None) -> Path:
    return configured_path or output_path.with_suffix(output_path.suffix + ".batch.state.json")


def _request_directory(output_path: Path, configured_path: Path | None) -> Path:
    return configured_path or output_path.with_suffix(output_path.suffix + ".batch")


def _failure_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".batch.failures.jsonl")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _error_detail(response: requests.Response) -> str:
    try:
        detail: object = response.json()
    except ValueError:
        detail = response.text
    text = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)
    return " ".join(text.split())[:500]


def _raise_for_api_error(response: requests.Response, operation: str) -> None:
    if response.ok:
        return
    raise BatchApiError(
        f"Batch API {operation} failed with HTTP {response.status_code}: "
        f"{_error_detail(response)}"
    )


def _api_get(base_url: str, api_key: str, path: str, timeout: float) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}{path}", headers=_headers(api_key), timeout=timeout
    )
    _raise_for_api_error(response, path)
    body = response.json()
    if not isinstance(body, dict):
        raise BatchApiError(f"Batch API {path} returned a non-object JSON response.")
    return body


def _upload_input_file(
    *, base_url: str, api_key: str, request_file: Path, timeout: float
) -> str:
    with request_file.open("rb") as file:
        response = requests.post(
            f"{base_url.rstrip('/')}/files",
            headers=_headers(api_key),
            data={"purpose": "batch"},
            files={"file": (request_file.name, file, "application/jsonl")},
            timeout=timeout,
        )
    _raise_for_api_error(response, "upload input file")
    body = response.json()
    file_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(file_id, str) or not file_id:
        raise BatchApiError("Batch API upload response did not contain a file ID.")
    return file_id


def _create_batch(
    *,
    base_url: str,
    api_key: str,
    input_file_id: str,
    metadata: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/batches",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json={
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": DEFAULT_COMPLETION_WINDOW,
            "metadata": metadata,
        },
        timeout=timeout,
    )
    _raise_for_api_error(response, "create batch")
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("id"), str):
        raise BatchApiError("Batch API create response did not contain a batch ID.")
    return body


def _download_file(
    *, base_url: str, api_key: str, file_id: str, destination: Path, timeout: float
) -> None:
    response = requests.get(
        f"{base_url.rstrip('/')}/files/{file_id}/content",
        headers=_headers(api_key),
        timeout=timeout,
    )
    _raise_for_api_error(response, f"download file {file_id}")
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(destination)


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        raise click.ClickException(
            f"No Batch API state file exists at {state_path}. Run 'submit' first."
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise click.ClickException(f"Could not read Batch API state: {error}") from error
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise click.ClickException("The Batch API state file has an unsupported format.")
    if not isinstance(state.get("jobs"), list):
        raise click.ClickException("The Batch API state file has no valid jobs list.")
    return state


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _now_utc()
    _atomic_write_json(state_path, state)


def _new_state(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    response_format: str,
    concept_column: str,
    pmid_column: str,
    max_concepts: int,
    records_per_request: int,
    test_mode: bool,
    test_limit: int,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at_utc": _now_utc(),
        "updated_at_utc": _now_utc(),
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "model": model,
        "base_url": base_url.rstrip("/"),
        "response_format": response_format,
        "concept_column": concept_column,
        "pmid_column": pmid_column,
        "max_concepts": max_concepts,
        "records_per_request": records_per_request,
        "test_mode": test_mode,
        "test_limit": test_limit if test_mode else None,
        "jobs": [],
    }


def _assert_matching_state(
    state: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    response_format: str,
    concept_column: str,
    pmid_column: str,
    max_concepts: int,
    records_per_request: int,
    test_mode: bool,
    test_limit: int,
) -> None:
    expected = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "model": model,
        "base_url": base_url.rstrip("/"),
        "response_format": response_format,
        "concept_column": concept_column,
        "pmid_column": pmid_column,
        "max_concepts": max_concepts,
        "records_per_request": records_per_request,
        "test_mode": test_mode,
        "test_limit": test_limit if test_mode else None,
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise click.ClickException(
            "The existing Batch API state uses different settings: "
            + ", ".join(mismatches)
            + ". Use its original settings, or choose a different output path."
        )


def _load_works(
    *, input_path: Path, pmid_column: str, test_mode: bool, test_limit: int
) -> pd.DataFrame:
    df = pd.read_csv(
        input_path,
        dtype={"id": str, pmid_column: str},
        nrows=test_limit if test_mode else None,
    )
    required_columns = {"id", pmid_column, "abstract"}
    missing = required_columns - set(df.columns)
    if missing:
        raise click.ClickException("Input is missing columns: " + ", ".join(sorted(missing)))
    df["id"] = df["id"].map(_clean_text)
    df[pmid_column] = df[pmid_column].map(_clean_text)
    if (df["id"] == "").any() or df["id"].duplicated().any():
        raise click.ClickException("Input 'id' values must be present and unique.")
    if (df[pmid_column] == "").any() or df[pmid_column].duplicated().any():
        raise click.ClickException(f"Input '{pmid_column}' values must be present and unique.")
    if (df["abstract"].map(_clean_text) == "").any():
        raise click.ClickException("Input abstracts must be present and non-empty.")
    return df


def _load_existing_concepts_preserving(
    df: pd.DataFrame,
    output_path: Path,
    *,
    pmid_column: str,
    concept_column: str,
) -> pd.DataFrame:
    """Load prior concepts by PMID without changing any nonblank value.

    Batch collection must not delete or normalize concepts created by an earlier
    synchronous run. A nonblank value, including ``[]``, is therefore treated as
    completed and copied verbatim. Only blank cells are eligible for a Batch
    result.
    """
    df[concept_column] = pd.NA
    if not output_path.exists():
        return df

    previous = pd.read_csv(output_path, dtype={pmid_column: str, concept_column: str})
    required = {pmid_column, concept_column}
    missing = required - set(previous.columns)
    if missing:
        raise click.ClickException(
            f"Cannot reuse {output_path}: missing " + ", ".join(sorted(missing))
        )
    previous[pmid_column] = previous[pmid_column].map(_clean_text)
    if (previous[pmid_column] == "").any() or previous[pmid_column].duplicated().any():
        raise click.ClickException(
            f"Cannot reuse {output_path}: '{pmid_column}' must be present and unique."
        )
    existing = {
        pmid: value
        for pmid, value in zip(
            previous[pmid_column], previous[concept_column], strict=True
        )
        if _clean_text(value)
    }
    df[concept_column] = df[pmid_column].map(existing)
    return df


def _active_submitted_pmids(state: dict[str, Any]) -> set[str]:
    """Return PMIDs in jobs not yet collected, preventing duplicate submissions."""
    active: set[str] = set()
    for job in state["jobs"]:
        if job.get("collected"):
            continue
        for request in job.get("requests", []):
            active.update(request.get("pmids", []))
    return active


def _request_body(
    *, model: str, records: list[dict[str, str]], response_format: str, max_concepts: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": make_messages(records, max_concepts),
    }
    format_payload = response_format_payload(response_format, max_concepts)
    if format_payload is not None:
        body["response_format"] = format_payload
    return body


def _write_request_files(
    *,
    df: pd.DataFrame,
    pending_indices: list[int],
    pmid_column: str,
    records_per_request: int,
    model: str,
    response_format: str,
    max_concepts: int,
    request_directory: Path,
    max_requests_per_file: int,
    max_file_bytes: int,
    max_estimated_input_tokens: int,
    max_files: int,
) -> list[dict[str, Any]]:
    """Write JSONL request files without loading the whole payload into memory."""
    request_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    drafts: list[dict[str, Any]] = []
    file_number = 0
    request_number = 0
    current_file: Any = None
    current_path: Path | None = None
    current_requests: list[dict[str, Any]] = []
    current_bytes = 0
    current_estimated_tokens = 0

    def close_current_file() -> None:
        nonlocal current_file, current_path, current_requests, current_bytes, current_estimated_tokens
        if current_file is None or current_path is None:
            return
        current_file.close()
        drafts.append(
            {
                "request_file": str(current_path.resolve()),
                "request_count": len(current_requests),
                "estimated_input_tokens": current_estimated_tokens,
                "requests": current_requests,
            }
        )
        current_file = None
        current_path = None
        current_requests = []
        current_bytes = 0
        current_estimated_tokens = 0

    for batch in iter_batches(df, pending_indices, records_per_request, pmid_column):
        if current_file is None and len(drafts) >= max_files:
            break
        request_number += 1
        custom_id = f"biomedical-concepts-{run_id}-{request_number:07d}"
        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _request_body(
                model=model,
                records=batch.records,
                response_format=response_format,
                max_concepts=max_concepts,
            ),
        }
        encoded_line = (json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        # The API applies an enqueued *token* limit in addition to its JSONL file
        # limit. This intentionally conservative approximation avoids a tiktoken
        # dependency while keeping a Tier-1 gpt-4o-mini job below its 2M queue cap.
        estimated_tokens = max(1, (len(encoded_line) + 3) // 4)
        if len(encoded_line) > max_file_bytes:
            raise click.ClickException(
                f"One Batch API request is {len(encoded_line):,} bytes, larger than "
                f"the configured per-file cap of {max_file_bytes:,} bytes."
            )
        if current_file is None:
            file_number += 1
            current_path = request_directory / f"{run_id}.{file_number:03d}.jsonl"
            current_file = current_path.open("wb")
        elif (
            len(current_requests) >= max_requests_per_file
            or current_bytes + len(encoded_line) > max_file_bytes
            or current_estimated_tokens + estimated_tokens > max_estimated_input_tokens
        ):
            close_current_file()
            if len(drafts) >= max_files:
                break
            file_number += 1
            current_path = request_directory / f"{run_id}.{file_number:03d}.jsonl"
            current_file = current_path.open("wb")

        current_file.write(encoded_line)
        current_bytes += len(encoded_line)
        current_estimated_tokens += estimated_tokens
        current_requests.append(
            {
                "custom_id": custom_id,
                "pmids": [record["pmid"] for record in batch.records],
            }
        )

    close_current_file()
    return drafts


def _refresh_job_statuses(
    *, state: dict[str, Any], base_url: str, api_key: str, timeout: float
) -> None:
    for job in state["jobs"]:
        if job.get("collected"):
            continue
        if job.get("status") == "not_found":
            continue
        batch_id = job.get("batch_id")
        if not isinstance(batch_id, str):
            continue
        try:
            remote = _api_get(base_url, api_key, f"/batches/{batch_id}", timeout)
        except BatchApiError as error:
            message = str(error)
            if "HTTP 404" not in message or "No batch found" not in message:
                raise
            job["status"] = "not_found"
            job["request_counts"] = job.get("request_counts") or {
                "total": job.get("request_count", 0),
                "completed": 0,
                "failed": job.get("request_count", 0),
            }
            job["output_file_id"] = None
            job["error_file_id"] = None
            job["last_checked_at_utc"] = _now_utc()
            job["remote_missing_at_utc"] = _now_utc()
            job["remote_missing_error"] = message
            continue
        job["status"] = remote.get("status", "unknown")
        job["request_counts"] = remote.get("request_counts")
        job["output_file_id"] = remote.get("output_file_id")
        job["error_file_id"] = remote.get("error_file_id")
        job["last_checked_at_utc"] = _now_utc()


def _append_failure(
    failure_log_path: Path,
    *,
    batch_id: str,
    custom_id: str,
    pmids: list[str],
    error: str,
) -> None:
    record = {
        "timestamp_utc": _now_utc(),
        "batch_id": batch_id,
        "custom_id": custom_id,
        "pmids": pmids,
        "error": " ".join(error.split())[:500],
    }
    with failure_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _request_map(job: dict[str, Any]) -> dict[str, list[str]]:
    return {
        request["custom_id"]: request["pmids"]
        for request in job.get("requests", [])
        if isinstance(request, dict)
        and isinstance(request.get("custom_id"), str)
        and isinstance(request.get("pmids"), list)
    }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchApiError(f"Invalid JSONL in {path} at line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise BatchApiError(f"Invalid JSONL object in {path} at line {line_number}.")
            yield value


def _collect_job(
    *,
    job: dict[str, Any],
    df: pd.DataFrame,
    pmid_column: str,
    concept_column: str,
    max_concepts: int,
    base_url: str,
    api_key: str,
    request_directory: Path,
    failure_log_path: Path,
    timeout: float,
) -> tuple[int, int]:
    """Collect one terminal job; return (new concepts, preserved concepts)."""
    batch_id = str(job["batch_id"])
    expected = _request_map(job)
    seen: set[str] = set()
    pmid_to_index = {str(pmid): index for index, pmid in df[pmid_column].items()}
    filled = 0
    preserved = 0

    output_file_id = job.get("output_file_id")
    if isinstance(output_file_id, str) and output_file_id:
        output_path = request_directory / f"{batch_id}.output.jsonl"
        if not output_path.exists():
            _download_file(
                base_url=base_url,
                api_key=api_key,
                file_id=output_file_id,
                destination=output_path,
                timeout=timeout,
            )
        for result in _read_jsonl(output_path):
            custom_id = result.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in expected:
                continue
            seen.add(custom_id)
            pmids = expected[custom_id]
            response = result.get("response")
            if not isinstance(response, dict) or response.get("status_code") != 200:
                _append_failure(
                    failure_log_path,
                    batch_id=batch_id,
                    custom_id=custom_id,
                    pmids=pmids,
                    error=f"Batch response failed: {response!r}",
                )
                continue
            try:
                body = response["body"]
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise InvalidResponseError("Completion response did not contain text.")
                concepts_by_pmid = validate_response(
                    json.loads(content), pmids, max_concepts
                )
            except (KeyError, TypeError, ValueError, InvalidResponseError) as error:
                _append_failure(
                    failure_log_path,
                    batch_id=batch_id,
                    custom_id=custom_id,
                    pmids=pmids,
                    error=f"Invalid Batch API completion: {error}",
                )
                continue

            for pmid, concepts in concepts_by_pmid.items():
                index = pmid_to_index.get(pmid)
                if index is None:
                    _append_failure(
                        failure_log_path,
                        batch_id=batch_id,
                        custom_id=custom_id,
                        pmids=[pmid],
                        error="PMID is not present in the current input CSV.",
                    )
                    continue
                if pd.isna(df.at[index, concept_column]):
                    df.at[index, concept_column] = json.dumps(concepts, ensure_ascii=False)
                    filled += 1
                else:
                    # A synchronous or earlier Batch run completed this row after
                    # submission. Preserve that value exactly.
                    preserved += 1

    error_file_id = job.get("error_file_id")
    if isinstance(error_file_id, str) and error_file_id:
        error_path = request_directory / f"{batch_id}.errors.jsonl"
        if not error_path.exists():
            _download_file(
                base_url=base_url,
                api_key=api_key,
                file_id=error_file_id,
                destination=error_path,
                timeout=timeout,
            )
        for result in _read_jsonl(error_path):
            custom_id = result.get("custom_id")
            if isinstance(custom_id, str) and custom_id in expected:
                seen.add(custom_id)
                _append_failure(
                    failure_log_path,
                    batch_id=batch_id,
                    custom_id=custom_id,
                    pmids=expected[custom_id],
                    error=f"Batch API request error: {result.get('error')!r}",
                )

    for custom_id, pmids in expected.items():
        if custom_id not in seen:
            if job.get("status") == "not_found":
                error_message = (
                    "Remote Batch API job was not found. The associated PMIDs "
                    "remain blank and are eligible for resubmission."
                )
            else:
                error_message = (
                    "No completed response was returned for this Batch API request."
                )
            _append_failure(
                failure_log_path,
                batch_id=batch_id,
                custom_id=custom_id,
                pmids=pmids,
                error=error_message,
            )
    return filled, preserved


def submit(
    *,
    input_path: Path,
    output_path: Path,
    state_path: Path,
    request_directory: Path,
    model: str,
    base_url: str,
    api_key: str,
    response_format: str,
    concept_column: str,
    pmid_column: str,
    records_per_request: int,
    max_concepts: int,
    max_requests_per_file: int,
    max_file_bytes: int,
    max_estimated_input_tokens: int,
    max_jobs_per_submit: int,
    test_mode: bool,
    test_limit: int,
    timeout: float,
) -> None:
    df = _load_works(
        input_path=input_path,
        pmid_column=pmid_column,
        test_mode=test_mode,
        test_limit=test_limit,
    )
    if test_mode:
        _guard_test_output(output_path, test_limit)
        click.echo(f"Test mode: considering only the first {len(df):,} input records.")
    df = _load_existing_concepts_preserving(
        df,
        output_path,
        pmid_column=pmid_column,
        concept_column=concept_column,
    )
    if state_path.exists():
        state = _load_state(state_path)
        _assert_matching_state(
            state,
            input_path=input_path,
            output_path=output_path,
            model=model,
            base_url=base_url,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            max_concepts=max_concepts,
            records_per_request=records_per_request,
            test_mode=test_mode,
            test_limit=test_limit,
        )
        _refresh_job_statuses(
            state=state, base_url=base_url, api_key=api_key, timeout=timeout
        )
    else:
        state = _new_state(
            input_path=input_path,
            output_path=output_path,
            model=model,
            base_url=base_url,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            max_concepts=max_concepts,
            records_per_request=records_per_request,
            test_mode=test_mode,
            test_limit=test_limit,
        )

    active_pmids = _active_submitted_pmids(state)
    pending_indices = [
        index
        for index in df[df[concept_column].isna()].index
        if df.at[index, pmid_column] not in active_pmids
    ]
    if not pending_indices:
        _write_state(state_path, state)
        click.echo("No blank, unsubmitted PMIDs remain. Use 'status' or 'collect'.")
        return

    drafts = _write_request_files(
        df=df,
        pending_indices=pending_indices,
        pmid_column=pmid_column,
        records_per_request=records_per_request,
        model=model,
        response_format=response_format,
        max_concepts=max_concepts,
        request_directory=request_directory,
        max_requests_per_file=max_requests_per_file,
        max_file_bytes=max_file_bytes,
        max_estimated_input_tokens=max_estimated_input_tokens,
        max_files=max_jobs_per_submit,
    )
    submitted_requests = 0
    for draft in drafts:
        request_file = Path(draft["request_file"])
        click.echo(
            f"Uploading {request_file.name} with {draft['request_count']:,} request(s)…"
        )
        input_file_id = _upload_input_file(
            base_url=base_url, api_key=api_key, request_file=request_file, timeout=timeout
        )
        remote = _create_batch(
            base_url=base_url,
            api_key=api_key,
            input_file_id=input_file_id,
            metadata={"pipeline": "biomedical-concepts", "model": model},
            timeout=timeout,
        )
        job = {
            **draft,
            "batch_id": remote["id"],
            "input_file_id": input_file_id,
            "status": remote.get("status", "validating"),
            "request_counts": remote.get("request_counts"),
            "output_file_id": remote.get("output_file_id"),
            "error_file_id": remote.get("error_file_id"),
            "submitted_at_utc": _now_utc(),
            "collected": False,
        }
        state["jobs"].append(job)
        _write_state(state_path, state)
        submitted_requests += int(draft["request_count"])
        click.echo(f"Created Batch API job {remote['id']}.")

    submitted_records = sum(
        len(request["pmids"]) for draft in drafts for request in draft["requests"]
    )
    click.echo(
        f"Submitted {submitted_records:,} abstracts in {submitted_requests:,} Batch API request(s).\n"
        f"Run 'status' with --output {output_path} to check progress, then 'collect' "
        "after the jobs complete. Run 'submit' again afterwards to queue the next "
        "blank rows."
    )


def status(*, state_path: Path, base_url: str, api_key: str, timeout: float) -> None:
    state = _load_state(state_path)
    _refresh_job_statuses(state=state, base_url=base_url, api_key=api_key, timeout=timeout)
    _write_state(state_path, state)
    for job in state["jobs"]:
        counts = job.get("request_counts") or {}
        click.echo(
            f"{job.get('batch_id', '<missing>')}: {job.get('status', 'unknown')} "
            f"({counts.get('completed', 0)}/{counts.get('total', job.get('request_count', 0))} "
            f"completed, {counts.get('failed', 0)} failed)"
        )


def collect(
    *,
    input_path: Path,
    output_path: Path,
    state_path: Path,
    request_directory: Path,
    base_url: str,
    api_key: str,
    concept_column: str,
    pmid_column: str,
    max_concepts: int,
    test_mode: bool,
    test_limit: int,
    timeout: float,
) -> None:
    state = _load_state(state_path)
    _assert_matching_state(
        state,
        input_path=input_path,
        output_path=output_path,
        model=state["model"],
        base_url=base_url,
        response_format=state["response_format"],
        concept_column=concept_column,
        pmid_column=pmid_column,
        max_concepts=max_concepts,
        records_per_request=state["records_per_request"],
        test_mode=test_mode,
        test_limit=test_limit,
    )
    df = _load_works(
        input_path=input_path,
        pmid_column=pmid_column,
        test_mode=test_mode,
        test_limit=test_limit,
    )
    if test_mode:
        _guard_test_output(output_path, test_limit)
    df = _load_existing_concepts_preserving(
        df,
        output_path,
        pmid_column=pmid_column,
        concept_column=concept_column,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request_directory.mkdir(parents=True, exist_ok=True)
    failure_log_path = _failure_log_path(output_path)

    _refresh_job_statuses(state=state, base_url=base_url, api_key=api_key, timeout=timeout)
    filled_total = 0
    preserved_total = 0
    collected_jobs = 0
    for job in state["jobs"]:
        if job.get("collected") or job.get("status") not in TERMINAL_BATCH_STATUSES:
            continue
        filled, preserved = _collect_job(
            job=job,
            df=df,
            pmid_column=pmid_column,
            concept_column=concept_column,
            max_concepts=max_concepts,
            base_url=base_url,
            api_key=api_key,
            request_directory=request_directory,
            failure_log_path=failure_log_path,
            timeout=timeout,
        )
        # Write after every remote job. If a later download fails, earlier results
        # are already durable and no existing concepts have been overwritten.
        _atomic_write_csv(df, output_path)
        job["collected"] = True
        job["collected_at_utc"] = _now_utc()
        job["concepts_written"] = filled
        job["existing_concepts_preserved"] = preserved
        _write_state(state_path, state)
        filled_total += filled
        preserved_total += preserved
        collected_jobs += 1
        click.echo(f"Collected {job['batch_id']}: wrote {filled:,}, preserved {preserved:,}.")

    if not collected_jobs:
        click.echo("No completed Batch API jobs are ready to collect yet.")
    else:
        remaining = int(df[concept_column].isna().sum())
        click.echo(
            f"Collection complete: wrote {filled_total:,} concepts and preserved "
            f"{preserved_total:,} already-computed values. {remaining:,} rows remain blank."
        )


def watch(
    *,
    input_path: Path,
    output_path: Path,
    state_path: Path,
    request_directory: Path,
    model: str,
    base_url: str,
    api_key: str,
    response_format: str,
    concept_column: str,
    pmid_column: str,
    records_per_request: int,
    max_concepts: int,
    max_requests_per_file: int,
    max_file_bytes: int,
    max_estimated_input_tokens: int,
    max_jobs_per_submit: int,
    test_mode: bool,
    test_limit: int,
    timeout: float,
    poll_seconds: float,
) -> None:
    """Continuously submit, poll, collect, and continue until no row is blank.

    The state file makes this safe to interrupt with Ctrl-C. Re-running ``watch``
    resumes from the same remote jobs and never overwrites existing concepts.
    """
    click.echo(
        f"Watching Batch API jobs every {poll_seconds:g} seconds. Press Ctrl-C to "
        "stop safely; rerun the same command to resume."
    )
    while True:
        if not state_path.exists():
            submit(
                input_path=input_path,
                output_path=output_path,
                state_path=state_path,
                request_directory=request_directory,
                model=model,
                base_url=base_url,
                api_key=api_key,
                response_format=response_format,
                concept_column=concept_column,
                pmid_column=pmid_column,
                records_per_request=records_per_request,
                max_concepts=max_concepts,
                max_requests_per_file=max_requests_per_file,
                max_file_bytes=max_file_bytes,
                max_estimated_input_tokens=max_estimated_input_tokens,
                max_jobs_per_submit=max_jobs_per_submit,
                test_mode=test_mode,
                test_limit=test_limit,
                timeout=timeout,
            )
            continue

        state = _load_state(state_path)
        _assert_matching_state(
            state,
            input_path=input_path,
            output_path=output_path,
            model=model,
            base_url=base_url,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            max_concepts=max_concepts,
            records_per_request=records_per_request,
            test_mode=test_mode,
            test_limit=test_limit,
        )
        _refresh_job_statuses(state=state, base_url=base_url, api_key=api_key, timeout=timeout)
        _write_state(state_path, state)
        terminal_uncollected = [
            job
            for job in state["jobs"]
            if not job.get("collected") and job.get("status") in TERMINAL_BATCH_STATUSES
        ]
        if terminal_uncollected:
            collect(
                input_path=input_path,
                output_path=output_path,
                state_path=state_path,
                request_directory=request_directory,
                base_url=base_url,
                api_key=api_key,
                concept_column=concept_column,
                pmid_column=pmid_column,
                max_concepts=max_concepts,
                test_mode=test_mode,
                test_limit=test_limit,
                timeout=timeout,
            )
            continue

        active_jobs = [job for job in state["jobs"] if not job.get("collected")]
        if active_jobs:
            counts = active_jobs[0].get("request_counts") or {}
            click.echo(
                f"{active_jobs[0].get('batch_id')}: {active_jobs[0].get('status')} "
                f"({counts.get('completed', 0)}/{counts.get('total', active_jobs[0].get('request_count', 0))} complete). "
                f"Checking again in {poll_seconds:g} seconds."
            )
            time.sleep(poll_seconds)
            continue

        df = _load_works(
            input_path=input_path,
            pmid_column=pmid_column,
            test_mode=test_mode,
            test_limit=test_limit,
        )
        df = _load_existing_concepts_preserving(
            df,
            output_path,
            pmid_column=pmid_column,
            concept_column=concept_column,
        )
        if not df[concept_column].isna().any():
            click.echo(f"Finished: all {len(df):,} rows have saved concepts.")
            return
        submit(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            request_directory=request_directory,
            model=model,
            base_url=base_url,
            api_key=api_key,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            records_per_request=records_per_request,
            max_concepts=max_concepts,
            max_requests_per_file=max_requests_per_file,
            max_file_bytes=max_file_bytes,
            max_estimated_input_tokens=max_estimated_input_tokens,
            max_jobs_per_submit=max_jobs_per_submit,
            test_mode=test_mode,
            test_limit=test_limit,
            timeout=timeout,
        )


@click.command()
@click.argument("action", type=click.Choice(["submit", "status", "collect", "watch"], case_sensitive=False))
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
@click.option("--api-key-env", default=DEFAULT_API_KEY_ENV, show_default=True)
@click.option("--batch-size", "records_per_request", default=DEFAULT_BATCH_SIZE, show_default=True, type=click.IntRange(1, 100), help="Abstracts represented by one Chat Completions request.")
@click.option("--max-concepts", default=DEFAULT_MAX_CONCEPTS, show_default=True, type=click.IntRange(1, MAX_CONCEPTS))
@click.option("--concept-column", default=DEFAULT_CONCEPT_COLUMN, show_default=True)
@click.option("--pmid-column", default=DEFAULT_PMID_COLUMN, show_default=True)
@click.option("--response-format", default=DEFAULT_RESPONSE_FORMAT, show_default=True, type=click.Choice(["json_schema", "json_object", "none"], case_sensitive=False))
@click.option("--timeout", default=DEFAULT_BATCH_TIMEOUT_SECONDS, show_default=True, type=click.FloatRange(min=1.0))
@click.option("--max-requests-per-file", default=DEFAULT_MAX_REQUESTS_PER_FILE, show_default=True, type=click.IntRange(1, 50_000))
@click.option("--max-file-mib", default=DEFAULT_MAX_FILE_MIB, show_default=True, type=click.IntRange(1, 200), help="Safety cap below OpenAI's 200 MiB Batch upload limit.")
@click.option("--max-estimated-input-tokens", default=DEFAULT_MAX_ESTIMATED_INPUT_TOKENS, show_default=True, type=click.IntRange(1), help="Conservative prompt-token cap for each remote Batch job.")
@click.option("--max-jobs-per-submit", default=DEFAULT_MAX_JOBS_PER_SUBMIT, show_default=True, type=click.IntRange(1), help="Remote jobs created by one submit command.")
@click.option("--poll-seconds", default=DEFAULT_POLL_SECONDS, show_default=True, type=click.FloatRange(min=1.0), help="How often watch checks an active remote job.")
@click.option("--batch-state", type=click.Path(path_type=Path), default=None)
@click.option("--batch-directory", type=click.Path(path_type=Path), default=None)
@click.option("--test", "test_mode", is_flag=True, help="Use only the first --test-limit input rows.")
@click.option("--test-limit", default=DEFAULT_TEST_LIMIT, show_default=True, type=click.IntRange(1))
def main(
    action: str,
    input_path: Path | None,
    output_path: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    records_per_request: int,
    max_concepts: int,
    concept_column: str,
    pmid_column: str,
    response_format: str,
    timeout: float,
    max_requests_per_file: int,
    max_file_mib: int,
    max_estimated_input_tokens: int,
    max_jobs_per_submit: int,
    poll_seconds: float,
    batch_state: Path | None,
    batch_directory: Path | None,
    test_mode: bool,
    test_limit: int,
) -> None:
    """Submit, inspect, or collect an OpenAI Batch API concept-extraction run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(output_path, batch_state)
    request_directory = _request_directory(output_path, batch_directory)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise click.ClickException(
            f"Set the API key in ${api_key_env} before using the OpenAI Batch API."
        )

    if action == "status":
        status(state_path=state_path, base_url=base_url, api_key=api_key, timeout=timeout)
        return
    if input_path is None:
        raise click.ClickException("--input is required for 'submit', 'collect', and 'watch'.")
    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("--input and --output must be different files.")

    if action == "submit":
        submit(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            request_directory=request_directory,
            model=model,
            base_url=base_url,
            api_key=api_key,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            records_per_request=records_per_request,
            max_concepts=max_concepts,
            max_requests_per_file=max_requests_per_file,
            max_file_bytes=max_file_mib * 1024 * 1024,
            max_estimated_input_tokens=max_estimated_input_tokens,
            max_jobs_per_submit=max_jobs_per_submit,
            test_mode=test_mode,
            test_limit=test_limit,
            timeout=timeout,
        )
    elif action == "collect":
        collect(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            request_directory=request_directory,
            base_url=base_url,
            api_key=api_key,
            concept_column=concept_column,
            pmid_column=pmid_column,
            max_concepts=max_concepts,
            test_mode=test_mode,
            test_limit=test_limit,
            timeout=timeout,
        )
    else:
        watch(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            request_directory=request_directory,
            model=model,
            base_url=base_url,
            api_key=api_key,
            response_format=response_format,
            concept_column=concept_column,
            pmid_column=pmid_column,
            records_per_request=records_per_request,
            max_concepts=max_concepts,
            max_requests_per_file=max_requests_per_file,
            max_file_bytes=max_file_mib * 1024 * 1024,
            max_estimated_input_tokens=max_estimated_input_tokens,
            max_jobs_per_submit=max_jobs_per_submit,
            test_mode=test_mode,
            test_limit=test_limit,
            timeout=timeout,
            poll_seconds=poll_seconds,
        )


if __name__ == "__main__":
    main()
