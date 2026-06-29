"""Filter a works CSV to rows with non-empty serialized concept lists.

Example:
    python -m materials_concepts.dataset.preparation.filter_rows_with_concepts \
      --input data/table/surgery.pubmed.gpt-5-nano.llm.csv \
      --output data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv
"""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any

import click
import pandas as pd


DEFAULT_INPUT = Path("data/table/surgery.pubmed.gpt-5-nano.llm.csv")
DEFAULT_OUTPUT = Path("data/table/surgery.pubmed.gpt-5-nano.llm.with-concepts.csv")
DEFAULT_CONCEPT_COLUMN = "llm_concepts"
DEFAULT_ID_COLUMN = "id"
DEFAULT_PMID_COLUMN = "pmid"
DEFAULT_TITLE_COLUMN = "display_name"
DEFAULT_YEAR_COLUMN = "publication_year"
DEFAULT_PRINT_LIMIT = 200


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _parse_serialized_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        raise ValueError("missing_concepts")

    text = str(value).strip()
    if not text:
        raise ValueError("missing_concepts")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = literal_eval(text)

    if not isinstance(parsed, list):
        raise ValueError("concept_value_is_not_a_list")
    return parsed


def _concept_status(value: object) -> tuple[bool, str, int]:
    """Return (has_concepts, exclusion_reason, concept_count)."""
    try:
        concepts = _parse_serialized_list(value)
    except Exception as error:
        reason = str(error) or error.__class__.__name__
        return False, reason, 0

    clean_concepts = [_clean_text(concept) for concept in concepts]
    clean_concepts = [concept for concept in clean_concepts if concept]

    if not clean_concepts:
        return False, "empty_concept_list", 0
    return True, "", len(clean_concepts)


def _default_excluded_output(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".excluded.csv")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _display_value(row: pd.Series, column: str) -> str:
    if column not in row:
        return ""
    return _clean_text(row[column])


@click.command()
@click.option(
    "--input",
    "input_path",
    default=DEFAULT_INPUT,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input CSV containing a serialized concept-list column.",
)
@click.option(
    "--output",
    "output_path",
    default=DEFAULT_OUTPUT,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Filtered CSV containing only rows with at least one concept.",
)
@click.option(
    "--excluded-output",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV containing excluded rows and exclusion reasons. Defaults to OUTPUT.excluded.csv.",
)
@click.option("--concept-column", default=DEFAULT_CONCEPT_COLUMN, show_default=True)
@click.option("--id-column", default=DEFAULT_ID_COLUMN, show_default=True)
@click.option("--pmid-column", default=DEFAULT_PMID_COLUMN, show_default=True)
@click.option("--title-column", default=DEFAULT_TITLE_COLUMN, show_default=True)
@click.option("--year-column", default=DEFAULT_YEAR_COLUMN, show_default=True)
@click.option(
    "--print-limit",
    default=DEFAULT_PRINT_LIMIT,
    show_default=True,
    type=click.IntRange(min=0),
    help="Maximum excluded papers printed to the terminal unless --print-all is used.",
)
@click.option(
    "--print-all",
    is_flag=True,
    help="Print every excluded paper to the terminal. This can be very verbose.",
)
def main(
    input_path: Path,
    output_path: Path,
    excluded_output: Path | None,
    concept_column: str,
    id_column: str,
    pmid_column: str,
    title_column: str,
    year_column: str,
    print_limit: int,
    print_all: bool,
) -> None:
    """Remove rows whose concept column is missing, invalid, or an empty list."""
    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("--input and --output must be different files.")

    excluded_output = excluded_output or _default_excluded_output(output_path)
    df = pd.read_csv(input_path, low_memory=False)
    if concept_column not in df.columns:
        raise click.ClickException(
            f"Input CSV does not contain concept column '{concept_column}'."
        )

    statuses = df[concept_column].apply(_concept_status)
    has_concepts = statuses.apply(lambda item: item[0])
    reasons = statuses.apply(lambda item: item[1])
    concept_counts = statuses.apply(lambda item: item[2])

    kept = df[has_concepts].copy()
    excluded = df[~has_concepts].copy()
    excluded.insert(0, "csv_line_number", excluded.index + 2)
    excluded.insert(1, "exclusion_reason", reasons[~has_concepts].to_numpy())
    excluded.insert(2, "parsed_concept_count", concept_counts[~has_concepts].to_numpy())

    _atomic_write_csv(kept, output_path)
    _atomic_write_csv(excluded, excluded_output)

    click.echo(f"Input rows:    {len(df):,}")
    click.echo(f"Kept rows:     {len(kept):,}")
    click.echo(f"Excluded rows: {len(excluded):,}")
    click.echo(f"Filtered CSV:  {output_path}")
    click.echo(f"Excluded CSV:  {excluded_output}")

    if len(excluded):
        click.echo("\nExclusion reasons:")
        for reason, count in excluded["exclusion_reason"].value_counts().items():
            click.echo(f"  {reason}: {count:,}")

        rows_to_print = len(excluded) if print_all else min(print_limit, len(excluded))
        click.echo(
            f"\nExcluded papers shown in terminal: {rows_to_print:,}/{len(excluded):,}"
        )
        if not print_all and rows_to_print < len(excluded):
            click.echo("Use --print-all to print every excluded paper.")

        for _, row in excluded.head(rows_to_print).iterrows():
            line = row["csv_line_number"]
            reason = row["exclusion_reason"]
            work_id = _display_value(row, id_column)
            pmid = _display_value(row, pmid_column)
            year = _display_value(row, year_column)
            title = _display_value(row, title_column)
            click.echo(
                f"  line={line} pmid={pmid or '<missing>'} "
                f"id={work_id or '<missing>'} year={year or '<missing>'} "
                f"reason={reason} title={title or '<missing>'}"
            )


if __name__ == "__main__":
    main()
