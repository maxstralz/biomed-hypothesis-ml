import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from materials_concepts.dataset.preparation.extract_llm_concepts_batch import (
    _collect_job,
    _load_existing_concepts_preserving,
    _read_jsonl,
    _write_request_files,
)


class TestLlmBatchExtraction(unittest.TestCase):
    def test_existing_nonblank_concepts_are_preserved_verbatim(self):
        input_df = pd.DataFrame(
            {
                "pmid": ["101", "102"],
                "abstract": ["First abstract.", "Second abstract."],
            }
        )
        prior_value = '["INR threshold for removal", "proctectomy"]'

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.csv"
            pd.DataFrame(
                {
                    "pmid": ["101", "102"],
                    "llm_concepts": [prior_value, pd.NA],
                }
            ).to_csv(output_path, index=False)
            result = _load_existing_concepts_preserving(
                input_df,
                output_path,
                pmid_column="pmid",
                concept_column="llm_concepts",
            )

        self.assertEqual(result.at[0, "llm_concepts"], prior_value)
        self.assertTrue(pd.isna(result.at[1, "llm_concepts"]))

    def test_request_files_contain_only_requested_pending_pmids(self):
        df = pd.DataFrame(
            [
                {"pmid": "101", "abstract": "First abstract."},
                {"pmid": "102", "abstract": "Second abstract."},
                {"pmid": "103", "abstract": "Third abstract."},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            drafts = _write_request_files(
                df=df,
                pending_indices=[1, 2],
                pmid_column="pmid",
                records_per_request=2,
                model="gpt-4o-mini",
                response_format="json_schema",
                max_concepts=12,
                request_directory=Path(tmpdir),
                max_requests_per_file=50_000,
                max_file_bytes=10_000,
                max_estimated_input_tokens=10_000,
                max_files=1,
            )

            self.assertEqual(len(drafts), 1)
            self.assertEqual(drafts[0]["requests"][0]["pmids"], ["102", "103"])
            line = next(_read_jsonl(Path(drafts[0]["request_file"])))
            request_records = json.loads(
                line["body"]["messages"][1]["content"].split("<records>", 1)[1]
                .split("</records>", 1)[0]
            )["records"]
            self.assertEqual([record["pmid"] for record in request_records], ["102", "103"])

    def test_collection_does_not_overwrite_existing_concepts(self):
        df = pd.DataFrame(
            {
                "pmid": ["101", "102"],
                "llm_concepts": [
                    json.dumps(["existing concept"]),
                    pd.NA,
                ],
            }
        )
        job = {
            "batch_id": "batch_test",
            "output_file_id": "file_output",
            "requests": [{"custom_id": "request-1", "pmids": ["101", "102"]}],
        }
        completion = {
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {
                                                "pmid": "101",
                                                "concepts": ["replacement concept"],
                                            },
                                            {
                                                "pmid": "102",
                                                "concepts": ["new surgical concept"],
                                            },
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "batch_test.output.jsonl").write_text(
                json.dumps(completion) + "\n", encoding="utf-8"
            )
            filled, preserved = _collect_job(
                job=job,
                df=df,
                pmid_column="pmid",
                concept_column="llm_concepts",
                max_concepts=12,
                base_url="https://example.test/v1",
                api_key="test-key",
                request_directory=directory,
                failure_log_path=directory / "failures.jsonl",
                timeout=10,
            )

            self.assertEqual((filled, preserved), (1, 1))
            self.assertEqual(json.loads(df.at[0, "llm_concepts"]), ["existing concept"])
            self.assertEqual(json.loads(df.at[1, "llm_concepts"]), ["new surgical concept"])


if __name__ == "__main__":
    unittest.main()
