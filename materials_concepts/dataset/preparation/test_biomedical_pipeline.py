import json
import pickle
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pandas as pd
import numpy as np
import requests
from click.testing import CliRunner

from materials_concepts.dataset.downloader.download_pubmed import (
    GLOBAL_FILTER,
    MODULES,
    abstract_length_is_eligible,
    build_date_range_query,
    build_query,
    collect_date_sliced_pmids,
    effective_date_bounds,
    fetch_articles,
    history_pmids,
    module_mapping,
    parse_pubmed_article,
)
from materials_concepts.dataset.preparation.extract_llm_concepts import (
    Batch,
    DEFAULT_TEST_LIMIT,
    ExtractionError,
    InvalidResponseError,
    main as extract_llm_concepts,
    request_batch,
    response_format_payload,
    validate_response,
)
from materials_concepts.graph.build import BiomedicalConceptFilter, main as build_graph
from materials_concepts.model.create_data import DataGenerator
from materials_concepts.utils.utils import prepare_dataframe


class TestPubmedDownloader(unittest.TestCase):
    def test_query_and_module_membership_are_preserved(self):
        query = build_query("laparoscopy[Title/Abstract]", 2000, 2025)
        self.assertIn('"2000"[Date - Publication]', query)
        self.assertIn('"2025"[Date - Publication]', query)
        self.assertEqual(query.count("hasabstract[text]"), 1)
        self.assertIn("english[lang]", query)
        self.assertIn("humans[Mesh]", query)
        self.assertIn("case reports[Publication Type]", query)
        self.assertTrue(all("hasabstract[text]" not in module for module in MODULES.values()))
        self.assertEqual(
            module_mapping(
                {
                    "core_surgery": ["101", "102"],
                    "hpb_surgery": ["101"],
                }
            ),
            {"101": ["core_surgery", "hpb_surgery"], "102": ["core_surgery"]},
        )

        date_query = build_date_range_query(
            "laparoscopy[Title/Abstract]", date(2020, 1, 1), date(2020, 1, 31)
        )
        self.assertIn('"2020/01/01"[Date - Publication]', date_query)
        self.assertIn('"2020/01/31"[Date - Publication]', date_query)
        self.assertEqual(date_query.count("hasabstract[text]"), 1)
        self.assertEqual(effective_date_bounds(2000, 2025), (date(2000, 1, 1), date(2025, 12, 31)))
        self.assertIn("meta-analysis[Publication Type]", GLOBAL_FILTER)

    def test_large_module_is_split_by_publication_date(self):
        records = {
            "2020-01-01": [str(number) for number in range(1, 6_002)],
            "2020-01-02": [str(number) for number in range(10_001, 16_002)],
        }

        class FakeResponse:
            def __init__(self, text):
                self.text = text

        class FakeClient:
            def get_json(self, endpoint, params):
                query = params["term"]
                if '"2020/01/01"' in query and '"2020/01/02"' in query:
                    key = "all"
                elif '"2020/01/02"' in query:
                    key = "2020-01-02"
                else:
                    key = "2020-01-01"
                return {
                    "esearchresult": {
                        "count": str(len(records[key])),
                        "webenv": key,
                        "querykey": "1",
                    }
                }

            def get(self, endpoint, params):
                values = records[params["WebEnv"]]
                start = params["retstart"]
                stop = start + params["retmax"]
                return FakeResponse("\n".join(values[start:stop]) + "\n")

        client = FakeClient()
        pmids, slices, missing_count, oversized_days = collect_date_sliced_pmids(
            client,
            module_query="laparoscopy[Title/Abstract]",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            reported_count=12_001,
            initial_webenv="all",
            initial_query_key="1",
            limit=10_000,
            progress_label="Test collection",
        )

        self.assertEqual(len(pmids), 10_000)
        self.assertEqual(pmids[0], "10001")
        self.assertEqual(pmids[-1], "3999")
        self.assertEqual(missing_count, 0)
        self.assertEqual(oversized_days, 0)
        self.assertEqual(len(slices), 3)

    def test_pubmed_xml_is_normalized_without_losing_nested_text(self):
        abstract_filler = " A" * 300
        article = ElementTree.fromstring(
            f"""
            <PubmedArticle>
              <MedlineCitation>
                <PMID>101</PMID>
                <Article>
                  <ArticleTitle>Robotic <i>rectal</i> resection</ArticleTitle>
                  <Abstract>
                    <AbstractText Label="BACKGROUND">A <b>useful</b> abstract.{abstract_filler}</AbstractText>
                  </Abstract>
                  <ArticleDate><Year>2020</Year><Month>Feb</Month><Day>3</Day></ArticleDate>
                  <Journal><Title>Surgical Journal</Title></Journal>
                  <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
                  <Language>eng</Language>
                </Article>
                <MeshHeadingList><MeshHeading><DescriptorName>Rectal Neoplasms</DescriptorName></MeshHeading></MeshHeadingList>
              </MedlineCitation>
              <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData>
            </PubmedArticle>
            """
        )

        result, exclusion_reason = parse_pubmed_article(article)

        self.assertIsNone(exclusion_reason)
        self.assertEqual(result["id"], "PMID:101")
        self.assertEqual(result["publication_date"], "2020-02-03")
        self.assertEqual(result["display_name"], "Robotic rectal resection")
        self.assertTrue(result["abstract"].startswith("BACKGROUND: A useful abstract."))
        self.assertEqual(result["doi"], "10.1/example")

    def test_post_fetch_filters_return_specific_exclusion_reasons(self):
        def article_xml(
            *,
            pmid="101",
            title="A valid surgical title",
            abstract="A" * 301,
            publication_date="<Year>2020</Year>",
            language="eng",
            publication_type="Journal Article",
        ):
            return ElementTree.fromstring(
                f"""
                <PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
                <ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText>{abstract}</AbstractText></Abstract>
                <Journal><JournalIssue><PubDate>{publication_date}</PubDate></JournalIssue><Title>Journal</Title></Journal>
                <PublicationTypeList><PublicationType>{publication_type}</PublicationType></PublicationTypeList>
                <Language>{language}</Language>
                </Article></MedlineCitation><PubmedData /></PubmedArticle>
                """
            )

        cases = [
            ({"pmid": ""}, "missing_pmid"),
            ({"title": ""}, "missing_title"),
            ({"abstract": ""}, "missing_abstract"),
            ({"publication_date": ""}, "missing_publication_date"),
            ({"title": "short"}, "title_too_short"),
            ({"abstract": "A" * 300}, "abstract_too_short"),
            ({"abstract": "A" * 4_000}, "abstract_too_long"),
            ({"language": "ger"}, "non_english"),
            ({"publication_type": "Review"}, "excluded_publication_type"),
        ]
        for overrides, expected_reason in cases:
            record, reason = parse_pubmed_article(article_xml(**overrides))
            self.assertIsNone(record)
            self.assertEqual(reason, expected_reason)

    def test_history_and_fetch_paths_detect_incomplete_records(self):
        class FakeHistoryClient:
            def get(self, endpoint, params):
                self.endpoint = endpoint
                self.params = params
                return type("Response", (), {"text": "101\n102\n"})()

        history_client = FakeHistoryClient()
        self.assertEqual(
            history_pmids(
                history_client,
                count=2,
                webenv="webenv",
                query_key="1",
                limit=0,
            ),
            ["101", "102"],
        )
        self.assertEqual(history_client.endpoint, "efetch.fcgi")
        self.assertEqual(history_client.params["query_key"], "1")
        self.assertEqual(history_client.params["rettype"], "uilist")

        class SparseHistoryClient:
            def get(self, endpoint, params):
                return type("Response", (), {"text": "101\n"})()

        # PubMed can omit an unavailable record from a UID-list page. Continue
        # with the returned PMID rather than failing the whole corpus run.
        self.assertEqual(
            history_pmids(
                SparseHistoryClient(),
                count=2,
                webenv="webenv",
                query_key="1",
                limit=0,
            ),
            ["101"],
        )

        class FakeResponse:
            content = (
                b"""
                <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>101</PMID>
                <Article><ArticleTitle>A sufficiently long title</ArticleTitle><Abstract><AbstractText>"""
                + b"A" * 301
                + b"""</AbstractText></Abstract>
                <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue><Title>Journal</Title></Journal>
                <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList><Language>eng</Language>
                </Article></MedlineCitation><PubmedData /></PubmedArticle></PubmedArticleSet>
                """
            )

        class FakeFetchClient:
            def get(self, endpoint, params):
                self.endpoint = endpoint
                self.params = params
                return FakeResponse()

        records, unresolved, excluded_pmids_by_reason, exclusion_counts = fetch_articles(
            FakeFetchClient(), ["101", "102"]
        )
        self.assertEqual(set(records), {"101"})
        self.assertEqual(unresolved, {"102"})
        self.assertEqual(excluded_pmids_by_reason, {})
        self.assertEqual(exclusion_counts, {})
        self.assertFalse(abstract_length_is_eligible("A" * 300))
        self.assertTrue(abstract_length_is_eligible("A" * 301))
        self.assertFalse(abstract_length_is_eligible("A" * 4_000))

    def test_malformed_efetch_batch_is_split_without_aborting_the_run(self):
        valid_article = (
            b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>101</PMID>"
            b"<Article><ArticleTitle>A sufficiently long title</ArticleTitle>"
            b"<Abstract><AbstractText>"
            + b"A" * 301
            + b"</AbstractText></Abstract><Journal><JournalIssue><PubDate>"
            b"<Year>2020</Year></PubDate></JournalIssue><Title>Journal</Title>"
            b"</Journal><PublicationTypeList><PublicationType>Journal Article"
            b"</PublicationType></PublicationTypeList><Language>eng</Language>"
            b"</Article></MedlineCitation><PubmedData /></PubmedArticle></PubmedArticleSet>"
        )

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class FakeFetchClient:
            max_retries = 1

            def get(self, endpoint, params):
                if params["id"] == "101":
                    return FakeResponse(valid_article)
                return FakeResponse(b"<PubmedArticleSet>")

        records, unresolved, excluded_pmids_by_reason, exclusion_counts = fetch_articles(
            FakeFetchClient(), ["101", "102"]
        )

        self.assertEqual(set(records), {"101"})
        self.assertEqual(unresolved, set())
        self.assertEqual(excluded_pmids_by_reason, {"unparseable_xml": {"102"}})
        self.assertEqual(exclusion_counts, {"unparseable_xml": 1})


@unittest.skip("Superseded by TestSimpleLlmExtractor after the extractor rewrite.")
class _LegacyLlmResponseValidation(unittest.TestCase):
    def test_test_flag_limits_the_run_to_twenty_records(self):
        test_limit = 20
        rows = [
            {
                "id": f"PMID:{number}",
                "pmid": str(number),
                "abstract": f"Abstract {number}",
            }
            for number in range(1, TEST_RECORD_LIMIT + 6)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "test-output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            pd.DataFrame(rows[:test_limit]).assign(llm_concepts="[]").to_csv(
                output_path, index=False
            )

            result = CliRunner().invoke(
                extract_llm_concepts,
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--model",
                    "test-model",
                    "--test",
                    "--test-limit",
                    str(test_limit),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(
                f"Test mode: limiting extraction to the first {test_limit}",
                result.output,
            )
            self.assertEqual(len(pd.read_csv(output_path)), test_limit)
            metadata = json.loads(
                output_path.with_suffix(output_path.suffix + ".metadata.json").read_text()
            )
            self.assertTrue(metadata["test_mode"])
            self.assertEqual(metadata["test_limit"], test_limit)
            self.assertEqual(metadata["input_records_loaded"], test_limit)

    def test_concurrent_batches_checkpoint_and_record_metadata(self):
        rows = [
            {
                "id": f"PMID:{number}",
                "pmid": str(number),
                "abstract": f"Abstract {number}",
            }
            for number in range(1, 5)
        ]

        def fake_extract(task, **kwargs):
            concepts = {
                record["pmid"]: ["validated surgical concept"]
                for record in task.records
            }
            return task, BatchResponse(
                concepts_by_pmid=concepts,
                telemetry=BatchTelemetry(
                    duration_seconds=0.25,
                    request_attempts=1,
                    rate_limit_retries=0,
                    other_retries=0,
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        reported_responses=1,
                    ),
                ),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)

            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_task",
                side_effect=fake_extract,
            ):
                result = CliRunner().invoke(
                    extract_llm_concepts,
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model",
                        "test-model",
                        "--batch-size",
                        "2",
                        "--workers",
                        "2",
                        "--checkpoint-every",
                        "3",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            output = pd.read_csv(output_path)
            self.assertEqual(len(output), 4)
            self.assertTrue(output["llm_concepts"].notna().all())
            metadata = json.loads(
                output_path.with_suffix(output_path.suffix + ".metadata.json").read_text()
            )
            execution = metadata["execution"]
            self.assertEqual(execution["status"], "complete")
            self.assertEqual(execution["workers"], 2)
            self.assertEqual(execution["checkpoint_every_abstracts"], 3)
            self.assertEqual(len(execution["batches"]), 2)
            self.assertEqual(execution["summary"]["usage"]["total_tokens"], 30)

    def test_failed_batch_is_logged_then_retried_by_resume(self):
        rows = [
            {
                "id": f"PMID:{number}",
                "pmid": str(number),
                "abstract": f"Abstract {number}",
            }
            for number in range(1, 5)
        ]

        def success(task, **kwargs):
            return task, BatchResponse(
                concepts_by_pmid={
                    record["pmid"]: ["validated surgical concept"]
                    for record in task.records
                },
                telemetry=BatchTelemetry(
                    duration_seconds=0.1,
                    request_attempts=1,
                    rate_limit_retries=0,
                    other_retries=0,
                    usage=TokenUsage(),
                ),
            )

        def fail_first_batch(task, **kwargs):
            if task.number == 1:
                raise ExtractionError("simulated malformed model response")
            return success(task, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            common_args = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--model",
                "test-model",
                "--batch-size",
                "2",
                "--workers",
                "1",
                "--checkpoint-every",
                "1",
            ]

            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_task",
                side_effect=fail_first_batch,
            ):
                first_run = CliRunner().invoke(extract_llm_concepts, common_args)

            self.assertEqual(first_run.exit_code, 0, first_run.output)
            first_output = pd.read_csv(output_path)
            self.assertEqual(first_output["llm_concepts"].isna().sum(), 2)
            first_metadata = json.loads(
                output_path.with_suffix(output_path.suffix + ".metadata.json").read_text()
            )
            self.assertEqual(first_metadata["execution"]["status"], "complete_with_failures")
            self.assertEqual(
                first_metadata["execution"]["failures"]["failed_pmids"], ["1", "2"]
            )

            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_task",
                side_effect=success,
            ):
                resumed_run = CliRunner().invoke(extract_llm_concepts, common_args)

            self.assertEqual(resumed_run.exit_code, 0, resumed_run.output)
            resumed_output = pd.read_csv(output_path)
            self.assertTrue(resumed_output["llm_concepts"].notna().all())

    def test_rate_limit_retry_records_usage_and_honours_retry_after(self):
        class FakeResponse:
            def __init__(self, *, status_code, body, retry_after=None):
                self.status_code = status_code
                self._body = body
                self.headers = {} if retry_after is None else {"Retry-After": retry_after}
                self.text = json.dumps(body)

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(response=self)

            def json(self):
                return self._body

        success = FakeResponse(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "pmid": "101",
                                            "concepts": ["hepatic vascular exclusion"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

        class FakeSession:
            def __init__(self):
                self.responses = [
                    FakeResponse(
                        status_code=429,
                        body={"error": {"code": "rate_limit_exceeded"}},
                        retry_after="0",
                    ),
                    success,
                ]

            def post(self, url, **kwargs):
                return self.responses.pop(0)

        with patch("materials_concepts.dataset.preparation.extract_llm_concepts.time.sleep") as sleep:
            response = request_batch_with_telemetry(
                FakeSession(),
                base_url="https://example.test/v1",
                api_key=None,
                model="test-model",
                records=[{"pmid": "101", "abstract": "An abstract."}],
                response_format="json_schema",
                timeout=10,
                max_retries=2,
                retry_delay_seconds=0,
            )

        self.assertEqual(response.concepts_by_pmid["101"], ["hepatic vascular exclusion"])
        self.assertEqual(response.telemetry.request_attempts, 2)
        self.assertEqual(response.telemetry.rate_limit_retries, 1)
        self.assertEqual(response.telemetry.total_retries, 1)
        self.assertEqual(response.telemetry.usage.reasoning_tokens, 3)
        sleep.assert_called_once_with(0.0)

    def test_misaligned_multi_record_response_is_not_resubmitted(self):
        def response_for(pmids):
            class FakeResponse:
                status_code = 200
                headers = {}

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "items": [
                                                {
                                                    "pmid": pmid,
                                                    "concepts": [
                                                        "hepatic vascular exclusion"
                                                    ],
                                                }
                                                for pmid in pmids
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    }

            return FakeResponse()

        class FakeSession:
            def __init__(self):
                self.responses = [
                    response_for([str(number) for number in range(1, 10)]),
                ]
                self.request_count = 0

            def post(self, url, **kwargs):
                self.request_count += 1
                return self.responses.pop(0)

        session = FakeSession()
        with self.assertRaises(ExtractionError):
            request_batch_with_telemetry(
                session,
                base_url="https://example.test/v1",
                api_key=None,
                model="test-model",
                records=[
                    {"pmid": str(number), "abstract": f"Abstract {number}"}
                    for number in range(1, 11)
                ],
                response_format="json_schema",
                timeout=10,
                max_retries=2,
                retry_delay_seconds=0,
            )

        self.assertEqual(session.request_count, 1)

    def test_request_uses_openai_compatible_chat_completion_payload(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {
                                                "pmid": "101",
                                                "concepts": [
                                                    "postoperative fistula"
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }

        class FakeSession:
            def post(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs
                return FakeResponse()

        session = FakeSession()
        result = request_batch(
            session,
            base_url="https://example.test/v1/",
            api_key="test-key",
            model="test-model",
            records=[{"pmid": "101", "abstract": "An abstract."}],
            response_format="json_schema",
            timeout=10,
            max_retries=1,
            retry_delay_seconds=0,
        )

        self.assertEqual(result, {"101": ["postoperative fistula"]})
        self.assertEqual(session.url, "https://example.test/v1/chat/completions")
        self.assertEqual(session.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(session.kwargs["json"]["model"], "test-model")
        self.assertEqual(
            session.kwargs["json"]["response_format"]["type"], "json_schema"
        )

    def test_valid_response_is_normalized_and_generic_labels_removed(self):
        payload = {
            "items": [
                {
                    "pmid": "101",
                    "concepts": [
                        "Postoperative Pancreatic Fistula",
                        "study",
                        "pcr",
                        "postoperative pancreatic fistula",
                    ],
                },
                {"pmid": "102", "concepts": []},
            ]
        }

        result = validate_response(payload, ["101", "102"])

        self.assertEqual(result["101"], ["postoperative pancreatic fistula", "pcr"])
        self.assertEqual(result["102"], [])

    def test_invalid_labels_are_dropped_without_resubmitting_a_batch(self):
        class FakeResponse:
            def __init__(self, concepts):
                self.concepts = concepts

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {"pmid": "101", "concepts": self.concepts}
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }

        class FakeSession:
            def __init__(self):
                self.responses = [
                    FakeResponse(
                        ["trisectionectomy", "hepatic trisectionectomy"]
                    ),
                ]
                self.requests = []

            def post(self, url, **kwargs):
                self.requests.append(kwargs["json"])
                return self.responses.pop(0)

        session = FakeSession()
        result = request_batch(
            session,
            base_url="https://example.test/v1",
            api_key=None,
            model="test-model",
            records=[{"pmid": "101", "abstract": "An abstract."}],
            response_format="json_object",
            timeout=10,
            max_retries=2,
            retry_delay_seconds=0,
        )

        self.assertEqual(result, {"101": ["hepatic trisectionectomy"]})
        self.assertEqual(len(session.requests), 1)

    def test_persistently_invalid_labels_are_dropped_not_fatal(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {
                                                "pmid": "101",
                                                "concepts": [
                                                    "trisectionectomy",
                                                    "hepatic vascular exclusion",
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }

        class FakeSession:
            def post(self, url, **kwargs):
                return FakeResponse()

        result = request_batch(
            FakeSession(),
            base_url="https://example.test/v1",
            api_key=None,
            model="test-model",
            records=[{"pmid": "101", "abstract": "An abstract."}],
            response_format="json_object",
            timeout=10,
            max_retries=1,
            retry_delay_seconds=0,
        )

        self.assertEqual(result, {"101": ["hepatic vascular exclusion"]})

    def test_misaligned_response_is_rejected(self):
        payload = {"items": [{"pmid": "unexpected", "concepts": []}]}
        with self.assertRaises(ExtractionError):
            validate_response(payload, ["101"])

    def test_json_schema_mode_contains_the_validation_schema(self):
        payload = response_format_payload("json_schema")
        self.assertEqual(payload["json_schema"]["schema"], RESPONSE_SCHEMA)
        self.assertTrue(payload["json_schema"]["strict"])


class TestSimpleLlmExtractor(unittest.TestCase):
    def test_test_mode_and_flags_use_their_passed_values(self):
        rows = [
            {"id": f"PMID:{number}", "pmid": str(number), "abstract": "Abstract"}
            for number in range(1, 26)
        ]
        test_limit = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            pd.DataFrame(rows[:test_limit]).assign(llm_concepts="[]").to_csv(
                output_path, index=False
            )

            result = CliRunner().invoke(
                extract_llm_concepts,
                [
                    "--input", str(input_path),
                    "--output", str(output_path),
                    "--model", "test-model",
                    "--test", "--test-limit", str(test_limit),
                    "--batch-size", "2", "--workers", "1",
                    "--checkpoint-every", "3", "--max-concepts", "5",
                    "--max-attempts", "1",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Test mode: limiting extraction to the first 7", result.output)
            self.assertEqual(len(pd.read_csv(output_path)), test_limit)
            metadata = json.loads(
                output_path.with_suffix(output_path.suffix + ".metadata.json").read_text()
            )
            self.assertTrue(metadata["test_mode"])
            self.assertEqual(metadata["test_limit"], test_limit)
            self.assertEqual(metadata["batch_size"], 2)
            self.assertEqual(metadata["workers"], 1)
            self.assertEqual(metadata["checkpoint_every_abstracts"], 3)
            self.assertEqual(metadata["max_concepts"], 5)
            self.assertEqual(metadata["max_attempts_per_batch"], 1)

    def test_invalid_batch_is_logged_and_resume_retries_only_blank_rows(self):
        rows = [
            {"id": f"PMID:{number}", "pmid": str(number), "abstract": "Abstract"}
            for number in range(1, 5)
        ]

        def failing_first_batch(batch, **kwargs):
            if batch.number == 1:
                raise InvalidResponseError("missing PMID")
            return batch, {
                record["pmid"]: ["hepatic vascular exclusion"]
                for record in batch.records
            }

        def succeeding_batch(batch, **kwargs):
            return batch, {
                record["pmid"]: ["hepatic vascular exclusion"]
                for record in batch.records
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            args = [
                "--input", str(input_path), "--output", str(output_path),
                "--model", "test-model", "--batch-size", "2", "--workers", "1",
                "--checkpoint-every", "1",
            ]

            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_batch",
                side_effect=failing_first_batch,
            ):
                first = CliRunner().invoke(extract_llm_concepts, args)

            self.assertEqual(first.exit_code, 0, first.output)
            first_output = pd.read_csv(output_path)
            self.assertEqual(first_output["llm_concepts"].isna().sum(), 2)
            failures = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
            self.assertTrue(failures.exists())
            self.assertIn('"pmids": ["1", "2"]', failures.read_text())

            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_batch",
                side_effect=succeeding_batch,
            ) as extract:
                second = CliRunner().invoke(extract_llm_concepts, args)

            self.assertEqual(second.exit_code, 0, second.output)
            self.assertEqual(extract.call_count, 1)
            self.assertTrue(pd.read_csv(output_path)["llm_concepts"].notna().all())

    def test_batches_are_fixed_and_not_split(self):
        rows = [
            {"id": f"PMID:{number}", "pmid": str(number), "abstract": "Abstract"}
            for number in range(1, 6)
        ]
        seen_pmids = []

        def success(batch, **kwargs):
            seen_pmids.append([record["pmid"] for record in batch.records])
            return batch, {
                record["pmid"]: ["hepatic vascular exclusion"]
                for record in batch.records
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            output_path = tmp / "output.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            with patch(
                "materials_concepts.dataset.preparation.extract_llm_concepts.extract_batch",
                side_effect=success,
            ):
                result = CliRunner().invoke(
                    extract_llm_concepts,
                    [
                        "--input", str(input_path), "--output", str(output_path),
                        "--model", "test-model", "--batch-size", "2", "--workers", "1",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(seen_pmids, [["1", "2"], ["3", "4"], ["5"]])

    def test_malformed_model_response_is_not_retried(self):
        class FakeResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": json.dumps({"items": []})}}]
                }

        with patch(
            "materials_concepts.dataset.preparation.extract_llm_concepts.requests.post",
            return_value=FakeResponse(),
        ) as post:
            with self.assertRaises(InvalidResponseError):
                request_batch(
                    base_url="https://example.test/v1",
                    api_key=None,
                    model="test-model",
                    records=[
                        {"pmid": "101", "abstract": "An abstract."},
                        {"pmid": "102", "abstract": "Another abstract."},
                    ],
                    response_format="json_schema",
                    timeout=10,
                    max_attempts=2,
                    retry_delay_seconds=0,
                    max_concepts=12,
                )
        self.assertEqual(post.call_count, 1)

    def test_request_payload_and_strict_validation(self):
        class FakeResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "items": [
                                            {
                                                "pmid": "101",
                                                "concepts": ["hepatic vascular exclusion"],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }

        with patch(
            "materials_concepts.dataset.preparation.extract_llm_concepts.requests.post",
            return_value=FakeResponse(),
        ) as post:
            result = request_batch(
                base_url="https://example.test/v1/",
                api_key="test-key",
                model="test-model",
                records=[{"pmid": "101", "abstract": "An abstract."}],
                response_format="json_schema",
                timeout=10,
                max_attempts=1,
                retry_delay_seconds=0,
                max_concepts=12,
            )

        self.assertEqual(result, {"101": ["hepatic vascular exclusion"]})
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["json"]["response_format"]["type"], "json_schema")
        cleaned = validate_response(
            {
                "items": [
                    {
                        "pmid": "101",
                        "concepts": [
                            "INR threshold for removal",
                            "proctectomy",
                            "MEFV mutations",
                            "study",
                            "weighted Likert scales",
                            "INR threshold for removal",
                        ],
                    }
                ]
            },
            ["101"],
            12,
        )
        self.assertEqual(
            cleaned["101"],
            ["inr threshold for removal", "mefv mutations", "weighted likert scales"],
        )
        self.assertEqual(
            response_format_payload("json_schema", 12)["json_schema"]["schema"]["properties"]["items"]["items"]["properties"]["concepts"]["maxItems"],
            12,
        )


class TestBiomedicalGraphCompatibility(unittest.TestCase):
    def test_biomedical_filter_keeps_numbers_unicode_and_approved_abbreviations(self):
        filtered = BiomedicalConceptFilter(min_n=2, max_n=6)(
            {
                "type 2 diabetes": 3,
                "α-fetoprotein level": 3,
                "pcr": 3,
                "proctectomy": 3,
                "a seven word label that exceeds the configured limit": 3,
            }
        )

        self.assertEqual(
            set(filtered),
            {"type 2 diabetes", "α-fetoprotein level", "pcr"},
        )

    def test_llm_concepts_build_a_concept_only_graph(self):
        works = pd.DataFrame(
            [
                {
                    "id": "PMID:101",
                    "publication_date": "2019-01-01",
                    "llm_concepts": json.dumps(
                        [
                            "postoperative fistula",
                            "drain amylase level",
                            "type 2 diabetes",
                            "α-fetoprotein level",
                            "pcr",
                            "proctectomy",
                        ]
                    ),
                },
                {
                    "id": "PMID:102",
                    "publication_date": "2020-01-01",
                    "llm_concepts": json.dumps(
                        ["postoperative fistula", "pancreatic gland texture"]
                    ),
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "works.csv"
            graph_path = tmp / "edges.pkl"
            lookup_path = tmp / "lookup.csv"
            works.to_csv(input_path, index=False)

            build_graph(
                input_path=str(input_path),
                output_path=str(graph_path),
                output_lookup_path=str(lookup_path),
                colname="llm_concepts",
                min_occurence=1,
                min_words=2,
                max_words=6,
                min_length=2,
                include_elements=False,
            )

            with open(graph_path, "rb") as graph_file:
                graph = pickle.load(graph_file)

            self.assertTrue(graph["include_elements"] is False)
            self.assertEqual(graph["colname"], "llm_concepts")
            self.assertGreater(len(graph["edges"]), 0)

            lookup = pd.read_csv(lookup_path)
            self.assertTrue(
                {"type 2 diabetes", "α-fetoprotein level", "pcr"}.issubset(
                    set(lookup.concept)
                )
            )
            self.assertNotIn("proctectomy", set(lookup.concept))

    def test_embedding_preparation_accepts_llm_concepts_without_elements(self):
        works = pd.DataFrame(
            {
                "id": ["PMID:101"],
                "publication_date": ["2020-01-01"],
                "abstract": ["The abstract."],
                "llm_concepts": [json.dumps(["postoperative fistula"])],
            }
        )
        lookup = pd.DataFrame({"id": [0], "concept": ["postoperative fistula"]})

        result = prepare_dataframe(
            works,
            lookup,
            cols=["id", "concepts", "publication_date"],
            concept_column="llm_concepts",
        )

        self.assertEqual(result.iloc[0].concepts, ["postoperative fistula"])


class TestTemporalDataSampling(unittest.TestCase):
    def test_stratified_evaluation_sampling_has_both_classes(self):
        generator = object.__new__(DataGenerator)
        generator.verbose = False
        generator.year_start = 2019

        class GraphStub:
            @staticmethod
            def get_vertices(**_kwargs):
                return np.arange(100)

        generator.graph = GraphStub()
        positives = [(index, index + 1) for index in range(20)]
        generator._get_pos_samples = lambda _vertices, _min_links: positives
        generator._get_neg_samples = lambda count, _vertices: [
            (100 + index, 200 + index) for index in range(count)
        ]

        _, labels = generator._generate_test(
            edges_used=100,
            min_links=1,
            max_v_degree=None,
            test_positive_ratio=0.1,
        )

        self.assertEqual(Counter(labels), Counter({0: 90, 1: 10}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
