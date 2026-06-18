import csv
import json
import tempfile
import unittest
from pathlib import Path

from summarize_news import (
    LLMResult,
    PipelineError,
    build_output,
    parse_json_object,
    read_news,
    save_json,
    validate_summary,
)


class FakeClient:
    def summarize(self, article):
        return LLMResult(
            data={
                "summary": f"Кратко: {article['title']}",
                "key_points": ["Первый факт", "Второй факт"],
                "category": "технологии",
            },
            response_model="test-model",
            usage={"total_tokens": 10},
        )


class JsonParsingTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        content = """```json
{"summary":"Текст","key_points":["A","B"],"category":"наука"}
```"""
        parsed = parse_json_object(content)
        self.assertEqual(parsed["category"], "наука")

    def test_extracts_json_from_extra_text(self):
        parsed = parse_json_object('Ответ: {"ok": true}')
        self.assertTrue(parsed["ok"])

    def test_rejects_invalid_key_points(self):
        with self.assertRaises(PipelineError):
            validate_summary(
                {
                    "summary": "Текст",
                    "key_points": ["Только один факт"],
                    "category": "наука",
                }
            )


class CsvTests(unittest.TestCase):
    def test_reads_valid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "news.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "id",
                        "published_at",
                        "title",
                        "text",
                        "source_url",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "1",
                        "published_at": "2025-01-01",
                        "title": "Заголовок",
                        "text": "Текст новости",
                        "source_url": "https://example.com",
                    }
                )

            records = read_news(path)
            self.assertEqual(records[0]["title"], "Заголовок")

    def test_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "news.csv"
            path.write_text("id,title\n1,Новость\n", encoding="utf-8")
            with self.assertRaises(PipelineError):
                read_news(path)


class PipelineTests(unittest.TestCase):
    def test_builds_and_saves_output(self):
        article = {
            "id": "1",
            "published_at": "2025-01-01",
            "title": "Новость",
            "text": "Текст",
            "source_url": "https://example.com",
        }
        result = build_output([article], FakeClient(), "requested-model")

        self.assertEqual(result["metadata"]["articles_count"], 1)
        self.assertEqual(result["metadata"]["usage"]["total_tokens"], 10)
        self.assertEqual(result["summaries"][0]["category"], "технологии")

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            save_json(result, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["summaries"][0]["id"], "1")


if __name__ == "__main__":
    unittest.main()
