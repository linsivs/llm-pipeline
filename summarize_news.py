#!/usr/bin/env python3
"""Summarize news from CSV with an OpenAI-compatible LLM API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://text.pollinations.ai/openai"
DEFAULT_MODEL = "openai"
REQUIRED_COLUMNS = {"id", "published_at", "title", "text", "source_url"}
ALLOWED_CATEGORIES = {"технологии", "наука", "бизнес", "общество", "другое"}

SYSTEM_PROMPT = """
Ты редактор новостной ленты. Кратко перескажи переданную новость на русском
языке, не добавляя фактов, которых нет во входном тексте.

Верни только один JSON-объект без Markdown и пояснений:
{
  "summary": "краткое содержание в 1–2 предложениях",
  "key_points": ["факт 1", "факт 2", "факт 3"],
  "category": "технологии"
}

Требования:
- summary содержит не более 45 слов;
- key_points содержит от 2 до 4 коротких фактов;
- category принимает одно значение: технологии, наука, бизнес, общество, другое.
""".strip()


class PipelineError(RuntimeError):
    """Raised when input data or an API response cannot be processed."""


@dataclass(frozen=True)
class LLMResult:
    data: dict[str, Any]
    response_model: str
    usage: dict[str, Any]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding existing variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def read_news(path: Path) -> list[dict[str, str]]:
    """Read and validate news records from a CSV file."""
    if not path.exists():
        raise PipelineError(f"Входной файл не найден: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise PipelineError(f"В CSV отсутствуют обязательные столбцы: {names}")

        records: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            record = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            empty = [name for name, value in record.items() if not value]
            if empty:
                names = ", ".join(sorted(empty))
                raise PipelineError(f"Строка {row_number}: пустые поля: {names}")
            if record["id"] in seen_ids:
                raise PipelineError(
                    f"Строка {row_number}: повторяющийся id {record['id']!r}"
                )
            seen_ids.add(record["id"])
            records.append(record)

    if not records:
        raise PipelineError("Входной CSV не содержит новостей")
    return records


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating an optional Markdown code fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise PipelineError("LLM вернула ответ без JSON-объекта") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise PipelineError(f"Не удалось разобрать JSON от LLM: {error}") from error

    if not isinstance(parsed, dict):
        raise PipelineError("LLM должна вернуть JSON-объект")
    return parsed


def validate_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the structured LLM response."""
    summary = data.get("summary")
    key_points = data.get("key_points")
    category = data.get("category")

    if not isinstance(summary, str) or not summary.strip():
        raise PipelineError("В ответе LLM отсутствует непустое поле summary")
    if (
        not isinstance(key_points, list)
        or not 2 <= len(key_points) <= 4
        or any(not isinstance(item, str) or not item.strip() for item in key_points)
    ):
        raise PipelineError("Поле key_points должно содержать от 2 до 4 строк")
    if not isinstance(category, str) or category.strip().lower() not in ALLOWED_CATEGORIES:
        allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
        raise PipelineError(f"Недопустимая категория. Ожидается: {allowed}")

    return {
        "summary": summary.strip(),
        "key_points": [item.strip() for item in key_points],
        "category": category.strip().lower(),
    }


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 90,
        retries: int = 2,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def summarize(self, article: dict[str, str]) -> LLMResult:
        user_payload = {
            "published_at": article["published_at"],
            "title": article["title"],
            "text": article["text"],
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
        }
        response = self._post_json(payload)

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise PipelineError("API вернуло ответ неизвестного формата") from error
        if not isinstance(content, str):
            raise PipelineError("Поле content в ответе API должно быть строкой")

        parsed = validate_summary(parse_json_object(content))
        response_model = str(response.get("model") or self.model)
        usage = response.get("usage")
        return LLMResult(
            data=parsed,
            response_model=response_model,
            usage=usage if isinstance(usage, dict) else {},
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "news-summary-lab/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(self.api_url, data=body, headers=headers, method="POST")
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    decoded = response.read().decode("utf-8")
                parsed = json.loads(decoded)
                if not isinstance(parsed, dict):
                    raise PipelineError("API вернуло JSON, который не является объектом")
                return parsed
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                last_error = PipelineError(
                    f"API вернуло HTTP {error.code}: {details[:300]}"
                )
                if error.code < 500 and error.code != 429:
                    break
            except (URLError, TimeoutError) as error:
                last_error = PipelineError(f"Ошибка соединения с API: {error}")
            except json.JSONDecodeError as error:
                last_error = PipelineError(f"API вернуло некорректный JSON: {error}")

            if attempt < self.retries:
                time.sleep(2**attempt)

        assert last_error is not None
        raise last_error


def build_output(
    articles: list[dict[str, str]],
    client: OpenAICompatibleClient,
    requested_model: str,
) -> dict[str, Any]:
    """Process all articles and build the final JSON document."""
    summaries: list[dict[str, Any]] = []
    response_models: set[str] = set()
    total_usage: dict[str, int] = {}

    for index, article in enumerate(articles, start=1):
        print(
            f"[{index}/{len(articles)}] Обрабатывается: {article['title']}",
            file=sys.stderr,
        )
        result = client.summarize(article)
        response_models.add(result.response_model)
        for name, value in result.usage.items():
            if isinstance(value, int):
                total_usage[name] = total_usage.get(name, 0) + value

        summaries.append(
            {
                "id": article["id"],
                "published_at": article["published_at"],
                "title": article["title"],
                "source_url": article["source_url"],
                **result.data,
            }
        )

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "articles_count": len(summaries),
            "requested_model": requested_model,
            "response_models": sorted(response_models),
            "usage": total_usage,
        },
        "summaries": summaries,
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON atomically so an interrupted run does not corrupt the result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Суммаризация новостей из CSV через LLM API"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/news.csv"),
        help="путь к входному CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summaries.json"),
        help="путь к выходному JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="обработать только первые N новостей",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file(Path(".env"))
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("Ошибка: --limit должен быть положительным числом", file=sys.stderr)
        return 2

    try:
        articles = read_news(args.input)
        if args.limit is not None:
            articles = articles[: args.limit]

        api_url = os.getenv("LLM_API_URL", DEFAULT_API_URL)
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
        client = OpenAICompatibleClient(
            api_url=api_url,
            model=model,
            api_key=os.getenv("LLM_API_KEY", ""),
            timeout=float(os.getenv("LLM_TIMEOUT", "90")),
            retries=int(os.getenv("LLM_RETRIES", "2")),
        )
        output = build_output(articles, client, model)
        save_json(output, args.output)
    except (PipelineError, OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    print(f"Готово: {len(articles)} новостей сохранено в {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
