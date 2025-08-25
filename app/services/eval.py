from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from fastapi.testclient import TestClient


@dataclass
class EvalExample:
    question: str
    expected_keywords: List[str]
    filters: Dict[str, str] | None = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate_with_client(client: TestClient, dataset_path: str) -> Dict[str, Any]:
    """Run a lightweight evaluation over a JSONL dataset using the public API.

    Each JSONL line: {"question": str, "expected_keywords": [str], "filters": {optional}}
    Metrics (heuristic, 0..1):
      - relevance: fraction of expected_keywords present in the answer
      - citation_accuracy: 1.0 if citations present, else 0.0 (averaged)
      - groundedness: proxy == citation_accuracy (no chunk text here)
    """
    rows = load_jsonl(dataset_path)
    n = 0
    rel_sum = 0.0
    cite_sum = 0.0
    for row in rows:
        q = row["question"]
        exp = [str(x).lower() for x in row.get("expected_keywords", [])]
        payload: Dict[str, Any] = {"text": q}
        if row.get("filters"):
            payload["filters"] = row["filters"]
        r = client.post("/v1/query", json=payload)
        r.raise_for_status()
        data = r.json()
        answer = str(data.get("answer", "")).lower()
        citations = data.get("citations", []) or []
        # relevance as keyword recall in answer
        hits = sum(1 for k in exp if k in answer)
        rel = _safe_ratio(hits, max(1, len(exp)))
        rel_sum += rel
        # citations present
        cite = 1.0 if len(citations) > 0 else 0.0
        cite_sum += cite
        n += 1
    relevance = _safe_ratio(rel_sum, n)
    citation_accuracy = _safe_ratio(cite_sum, n)
    groundedness = citation_accuracy  # proxy until chunk text is accessible
    return {
        "count": n,
        "relevance": round(relevance, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "groundedness": round(groundedness, 4),
    }
