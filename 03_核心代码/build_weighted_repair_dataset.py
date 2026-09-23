#!/usr/bin/env python3
"""Build a deterministic, training-only repair set by upweighting rare tool routes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


WEIGHTS = {
    "query_route": 8,
    "recommend_hotels": 2,
    "search_travel_guide": 2,
    "search_travel_guide,get_weather_info": 2,
}


def last_assistant_sequence(row: dict) -> str:
    messages = row.get("conversation") or row.get("messages") or []
    assistant = next((item for item in reversed(messages) if item.get("role") == "assistant"), {})
    names = [
        (call.get("function") or {}).get("name", "")
        for call in assistant.get("tool_calls") or []
    ]
    return ",".join(name for name in names if name) or "reply"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    weighted = []
    original_counts = Counter()
    weighted_counts = Counter()
    for row in rows:
        sequence = last_assistant_sequence(row)
        original_counts[sequence] += 1
        copies = WEIGHTS.get(sequence, 1)
        weighted.extend([row] * copies)
        weighted_counts[sequence] += copies

    random.Random(args.seed).shuffle(weighted)
    args.output.write_text(json.dumps(weighted, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input_rows": len(rows),
        "output_rows": len(weighted),
        "weights": WEIGHTS,
        "original_counts": original_counts,
        "weighted_counts": weighted_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
