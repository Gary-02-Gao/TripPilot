#!/usr/bin/env python3
"""Merge non-overlapping evaluate_qwen_fc.py shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COUNT_KEYS = (
    "targets",
    "decision_correct",
    "tool_targets",
    "tool_name_sequence_correct",
    "tool_arguments_correct",
    "tool_call_exact",
    "predicted_tool_outputs",
    "valid_tool_json",
    "reply_targets",
    "nonempty_reply",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shards = []
    for directory in args.shard:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        shards.append((metrics.get("offset", 0), directory, metrics))
    shards.sort(key=lambda item: item[0])

    merged = {key: sum(item[2][key] for item in shards) for key in COUNT_KEYS}

    def ratio(numerator: str, denominator: str) -> float | None:
        value = merged[denominator]
        return round(merged[numerator] / value, 6) if value else None

    merged.update(
        {
            "decision_accuracy": ratio("decision_correct", "targets"),
            "tool_name_sequence_accuracy": ratio("tool_name_sequence_correct", "tool_targets"),
            "tool_arguments_accuracy": ratio("tool_arguments_correct", "tool_targets"),
            "tool_call_exact_match": ratio("tool_call_exact", "tool_targets"),
            "valid_tool_json_rate": ratio("valid_tool_json", "predicted_tool_outputs"),
            "nonempty_reply_rate": ratio("nonempty_reply", "reply_targets"),
            "model_path": shards[0][2]["model_path"],
            "data_file": shards[0][2]["data_file"],
            "shards": len(shards),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for _, directory, _ in shards:
            output.write((directory / "predictions.jsonl").read_text(encoding="utf-8"))
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
