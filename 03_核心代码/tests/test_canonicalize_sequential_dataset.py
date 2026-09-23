import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from canonicalize_sequential_dataset import sequentialize


class SequentialDatasetTests(unittest.TestCase):
    def test_splits_guide_and_weather_and_exposes_days(self):
        messages = [
            {"role": "user", "content": "嘉兴三日游"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "function": {"name": "search_travel_guide", "arguments": '{"location":"嘉兴"}'}},
                    {"id": "b", "function": {"name": "get_weather_info", "arguments": '{"location":"330400","start_date":"2025-09-21","num_days":3}'}},
                ],
            },
        ]
        result = sequentialize(messages, {"330400": "嘉兴"})
        assistants = [item for item in result if item["role"] == "assistant"]
        self.assertEqual([call["function"]["name"] for call in assistants[0]["tool_calls"]], ["search_travel_guide"])
        self.assertEqual([call["function"]["name"] for call in assistants[1]["tool_calls"]], ["get_weather_info"])
        self.assertIn("3天", result[2]["content"])
        weather = json.loads(assistants[1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(weather["location"], "嘉兴")

    def test_removes_optional_route_city_code(self):
        messages = [
            {"role": "user", "content": "去车站"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "query_route",
                        "arguments": '{"start_location":"1,2","end_location":"车站","city_code":"123"}',
                    }
                }],
            },
        ]
        result = sequentialize(messages, {})
        arguments = json.loads(result[-1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"start_location": "1,2", "end_location": "车站"})

    def test_synthetic_examples_run_through_cli(self):
        core = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "canonicalized.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(core / "canonicalize_sequential_dataset.py"),
                    "--input", str(core / "examples" / "synthetic_conversations.json"),
                    "--mapping", str(core / "examples" / "city_code_mapping.json"),
                    "--output", str(output),
                    "--training",
                ],
                cwd=core.parent,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            self.assertEqual(json.loads(result.stdout), {"input_rows": 2, "output_rows": 3})
            rows = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 3)
        targets = [row["conversation"][-1]["tool_calls"][0]["function"] for row in rows]
        self.assertEqual(
            [target["name"] for target in targets],
            ["search_travel_guide", "get_weather_info", "query_route"],
        )
        self.assertEqual(
            json.loads(targets[0]["arguments"]),
            {"location": "嘉兴", "search_mode": "hybrid"},
        )
        self.assertEqual(
            json.loads(targets[1]["arguments"]),
            {"location": "嘉兴", "start_date": "2026-10-01", "num_days": 3},
        )
        weather_messages = rows[1]["conversation"]
        guide_call = weather_messages[-3]["tool_calls"][0]
        self.assertEqual(guide_call["function"]["name"], "search_travel_guide")
        self.assertEqual(weather_messages[-2]["role"], "tool")
        self.assertEqual(weather_messages[-2]["tool_call_id"], guide_call["id"])
        self.assertIn("3天", weather_messages[-2]["content"])
        self.assertEqual(
            json.loads(targets[2]["arguments"]),
            {"start_location": "121.473701,31.230416", "end_location": "上海博物馆"},
        )


if __name__ == "__main__":
    unittest.main()
