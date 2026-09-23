import json
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


if __name__ == "__main__":
    unittest.main()
