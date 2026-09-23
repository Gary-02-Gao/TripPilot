import json
import unittest

from evaluate_qwen_fc import calls_from_message, calls_from_text, score_tool_calls


class FunctionCallParserTests(unittest.TestCase):
    def test_parses_qwen_tool_block(self):
        text = '<tool_call>\n{"name":"query_route","arguments":{"end_location":"上海博物馆"}}\n</tool_call>'
        calls, valid = calls_from_text(text)
        self.assertTrue(valid)
        self.assertEqual(calls[0]["name"], "query_route")
        self.assertEqual(calls[0]["arguments"]["end_location"], "上海博物馆")

    def test_rejects_malformed_tool_json(self):
        calls, valid = calls_from_text('<tool_call>{"name":"query_route",}</tool_call>')
        self.assertFalse(valid)
        self.assertEqual(calls, [])

    def test_rejects_non_object_tool_calls_without_crashing(self):
        for value in [[], None, "query_route", 1, {"function": "query_route"}]:
            with self.subTest(value=value):
                calls, valid = calls_from_text(f'<tool_call>{json.dumps(value)}</tool_call>')
                self.assertFalse(valid)
                self.assertEqual(calls, [])

    def test_rejects_non_string_tool_name(self):
        calls, valid = calls_from_text('<tool_call>{"name":123,"arguments":{}}</tool_call>')
        self.assertFalse(valid)
        self.assertFalse(calls[0]["valid_json"])

    def test_parses_nested_function_object(self):
        text = '<tool_call>{"function":{"name":"query_route","arguments":{"end_location":"上海博物馆"}}}</tool_call>'
        calls, valid = calls_from_text(text)
        self.assertTrue(valid)
        self.assertEqual(calls[0]["arguments"]["end_location"], "上海博物馆")

    def test_extra_malformed_call_prevents_strict_match(self):
        correct = '<tool_call>{"name":"query_route","arguments":{"end_location":"上海博物馆"}}</tool_call>'
        gold, _ = calls_from_text(correct)
        self.assertEqual(score_tool_calls(gold, gold, True), (True, True, True))
        for suffix in ['<tool_call>bad</tool_call>', '<tool_call>bad']:
            with self.subTest(suffix=suffix):
                calls, valid = calls_from_text(correct + suffix)
                self.assertFalse(valid)
                self.assertFalse(score_tool_calls(gold, calls, valid)[2])

    def test_normalizes_gold_argument_string(self):
        message = {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_travel_guide",
                        "arguments": '{"location":"嘉兴","search_mode":"hybrid"}',
                    }
                }
            ]
        }
        calls = calls_from_message(message)
        self.assertTrue(calls[0]["valid_json"])
        self.assertEqual(calls[0]["arguments"]["location"], "嘉兴")


if __name__ == "__main__":
    unittest.main()
