from __future__ import annotations

import unittest

from demo_agent import TravelDemoAgent


class TravelDemoAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TravelDemoAgent(travel_date="2026-10-01")

    def test_planning_uses_ordered_tool_chain(self) -> None:
        response = self.agent.chat("嘉兴三日游")
        self.assertEqual(response.workflow, "旅行规划")
        self.assertEqual(
            [item.name for item in response.tools],
            ["search_travel_guide", "get_weather_info"],
        )
        self.assertIn("嘉兴3日旅行计划", response.text)

    def test_missing_city_is_completed_in_next_turn(self) -> None:
        first = self.agent.chat("帮我规划一次旅行")
        self.assertTrue(first.needs_input)
        second = self.agent.chat("杭州")
        self.assertFalse(second.needs_input)
        self.assertEqual([item.name for item in second.tools], ["search_travel_guide", "get_weather_info"])

    def test_hotel_chain_recommends_then_reviews(self) -> None:
        response = self.agent.chat("推荐杭州500-800元的酒店")
        self.assertEqual(
            [item.name for item in response.tools],
            ["recommend_hotels", "get_hotel_reviews"],
        )

    def test_route_uses_single_tool(self) -> None:
        response = self.agent.chat("从外滩到上海博物馆怎么走")
        self.assertEqual([item.name for item in response.tools], ["query_route"])
        self.assertEqual(response.tools[0].arguments["end_location"], "上海博物馆")
        self.assertEqual(set(response.tools[0].arguments), {"start_location", "end_location"})

    def test_unrelated_request_is_refused(self) -> None:
        response = self.agent.chat("帮我写一段Python代码")
        self.assertEqual(response.workflow, "领域拒答")
        self.assertFalse(response.tools)


if __name__ == "__main__":
    unittest.main()
