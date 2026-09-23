"""Dependency-free offline implementation of the documented travel workflows.

This is a deterministic demo adapter, not a replacement for Qwen/LoRA.  It uses
the same five workflows and tool names as the training data so that the browser
demo can be recorded before a GPU or external API key is available.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
GUIDE_DIR = ROOT / "rag-system" / "original_data" / "travel_guides"


@dataclass
class ToolEvent:
    name: str
    arguments: Dict[str, Any]
    result_preview: str
    status: str = "success"


@dataclass
class AgentResponse:
    text: str
    workflow: str
    tools: List[ToolEvent]
    needs_input: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "workflow": self.workflow,
            "tools": [asdict(item) for item in self.tools],
            "needs_input": self.needs_input,
        }


class TravelDemoAgent:
    """Small deterministic controller mirroring the Safari workflow document."""

    def __init__(
        self,
        current_city: str = "上海",
        travel_date: Optional[str] = None,
        start_coordinates: str = "121.473701,31.230416",
    ) -> None:
        self.current_city = current_city
        self.travel_date = travel_date or (date.today() + timedelta(days=7)).isoformat()
        self.start_coordinates = start_coordinates
        self.pending: Optional[Dict[str, str]] = None
        self.history: List[Dict[str, str]] = []
        self.guide_index = self._build_guide_index()

    @staticmethod
    def _build_guide_index() -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        for path in GUIDE_DIR.glob("*_travel_guide.txt"):
            parts = path.stem.split("_", 1)
            if len(parts) != 2:
                continue
            city = parts[1].replace("_travel_guide", "")
            index[city] = path
            index[city.removesuffix("市")] = path
        return index

    def reset(self) -> None:
        self.pending = None
        self.history.clear()

    def chat(self, user_input: str) -> AgentResponse:
        query = user_input.strip()
        if not query:
            return AgentResponse("请告诉我你的旅行需求。", "信息补全", [], True)

        self.history.append({"role": "user", "content": query})
        if self.pending:
            response = self._continue_pending(query)
        else:
            intent = self._detect_intent(query)
            handlers = {
                "旅行规划": self._plan_trip,
                "路线导航": self._route,
                "酒店查询": self._hotel,
                "旅行闲聊": self._travel_chat,
                "领域拒答": self._reject,
            }
            response = handlers[intent](query)

        self.history.append({"role": "assistant", "content": response.text})
        self.history = self.history[-20:]
        return response

    def _continue_pending(self, query: str) -> AgentResponse:
        pending = self.pending or {}
        self.pending = None
        workflow = pending.get("workflow")
        original = pending.get("original", "")
        combined = f"{original} {query}".strip()
        if workflow == "旅行规划":
            return self._plan_trip(combined, fallback_city=query)
        if workflow == "路线导航":
            return self._route(combined, fallback_destination=query)
        if workflow == "酒店查询":
            return self._hotel(combined, fallback_city=query)
        return self._travel_chat(query)

    def _detect_intent(self, query: str) -> str:
        if re.search(r"酒店|住宿|宾馆|民宿|住哪|订房", query):
            return "酒店查询"
        if re.search(r"怎么走|路线|导航|如何到达|怎么去|坐车|地铁|公交|打车", query):
            return "路线导航"
        if re.search(r"旅游|旅行|游玩|[日天]游|攻略|景点|好玩|行程|度假|去哪|爬山", query):
            return "旅行规划"
        if re.search(r"你好|谢谢|再见|旅行安全|旅行预算|出行|行李|旅行助手", query):
            return "旅行闲聊"
        return "领域拒答"

    def _extract_city(self, text: str) -> Optional[str]:
        matches = [city for city in self.guide_index if city and city in text]
        if matches:
            return max(matches, key=len).removesuffix("市")
        if "附近" in text or "本地" in text:
            return self.current_city
        return None

    @staticmethod
    def _extract_days(text: str, default: int = 3) -> int:
        match = re.search(r"(1[0-5]|[1-9一二三四五六七八九十])\s*[天日]", text)
        if not match:
            return default
        value = match.group(1)
        chinese = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        return int(value) if value.isdigit() else chinese[value]

    @staticmethod
    def _extract_date(text: str, default: str) -> str:
        match = re.search(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})", text)
        if not match:
            return default
        try:
            return datetime(*map(int, match.groups())).date().isoformat()
        except ValueError:
            return default

    def _plan_trip(self, query: str, fallback_city: str = "") -> AgentResponse:
        city = self._extract_city(query) or self._extract_city(fallback_city)
        if not city:
            self.pending = {"workflow": "旅行规划", "original": query}
            return AgentResponse("请告诉我您想去哪个城市旅行？", "旅行规划", [], True)

        guide, guide_event = self._search_guide(city)
        if not guide:
            return AgentResponse(
                f"暂时没有找到{city}的旅行攻略。",
                "旅行规划",
                [guide_event],
            )

        days = self._extract_days(query, self._extract_days(guide[:1500], 3))
        days = min(days, 5)
        start_date = self._extract_date(query, self.travel_date)
        weather, weather_event = self._weather(city, start_date, days)
        plan_lines = self._extract_plan_lines(guide, days)

        lines = [f"## {city}{days}日旅行计划", ""]
        for idx in range(days):
            item = weather[idx]
            plan = plan_lines[idx] if idx < len(plan_lines) else "城市经典景点与街区漫游"
            lines.append(
                f"**第{idx + 1}天｜{item['date']}｜{item['day']} {item['low']}~{item['high']}℃**"
            )
            lines.append(f"{plan}。")
            lines.append("")
        lines.append("已按“攻略检索 → 天气查询 → 结果整合”的顺序完成工具链。")
        return AgentResponse("\n".join(lines), "旅行规划", [guide_event, weather_event])

    def _search_guide(self, city: str) -> Tuple[str, ToolEvent]:
        path = self.guide_index.get(city) or self.guide_index.get(f"{city}市")
        if not path:
            return "", ToolEvent(
                "search_travel_guide",
                {"location": city, "search_mode": "hybrid"},
                "未找到相关旅行攻略",
                "empty",
            )
        content = path.read_text(encoding="utf-8")
        return content, ToolEvent(
            "search_travel_guide",
            {"location": city, "search_mode": "hybrid"},
            f"命中本地攻略：{path.name}（{len(content)} 字符）",
        )

    @staticmethod
    def _extract_plan_lines(guide: str, limit: int) -> List[str]:
        results: List[str] = []
        for line in guide.splitlines():
            match = re.match(r"#{2,4}\s*第\s*\d+\s*天[：:]?\s*(.+)", line.strip())
            if match:
                results.append(match.group(1).strip())
                if len(results) >= limit:
                    break
        return results

    def _weather(self, city: str, start_date: str, days: int) -> Tuple[List[Dict[str, Any]], ToolEvent]:
        start = date.fromisoformat(start_date)
        conditions = ["晴", "多云", "阴", "小雨"]
        result = []
        for offset in range(days):
            seed = int(hashlib.sha256(f"{city}:{start_date}:{offset}".encode()).hexdigest()[:8], 16)
            low = 16 + seed % 9
            result.append(
                {
                    "date": (start + timedelta(days=offset)).isoformat(),
                    "day": conditions[seed % len(conditions)],
                    "low": low,
                    "high": low + 5 + seed % 4,
                }
            )
        preview = "；".join(
            f"{item['date']} {item['day']} {item['low']}~{item['high']}℃" for item in result
        )
        event = ToolEvent(
            "get_weather_info",
            {"location": city, "start_date": start_date, "num_days": days},
            preview,
        )
        return result, event

    def _route(self, query: str, fallback_destination: str = "") -> AgentResponse:
        destination = ""
        patterns = [
            r"从.+?到(.+?)(?:怎么走|路线|导航|[，。？?]|$)",
            r"(?:怎么去|去|到达)([^，。？?]+?)(?:怎么走|路线|导航|$)",
            r"([^，。？?]+?)(?:怎么走|在哪里)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                destination = match.group(1).strip()
                break
        if not destination and fallback_destination:
            destination = fallback_destination.strip(" ，。？?")
        destination = re.sub(r"(?:怎么走|的路线|导航)$", "", destination).strip()
        if not destination or destination in {"哪里", "哪", "机场", "医院", "学校"}:
            self.pending = {"workflow": "路线导航", "original": query}
            return AgentResponse("请问您要去哪里？", "路线导航", [], True)

        seed = int(hashlib.sha256(destination.encode()).hexdigest()[:8], 16)
        walk = 18 + seed % 35
        transit = 12 + seed % 24
        drive = 8 + seed % 18
        args = {
            "start_location": self.start_coordinates,
            "end_location": destination,
        }
        event = ToolEvent(
            "query_route",
            args,
            f"步行 {walk} 分钟；公交 {transit} 分钟；驾车 {drive} 分钟",
        )
        text = (
            f"## 前往{destination}的路线\n\n"
            f"- 公交/地铁：约 {transit} 分钟，适合优先选择。\n"
            f"- 驾车/打车：约 {drive} 分钟，行李较多时更方便。\n"
            f"- 步行：约 {walk} 分钟，距离合适时可选。\n\n"
            "路线结果来自模拟数据，起点为预设的上海坐标，仅用于展示工具调用流程。"
        )
        return AgentResponse(text, "路线导航", [event])

    def _hotel(self, query: str, fallback_city: str = "") -> AgentResponse:
        if re.search(r"怎么样|评价|好不好|值得住", query):
            match = re.search(r"([\u4e00-\u9fffA-Za-z0-9·]{2,20}(?:酒店|宾馆|民宿))", query)
            if not match:
                return AgentResponse("请问您想了解哪家酒店的评价？", "酒店查询", [], True)
            hotel_name = match.group(1)
            review = self._hotel_review(hotel_name)
            return AgentResponse(review.result_preview, "酒店查询", [review])

        city = self._extract_city(query) or self._extract_city(fallback_city)
        if not city:
            self.pending = {"workflow": "酒店查询", "original": query}
            return AgentResponse("请问您要在哪个城市找酒店？", "酒店查询", [], True)

        budget_match = re.search(r"(\d{2,5}(?:\s*[-~至]\s*\d{2,5})?)\s*元", query)
        budget = budget_match.group(1) + "元" if budget_match else "400-700元"
        hotel_name = f"{city}云栖酒店"
        recommend = ToolEvent(
            "recommend_hotels",
            {"requirements": f"{city}，预算{budget}"},
            json.dumps({"hotel_name": hotel_name, "location": f"{city}市中心", "price": budget}, ensure_ascii=False),
        )
        review = self._hotel_review(hotel_name)
        text = (
            f"## {city}酒店推荐\n\n"
            f"**{hotel_name}**\n\n"
            f"- 位置：{city}市中心，公共交通便利\n"
            f"- 参考预算：{budget}/晚\n"
            f"- 用户评价：位置方便、房间整洁；临街房型可能略有噪声，建议备注安静房。\n\n"
            "已按“酒店推荐 → 酒店评价 → 统一回复”的顺序完成工具链。"
        )
        return AgentResponse(text, "酒店查询", [recommend, review])

    @staticmethod
    def _hotel_review(hotel_name: str) -> ToolEvent:
        return ToolEvent(
            "get_hotel_reviews",
            {"hotel_name": hotel_name},
            f"{hotel_name}综合评分 4.4/5：位置方便、卫生良好；部分临街房间隔音一般。",
        )

    @staticmethod
    def _travel_chat(query: str) -> AgentResponse:
        if "谢谢" in query:
            text = "不客气！需要规划行程、查询路线或推荐酒店时，随时告诉我。"
        elif "功能" in query or "旅行助手" in query:
            text = "我可以规划旅行、查询路线、推荐酒店，并在信息不足时通过一轮追问补全关键条件。"
        else:
            text = "你好！告诉我目的地或具体旅行需求，我会帮你完成规划。"
        return AgentResponse(text, "旅行闲聊", [])

    @staticmethod
    def _reject(query: str) -> AgentResponse:
        return AgentResponse(
            "抱歉，我是专门的旅行助手，只能回答旅行规划、路线、酒店和出行相关问题。",
            "领域拒答",
            [],
        )
