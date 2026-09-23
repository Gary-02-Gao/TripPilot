#!/usr/bin/env python3
"""Create a learnable sequential Function Calling protocol from existing data."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


DOUBLE = ["search_travel_guide", "get_weather_info"]


def name_of(call: dict) -> str:
    return (call.get("function") or {}).get("name", "")


def args_of(call: dict) -> dict:
    value = (call.get("function") or {}).get("arguments", {})
    return json.loads(value) if isinstance(value, str) else copy.deepcopy(value)


def with_args(call: dict, arguments: dict) -> dict:
    result = copy.deepcopy(call)
    result.setdefault("function", {})["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result


def last_user(messages: list[dict]) -> str:
    return next(
        (item.get("content", "") for item in reversed(messages) if item.get("role") == "user"),
        "",
    )


def load_city_names(mapping_file: Path) -> dict[str, str]:
    payload = json.loads(mapping_file.read_text(encoding="utf-8"))["城市编码映射"]
    return {code: item["name"].removesuffix("市") for code, item in payload.items()}


def canonical_call(
    call: dict,
    messages: list[dict],
    code_to_name: dict[str, str],
    guide_city: str = "",
) -> dict:
    name = name_of(call)
    arguments = args_of(call)
    if name == "search_travel_guide":
        location = code_to_name.get(str(arguments.get("location", "")), str(arguments.get("location", "")))
        arguments = {"location": location.removesuffix("市"), "search_mode": "hybrid"}
    elif name == "get_weather_info":
        location = guide_city or code_to_name.get(
            str(arguments.get("location", "")), str(arguments.get("location", ""))
        )
        arguments["location"] = location.removesuffix("市")
        arguments["num_days"] = int(arguments.get("num_days", 1))
        arguments = {
            key: arguments[key]
            for key in ("location", "start_date", "num_days")
            if key in arguments
        }
    elif name == "query_route":
        arguments.pop("city_code", None)
        arguments = {
            key: arguments[key]
            for key in ("start_location", "end_location")
            if key in arguments
        }
    elif name == "recommend_hotels":
        requirement = re.sub(r"\s+", "", last_user(messages)).strip("，。！？? ")
        arguments = {"requirements": requirement}
    return with_args(call, arguments)


def sequentialize(messages: list[dict], code_to_name: dict[str, str]) -> list[dict]:
    output: list[dict] = []
    index = 0
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        calls = message.get("tool_calls") or []
        names = [name_of(call) for call in calls]
        if message.get("role") == "assistant" and names == DOUBLE:
            search = canonical_call(calls[0], output, code_to_name)
            guide_city = args_of(search).get("location", "")
            weather = canonical_call(calls[1], output, code_to_name, guide_city)
            first = copy.deepcopy(message)
            first["tool_calls"] = [search]
            output.append(first)
            days = args_of(weather).get("num_days", 1)
            if (
                index + 2 < len(messages)
                and messages[index + 1].get("role") == "tool"
                and messages[index + 2].get("role") == "tool"
            ):
                guide_result = copy.deepcopy(messages[index + 1])
                guide_result["content"] = (
                    f"攻略检索完成，目的地：{guide_city}，建议行程天数：{days}天"
                )
                output.append(guide_result)
                output.append({"role": "assistant", "content": "", "tool_calls": [weather]})
                output.append(copy.deepcopy(messages[index + 2]))
                index += 3
                continue
            output.append(
                {
                    "role": "tool",
                    "tool_call_id": search.get("id", "call_search_guide"),
                    "content": f"攻略检索完成，建议行程天数：{days}天",
                }
            )
            output.append({"role": "assistant", "content": "", "tool_calls": [weather]})
            index += 1
            continue
        if message.get("role") == "assistant" and calls:
            message["tool_calls"] = [
                canonical_call(call, output, code_to_name) for call in calls
            ]
        output.append(message)
        index += 1
    return output


def transform(
    rows: list[dict],
    code_to_name: dict[str, str],
    training: bool,
    weather_weight: int,
) -> list[dict]:
    result = []
    for row in rows:
        key = "conversation" if "conversation" in row else "messages"
        original = row[key]
        transformed = sequentialize(original, code_to_name)
        last = next(
            (item for item in reversed(original) if item.get("role") == "assistant"), {}
        )
        if training and [name_of(call) for call in last.get("tool_calls") or []] == DOUBLE:
            positions = [
                i for i, item in enumerate(transformed) if item.get("role") == "assistant"
            ]
            first, second = positions[-2:]
            result.append({key: transformed[: first + 1]})
            result.extend(
                {key: transformed[: second + 1]} for _ in range(weather_weight)
            )
        else:
            result.append({**row, key: transformed})
    return result


def write_canonical_tools(source: Path, output: Path) -> None:
    tools = json.loads(source.read_text(encoding="utf-8"))
    for tool in tools:
        function = tool["function"]
        schema = function["parameters"]
        schema["additionalProperties"] = False
        if function["name"] == "search_travel_guide":
            schema["required"] = ["location", "search_mode"]
        elif function["name"] == "get_weather_info":
            schema["required"] = ["location", "start_date", "num_days"]
            schema["properties"]["location"]["description"] = "目的地城市名称，如'北京'、'上海'"
        elif function["name"] == "query_route":
            schema["properties"].pop("city_code", None)
            schema["required"] = ["start_location", "end_location"]
    output.write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--weather-weight", type=int, default=1)
    parser.add_argument("--tools-input", type=Path)
    parser.add_argument("--tools-output", type=Path)
    args = parser.parse_args()
    code_to_name = load_city_names(args.mapping)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.weather_weight < 1:
        raise ValueError("--weather-weight must be at least 1")
    transformed = transform(rows, code_to_name, args.training, args.weather_weight)
    args.output.write_text(json.dumps(transformed, ensure_ascii=False), encoding="utf-8")
    if args.tools_input and args.tools_output:
        write_canonical_tools(args.tools_input, args.tools_output)
    print(json.dumps({"input_rows": len(rows), "output_rows": len(transformed)}))


if __name__ == "__main__":
    main()
