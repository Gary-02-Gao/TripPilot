#!/usr/bin/env python3
"""Teacher-forced Function Calling evaluation for base or fine-tuned Qwen models."""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def normalize_arguments(value: Any) -> tuple[bool, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            return False, parsed
        return True, parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, value


def calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        valid, arguments = normalize_arguments(function.get("arguments", {}))
        calls.append({"name": function.get("name"), "arguments": arguments, "valid_json": valid})
    return calls


def calls_from_text(text: str) -> tuple[list[dict[str, Any]], bool]:
    blocks = TOOL_BLOCK.findall(text)
    if not blocks and "<tool_call>" in text:
        blocks = [text.split("<tool_call>", 1)[1].split("<|im_end|>", 1)[0]]
    calls = []
    # Do not silently ignore a truncated extra call after a complete block.
    all_valid = text.count("<tool_call>") == len(blocks)
    for block in blocks:
        try:
            obj = json.loads(block.strip())
            if not isinstance(obj, dict):
                raise ValueError("tool call must be an object")
            function = obj.get("function", {})
            if not isinstance(function, dict):
                raise ValueError("function must be an object")
            valid_args, arguments = normalize_arguments(obj.get("arguments", {}))
            name = obj.get("name") or function.get("name")
            if "function" in obj and "arguments" not in obj:
                valid_args, arguments = normalize_arguments(function.get("arguments", {}))
            valid = isinstance(name, str) and bool(name.strip()) and valid_args
            calls.append({"name": name, "arguments": arguments, "valid_json": valid})
            all_valid = all_valid and valid
        except (TypeError, ValueError, json.JSONDecodeError):
            all_valid = False
    return calls, all_valid


def score_tool_calls(
    gold_calls: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
    predicted_json_valid: bool,
) -> tuple[bool, bool, bool]:
    names_ok = [item["name"] for item in gold_calls] == [item["name"] for item in predicted_calls]
    args_ok = len(gold_calls) == len(predicted_calls) and all(
        gold["arguments"] == predicted["arguments"]
        for gold, predicted in zip(gold_calls, predicted_calls)
    )
    return names_ok, args_ok, names_ok and args_ok and predicted_json_valid


def load_targets(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for row_index, row in enumerate(rows):
        messages = row.get("conversation") or row.get("messages") or []
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            targets.append(
                {
                    "row_index": row_index,
                    "message_index": message_index,
                    "prefix": messages[:message_index],
                    "gold": message,
                }
            )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Function Calling decisions and arguments")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=ROOT / "merged_test_final.json")
    parser.add_argument("--tools-file", type=Path, default=ROOT / "all_tools.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many assistant targets")
    parser.add_argument("--limit", type=int, default=0, help="Maximum assistant targets; 0 evaluates all")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tools = json.loads(args.tools_file.read_text(encoding="utf-8"))
    targets = load_targets(args.data_file)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    targets = targets[args.offset :]
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        raise ValueError("No assistant targets found")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto" if args.device_map_auto else None,
    )
    model.eval()

    supports_tools = "tools" in inspect.signature(tokenizer.apply_chat_template).parameters
    counts = {
        "targets": 0,
        "decision_correct": 0,
        "tool_targets": 0,
        "tool_name_sequence_correct": 0,
        "tool_arguments_correct": 0,
        "tool_call_exact": 0,
        "predicted_tool_outputs": 0,
        "valid_tool_json": 0,
        "reply_targets": 0,
        "nonempty_reply": 0,
    }
    records = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, target in enumerate(targets, start=1):
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        if supports_tools:
            template_kwargs["tools"] = tools
        input_ids = tokenizer.apply_chat_template(target["prefix"], **template_kwargs).to(model.device)
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids or None,
                pad_token_id=tokenizer.pad_token_id,
            )
        prediction = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=False)
        prediction = prediction.split("<|im_end|>", 1)[0].strip()
        predicted_calls, predicted_json_valid = calls_from_text(prediction)
        gold_calls = calls_from_message(target["gold"])
        gold_is_tool = bool(gold_calls)
        predicted_is_tool = bool(predicted_calls) or "<tool_call>" in prediction

        counts["targets"] += 1
        counts["decision_correct"] += int(gold_is_tool == predicted_is_tool)
        if gold_is_tool:
            counts["tool_targets"] += 1
            names_ok, args_ok, exact_ok = score_tool_calls(gold_calls, predicted_calls, predicted_json_valid)
            counts["tool_name_sequence_correct"] += int(names_ok)
            counts["tool_arguments_correct"] += int(args_ok)
            counts["tool_call_exact"] += int(exact_ok)
        else:
            counts["reply_targets"] += 1
            counts["nonempty_reply"] += int(bool(prediction) and not predicted_is_tool)
        if predicted_is_tool:
            counts["predicted_tool_outputs"] += 1
            counts["valid_tool_json"] += int(predicted_json_valid and bool(predicted_calls))

        records.append(
            {
                "row_index": target["row_index"],
                "message_index": target["message_index"],
                "gold_calls": gold_calls,
                "predicted_calls": predicted_calls,
                "prediction": prediction,
            }
        )
        if number % 25 == 0 or number == len(targets):
            print(f"evaluated {number}/{len(targets)}", flush=True)

    def ratio(numerator: str, denominator: str) -> float | None:
        return round(counts[numerator] / counts[denominator], 6) if counts[denominator] else None

    metrics = {
        **counts,
        "decision_accuracy": ratio("decision_correct", "targets"),
        "tool_name_sequence_accuracy": ratio("tool_name_sequence_correct", "tool_targets"),
        "tool_arguments_accuracy": ratio("tool_arguments_correct", "tool_targets"),
        "tool_call_exact_match": ratio("tool_call_exact", "tool_targets"),
        "valid_tool_json_rate": ratio("valid_tool_json", "predicted_tool_outputs"),
        "nonempty_reply_rate": ratio("nonempty_reply", "reply_targets"),
        "model_path": str(args.model_path.resolve()),
        "data_file": str(args.data_file.resolve()),
        "offset": args.offset,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
