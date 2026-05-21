from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from disambiguation.object_vocabulary import GENERIC_OBJECTS

ATTRIBUTE_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "small",
    "large",
    "big",
    "tiny",
    "wide",
    "thin",
    "left",
    "right",
    "top",
    "bottom",
    "front",
    "back",
    "behind",
    "center",
    "middle",
}

RELATION_WORDS = {
    "behind",
    "front",
    "left",
    "right",
    "top",
    "bottom",
    "under",
    "above",
    "below",
    "beside",
    "near",
    "next",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        return [row for row in data if isinstance(row, dict)]

    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
        return rows

    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet files requires pandas") from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def clean_instruction(record: dict[str, Any]) -> str:
    for key in ("problem", "instruction", "question", "position_instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            text = re.sub(r"^Your task instruction:\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*Use 2D points.*$", "", text, flags=re.IGNORECASE)
            return " ".join(text.split())
    return ""


def sample_id(record: dict[str, Any], index: int) -> str:
    for key in ("question_id", "id", "sample_id"):
        if key in record:
            return str(record[key])
    return str(index)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def mentioned_generic_objects(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for obj in sorted(GENERIC_OBJECTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(obj)}s?\b", lowered):
            found.append(obj)
    return found


def has_attribute_or_spatial_cue(text: str) -> bool:
    return bool(set(tokens(text)) & ATTRIBUTE_WORDS)


def repeated_scene_objects(record: dict[str, Any]) -> list[str]:
    names = record.get("selected_obj_names")
    if not isinstance(names, list):
        return []
    normalized = [str(name).strip().lower() for name in names if str(name).strip()]
    counts = Counter(normalized)
    return sorted(name for name, count in counts.items() if count > 1)


def mentioned_repeated_scene_objects(record: dict[str, Any], instruction: str) -> dict[str, int]:
    names = record.get("selected_obj_names")
    if not isinstance(names, list):
        return {}

    normalized = [str(name).strip().lower() for name in names if str(name).strip()]
    counts = Counter(normalized)
    lowered = instruction.lower()
    mentioned = {}
    for name, count in counts.items():
        if count <= 1:
            continue
        if re.search(rf"\b{re.escape(name)}s?\b", lowered):
            mentioned[name] = count
    return mentioned


def infer_risk(record: dict[str, Any], instruction: str) -> tuple[str, list[str]]:
    reasons = []
    generic_mentions = mentioned_generic_objects(instruction)
    mentioned_duplicates = mentioned_repeated_scene_objects(record, instruction)

    if mentioned_duplicates:
        reasons.append("instruction_mentions_duplicate_scene_object")
    if generic_mentions and not has_attribute_or_spatial_cue(instruction):
        reasons.append("generic_object_without_text_cue")
    if instruction.lower().count(" object") >= 1:
        reasons.append("explicit_generic_object_token")
    if set(tokens(instruction)) & RELATION_WORDS and mentioned_duplicates:
        reasons.append("relation_mentions_duplicate_scene_object")

    if mentioned_duplicates:
        return "high", reasons
    if "explicit_generic_object_token" in reasons:
        return "high", reasons
    if reasons:
        return "medium", reasons
    return "low", reasons


def annotation_for(record: dict[str, Any], instruction: str, risk_level: str) -> dict[str, Any]:
    mentioned_duplicates = mentioned_repeated_scene_objects(record, instruction)
    if risk_level == "high" and mentioned_duplicates:
        return {
            "ambiguity_label": "ambiguous",
            "valid_candidate_count": sum(mentioned_duplicates.values()),
            "expected_ids_or_objects": sorted(mentioned_duplicates),
            "expected_ids_behavior": "ask_clarification",
            "notes": "Auto-labeled from original metadata: the instruction mentions an object name that appears multiple times in selected_obj_names.",
        }

    return {
        "ambiguity_label": "unlabeled",
        "valid_candidate_count": None,
        "expected_ids_or_objects": [],
        "expected_ids_behavior": "unlabeled",
        "notes": "",
    }


def build_subset(dataset_paths: list[Path], include_low_risk: bool) -> dict[str, Any]:
    rows = []
    source_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for dataset_path in dataset_paths:
        records = load_records(dataset_path)
        source_name = dataset_path.name
        source_counts[source_name] += len(records)

        for index, record in enumerate(records):
            instruction = clean_instruction(record)
            if not instruction:
                continue

            risk_level, reasons = infer_risk(record, instruction)
            if risk_level == "low" and not include_low_risk:
                continue

            risk_counts[risk_level] += 1
            reason_counts.update(reasons)

            rows.append(
                {
                    "source": source_name,
                    "sample_id": sample_id(record, index),
                    "instruction": instruction,
                    "image": record.get("image") or record.get("rgb_image_path") or record.get("images"),
                    "bbox": record.get("bbox"),
                    "selected_obj_names": record.get("selected_obj_names"),
                    "target_obj_name": record.get("target_obj_name"),
                    "risk_level": risk_level,
                    "risk_reasons": reasons,
                    "annotation": annotation_for(record, instruction, risk_level),
                }
            )

    rows.sort(key=lambda row: ({"high": 0, "medium": 1, "low": 2}[row["risk_level"]], row["source"], row["sample_id"]))
    return {
        "summary": {
            "input_records_by_source": dict(source_counts),
            "selected_samples": len(rows),
            "risk_counts": dict(risk_counts),
            "reason_counts": dict(reason_counts),
        },
        "annotation_schema": {
            "ambiguity_label": "ambiguous | resolved | unknown | invalid",
            "valid_candidate_count": "number of visually valid targets for the instruction",
            "expected_ids_behavior": "ask_clarification | execute_resolved_instruction | reject_unknown",
        },
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eval/roborefit_test.json", "eval/3d_dataset.json"],
        help="JSON, JSONL, or parquet datasets to scan.",
    )
    parser.add_argument("--output", default="output_results/ids_original_ambiguity_subset.json")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--include-low-risk", action="store_true")
    args = parser.parse_args()

    report = build_subset([Path(path) for path in args.datasets], include_low_risk=args.include_low_risk)
    if args.max_samples > 0:
        report["samples"] = report["samples"][: args.max_samples]
        report["summary"]["selected_samples_after_limit"] = len(report["samples"])

    print("Summary")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved subset to {output_path}")


if __name__ == "__main__":
    main()
