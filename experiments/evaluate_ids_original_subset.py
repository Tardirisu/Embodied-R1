from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from disambiguation import InstructionResolver, MCPStateTracker
from experiments.build_ids_original_subset import build_subset


def objects_from_names(names: list[Any] | None) -> list[dict[str, str]]:
    objects = []
    counts: Counter[str] = Counter()
    for raw_name in names or []:
        category = str(raw_name).strip().lower()
        if not category:
            continue
        counts[category] += 1
        objects.append(
            {
                "id": f"{category.replace(' ', '_')}_{counts[category]}",
                "category": category,
                "name": category,
                "location": "",
                "state": {},
            }
        )
    return objects


def candidate_categories(candidates: list[dict[str, str]]) -> Counter[str]:
    categories: Counter[str] = Counter()
    for candidate in candidates:
        name = str(candidate.get("name", "")).strip().lower()
        if name:
            categories[name] += 1
    return categories


def evaluate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    auto_labeled = 0
    ambiguous_correct = 0
    count_correct = 0
    candidate_recall_total = 0.0

    for sample in samples:
        objects = objects_from_names(sample.get("selected_obj_names"))
        tracker = MCPStateTracker(objects)
        resolver = InstructionResolver(state_tracker=tracker)
        result = resolver.resolve(image=None, instruction=sample["instruction"], mode="REG")

        annotation = sample.get("annotation", {})
        expected_label = annotation.get("ambiguity_label")
        expected_objects = [str(item).strip().lower() for item in annotation.get("expected_ids_or_objects", [])]
        expected_count = annotation.get("valid_candidate_count")
        predicted_categories = candidate_categories(result.candidates)

        is_auto_ambiguous = expected_label == "ambiguous"
        if is_auto_ambiguous:
            auto_labeled += 1
            ambiguous_correct += int(result.status == "ambiguous" and result.needs_clarification)
            count_correct += int(expected_count is not None and len(result.candidates) == expected_count)
            if expected_objects:
                recalled = sum(1 for obj in expected_objects if predicted_categories[obj] > 0)
                candidate_recall_total += recalled / len(expected_objects)

        rows.append(
            {
                "source": sample.get("source"),
                "sample_id": sample.get("sample_id"),
                "instruction": sample.get("instruction"),
                "risk_level": sample.get("risk_level"),
                "expected_label": expected_label,
                "expected_objects": expected_objects,
                "expected_candidate_count": expected_count,
                "actual_status": result.status,
                "needs_clarification": result.needs_clarification,
                "actual_candidate_count": len(result.candidates),
                "actual_candidate_categories": dict(predicted_categories),
                "reason": result.reason,
            }
        )

    summary = {
        "num_samples": len(samples),
        "auto_labeled_ambiguous": auto_labeled,
        "ambiguous_detection_accuracy": ambiguous_correct / auto_labeled if auto_labeled else None,
        "candidate_count_accuracy": count_correct / auto_labeled if auto_labeled else None,
        "expected_object_recall": candidate_recall_total / auto_labeled if auto_labeled else None,
        "unsafe_pointing_rate_baseline_on_auto_ambiguous": 1.0 if auto_labeled else None,
        "unsafe_pointing_rate_ids_on_detected_ambiguous": 0.0 if auto_labeled else None,
    }
    return {"summary": summary, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eval/roborefit_test.json", "eval/3d_dataset.json"],
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="output_results/ids_original_subset_eval.json")
    args = parser.parse_args()

    report = build_subset([Path(path) for path in args.datasets], include_low_risk=False)
    samples = report["samples"]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    eval_report = evaluate(samples)
    eval_report["subset_summary"] = report["summary"]

    print("Summary")
    print(json.dumps(eval_report["summary"], ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
