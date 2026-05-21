from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments.ids_oracle_clarification import build_oracle_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/3d_dataset.json")
    parser.add_argument("--output", default="output_results/ids_oracle_clarification_3d.json")
    parser.add_argument("--ambiguous-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for index, record in enumerate(records):
        row = build_oracle_row(record, index)
        if args.ambiguous_only and row["ids_status"] != "ambiguous":
            continue
        rows.append(row)
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    status_counts = Counter(row["ids_status"] for row in rows)
    quality_counts = Counter(row["oracle_quality"] for row in rows)
    summary = {
        "dataset": args.dataset,
        "num_input_records": len(records),
        "num_output_records": len(rows),
        "ids_status_counts": dict(status_counts),
        "oracle_available": sum(1 for row in rows if row["oracle_available"]),
        "oracle_quality_counts": dict(quality_counts),
        "reference_duplicate_oracles": sum(
            1 for row in rows if row["oracle_available"] and row["reference_duplicate"]
        ),
        "target_duplicate_oracles": sum(
            1 for row in rows if row["oracle_available"] and row["target_duplicate"]
        ),
    }

    report = {"summary": summary, "samples": rows}
    print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved oracle clarification dataset to {output_path}")


if __name__ == "__main__":
    main()
