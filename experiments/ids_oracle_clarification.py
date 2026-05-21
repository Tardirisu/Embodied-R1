from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from disambiguation import InstructionResolver, MCPStateTracker


@dataclass(frozen=True)
class OracleObject:
    index: int
    category: str
    position: list[float]
    label: str


def clean_instruction(record: dict[str, Any]) -> str:
    for key in ("position_instruction", "instruction", "problem", "question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            text = re.sub(r"^Your task instruction:\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*Use 2D points.*$", "", text, flags=re.IGNORECASE)
            return " ".join(text.split())
    return ""


def object_states_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    states = []
    counts: Counter[str] = Counter()
    names = record.get("selected_obj_names") or []
    positions = record.get("init_obj_pos") or []
    labels = instance_labels(names, positions)

    for index, raw_name in enumerate(names):
        category = str(raw_name).strip().lower()
        if not category:
            continue
        counts[category] += 1
        states.append(
            {
                "id": f"{category.replace(' ', '_')}_{counts[category]}",
                "category": category,
                "name": category,
                "location": labels.get(index, ""),
                "state": {},
            }
        )
    return states


def resolve_record(record: dict[str, Any]) -> dict[str, Any]:
    instruction = clean_instruction(record)
    tracker = MCPStateTracker(object_states_from_record(record))
    resolver = InstructionResolver(state_tracker=tracker)
    return resolver.resolve(image=None, instruction=instruction, mode="REG").to_dict()


def instance_labels(names: list[Any], positions: list[Any]) -> dict[int, str]:
    xyz = {
        index: _xyz(position)
        for index, position in enumerate(positions)
        if index < len(names) and _xyz(position) is not None
    }
    if not xyz:
        return {}

    xs = [position[0] for position in xyz.values()]
    ys = [position[1] for position in xyz.values()]
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)

    labels = {}
    for index, position in xyz.items():
        parts = []
        if x_span > 0.03:
            parts.append("back" if position[0] > x_mid else "front")
        if y_span > 0.03:
            parts.append("left" if position[1] > y_mid else "right")
        labels[index] = "-".join(parts) if parts else f"object {index + 1}"
    return labels


def reference_objects(record: dict[str, Any]) -> list[OracleObject]:
    names = [str(name).strip().lower() for name in record.get("selected_obj_names", [])]
    positions = record.get("init_obj_pos") or []
    answer = record.get("answer") or {}
    answer_positions = answer.get("object") if isinstance(answer, dict) else None
    labels = instance_labels(names, positions)
    refs = []

    if not isinstance(answer_positions, list):
        return refs

    used_indices: set[int] = set()
    for answer_position in answer_positions:
        answer_xyz = _xyz(answer_position)
        if answer_xyz is None:
            continue
        best_index = None
        best_distance = float("inf")
        for index, position in enumerate(positions):
            if index in used_indices:
                continue
            xyz = _xyz(position)
            if xyz is None:
                continue
            distance = math.dist(xyz, answer_xyz)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None and best_index < len(names):
            used_indices.add(best_index)
            refs.append(
                OracleObject(
                    index=best_index,
                    category=names[best_index],
                    position=list(_xyz(positions[best_index]) or []),
                    label=labels.get(best_index, f"object {best_index + 1}"),
                )
            )
    return refs


def duplicate_categories(record: dict[str, Any]) -> set[str]:
    names = [str(name).strip().lower() for name in record.get("selected_obj_names", []) if str(name).strip()]
    counts = Counter(names)
    return {name for name, count in counts.items() if count > 1}


def oracle_clarification(record: dict[str, Any], ids_result: dict[str, Any]) -> dict[str, Any]:
    instruction = clean_instruction(record)
    refs = reference_objects(record)
    duplicate_names = duplicate_categories(record)
    target = str(record.get("target_obj_name", "")).strip().lower()
    target_duplicate = bool(target and target in duplicate_names)
    reference_duplicate = any(ref.category in duplicate_names for ref in refs)

    if ids_result.get("status") != "ambiguous":
        return {
            "oracle_available": False,
            "oracle_instruction": instruction,
            "oracle_clarification": "",
            "oracle_quality": "not_invoked_resolved",
            "reference_objects": [asdict(ref) for ref in refs],
            "target_duplicate": target_duplicate,
            "reference_duplicate": reference_duplicate,
        }

    clauses = []
    if refs:
        if len(refs) == 1:
            ref = refs[0]
            clauses.append(f"use the {ref.label} {ref.category} as the reference object")
        else:
            joined_refs = " and ".join(f"the {ref.label} {ref.category}" for ref in refs)
            clauses.append(f"use {joined_refs} as the reference objects")

    if target_duplicate:
        target_clause = _target_clause(record, refs)
        if target_clause:
            clauses.append(target_clause)
        else:
            clauses.append(f"the moved target is the oracle-selected {target}")

    if not clauses:
        return {
            "oracle_available": False,
            "oracle_instruction": instruction,
            "oracle_clarification": "",
            "oracle_quality": "not_needed",
            "reference_objects": [asdict(ref) for ref in refs],
            "target_duplicate": target_duplicate,
            "reference_duplicate": reference_duplicate,
        }

    clarification = "Clarification: " + "; ".join(clauses) + "."
    quality = "full_reference_oracle" if reference_duplicate else "target_oracle_text_only"
    if target_duplicate and not reference_duplicate:
        quality = "target_oracle_text_only"

    return {
        "oracle_available": True,
        "oracle_instruction": f"{instruction} {clarification}",
        "oracle_clarification": clarification,
        "oracle_quality": quality,
        "reference_objects": [asdict(ref) for ref in refs],
        "target_duplicate": target_duplicate,
        "reference_duplicate": reference_duplicate,
    }


def build_oracle_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    instruction = clean_instruction(record)
    ids_result = resolve_record(record)
    oracle = oracle_clarification(record, ids_result)
    answer = record.get("answer")
    return {
        "sample_index": index,
        "sample_id": record.get("id", index),
        "baseline_instruction": instruction,
        "ids_oracle_instruction": oracle["oracle_instruction"] if oracle["oracle_available"] else instruction,
        "ids_status": ids_result.get("status"),
        "ids_needs_clarification": ids_result.get("needs_clarification"),
        "ids_candidates": ids_result.get("candidates", []),
        "oracle_available": oracle["oracle_available"],
        "oracle_clarification": oracle["oracle_clarification"],
        "oracle_quality": oracle["oracle_quality"],
        "reference_objects": oracle["reference_objects"],
        "target_duplicate": oracle["target_duplicate"],
        "reference_duplicate": oracle["reference_duplicate"],
        "selected_obj_names": record.get("selected_obj_names"),
        "target_obj_name": record.get("target_obj_name"),
        "position_tag": record.get("position_tag"),
        "answer": answer,
        "rgb_image_path": record.get("rgb_image_path"),
        "depth_image_path": record.get("depth_image_path"),
        "images": record.get("images"),
        "source_record": record,
    }


def _target_clause(record: dict[str, Any], refs: list[OracleObject]) -> str:
    target = str(record.get("target_obj_name", "")).strip().lower()
    if not target:
        return ""
    names = [str(name).strip().lower() for name in record.get("selected_obj_names", [])]
    positions = record.get("init_obj_pos") or []
    labels = instance_labels(names, positions)
    ref_indices = {ref.index for ref in refs}
    target_indices = [index for index, name in enumerate(names) if name == target]
    non_reference_targets = [index for index in target_indices if index not in ref_indices]
    if len(non_reference_targets) == 1:
        index = non_reference_targets[0]
        return f"move the {labels.get(index, f'object {index + 1}')} {target}"
    return ""


def _xyz(position: Any) -> tuple[float, float, float] | None:
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError):
        return None
