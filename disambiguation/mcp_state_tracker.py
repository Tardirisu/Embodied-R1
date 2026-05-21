from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any


_COLOR_WORDS = {
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
}

_SIZE_WORDS = {"small", "large", "big", "tiny", "wide", "thin"}


@dataclass
class MCPObjectState:
    object_id: str
    category: str
    name: str = ""
    color: str = ""
    size: str = ""
    location: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def to_candidate(self) -> dict[str, str]:
        attributes = ", ".join(value for value in [self.color, self.size] if value)
        state_text = ", ".join(
            self._state_phrase(key, value)
            for key, value in sorted(self.state.items())
            if self._state_phrase(key, value)
        )
        if state_text:
            attributes = f"{attributes}, {state_text}" if attributes else state_text
        return {
            "name": self.name or self.category,
            "visual_attributes": attributes,
            "location": self.location,
            "object_id": self.object_id,
        }

    def _state_phrase(self, key: str, value: Any) -> str:
        if value in (None, ""):
            return ""
        if key == "on" and isinstance(value, bool):
            return "on" if value else "off"
        if key == "open" and isinstance(value, bool):
            return "open" if value else "closed"
        return f"{key}={value}"


class MCPStateTracker:
    """Lightweight structured state tracker for instruction disambiguation."""

    def __init__(self, objects: list[MCPObjectState | dict[str, Any]] | None = None):
        self.objects: list[MCPObjectState] = []
        for obj in objects or []:
            self.add_object(obj)

    def add_object(self, obj: MCPObjectState | dict[str, Any]) -> None:
        if isinstance(obj, MCPObjectState):
            self.objects.append(obj)
            return
        self.objects.append(
            MCPObjectState(
                object_id=str(obj.get("object_id", obj.get("id", ""))).strip(),
                category=str(obj.get("category", "")).strip().lower(),
                name=str(obj.get("name", "")).strip(),
                color=str(obj.get("color", "")).strip().lower(),
                size=str(obj.get("size", "")).strip().lower(),
                location=str(obj.get("location", "")).strip(),
                state=dict(obj.get("state", {})),
            )
        )

    def retrieve_candidates(self, instruction: str, mode: str | None = None) -> list[dict[str, str]]:
        tokens = set(self._tokens(instruction))
        categories = self._mentioned_categories(instruction)
        if not categories:
            return []

        if "object" in tokens or "thing" in tokens:
            candidate_categories = categories
        elif len(categories) > 1:
            counts = Counter(obj.category for obj in self.objects)
            candidate_categories = {category for category in categories if counts[category] > 1}
            if not candidate_categories:
                return []
        else:
            candidate_categories = categories

        candidates = [obj for obj in self.objects if obj.category in candidate_categories]
        candidates = self._filter_by_text_constraints(candidates, tokens, instruction)
        return [obj.to_candidate() for obj in candidates]

    def has_mentioned_category(self, instruction: str) -> bool:
        return bool(self._mentioned_categories(instruction))

    def has_single_mentioned_category(self, instruction: str) -> bool:
        return len(self._mentioned_categories(instruction)) == 1

    def _mentioned_categories(self, instruction: str) -> set[str]:
        tokens = set(self._tokens(instruction))
        lowered = instruction.lower()
        categories = {obj.category for obj in self.objects if obj.category}
        mentioned = set()
        if "object" in tokens or "thing" in tokens:
            mentioned.update(categories)
            return mentioned

        for category in categories:
            aliases = {category, category.rstrip("s"), f"{category}s"}
            for alias in aliases:
                if alias and re.search(rf"\b{re.escape(alias)}\b", lowered):
                    mentioned.add(category)
                    break
        return mentioned

    def _filter_by_text_constraints(
        self, objects: list[MCPObjectState], tokens: set[str], instruction: str
    ) -> list[MCPObjectState]:
        category_tokens = {
            token
            for obj in self.objects
            for token in self._tokens(obj.category)
            if token
        }
        locations = {
            word
            for obj in self.objects
            for word in self._tokens(obj.location)
            if word in {"left", "right", "center", "middle", "top", "bottom", "front", "back"}
        }

        color_constraints = (tokens & _COLOR_WORDS) - category_tokens
        size_constraints = (tokens & _SIZE_WORDS) - category_tokens
        location_constraints = self._location_constraints(tokens, instruction, locations)

        filtered = objects
        if color_constraints:
            filtered = [obj for obj in filtered if obj.color in color_constraints]
        if size_constraints:
            filtered = [obj for obj in filtered if obj.size in size_constraints]
        if location_constraints:
            filtered = [
                obj
                for obj in filtered
                if location_constraints & set(self._tokens(obj.location))
            ]

        state_constraints = self._state_constraints(instruction, tokens)
        for key, expected in state_constraints.items():
            filtered = [obj for obj in filtered if obj.state.get(key) == expected]

        return filtered

    def _location_constraints(self, tokens: set[str], instruction: str, locations: set[str]) -> set[str]:
        lowered = instruction.lower()
        if re.search(r"\b(left|right|front|back|top|bottom)\s+of\b", lowered):
            return set()
        if re.search(r"\bon\s+top\s+of\b", lowered):
            return set()
        return tokens & locations

    def _state_constraints(self, instruction: str, tokens: set[str]) -> dict[str, Any]:
        lowered = instruction.lower()
        constraints: dict[str, Any] = {}
        if re.search(r"\b(that is|currently|already)\s+open\b", lowered):
            constraints["open"] = True
        elif re.search(r"\b(that is|currently|already)\s+(closed|close)\b", lowered):
            constraints["open"] = False
        elif re.search(r"^\s*open\b", lowered):
            constraints["open"] = False
        elif re.search(r"^\s*close\b", lowered):
            constraints["open"] = True

        if re.search(r"\b(that is|currently|already)\s+(on|enabled)\b", lowered):
            constraints["on"] = True
        elif re.search(r"\b(that is|currently|already)\s+(off|disabled)\b", lowered):
            constraints["on"] = False
        elif re.search(r"^\s*turn\s+on\b", lowered):
            constraints["on"] = False
        elif re.search(r"^\s*turn\s+off\b", lowered):
            constraints["on"] = True

        if "open" in tokens and "open" not in constraints:
            constraints["open"] = True
        if ("closed" in tokens or "close" in tokens) and "open" not in constraints:
            constraints["open"] = False
        return constraints

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())
