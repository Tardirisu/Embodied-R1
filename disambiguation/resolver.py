from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .object_vocabulary import ATTRIBUTE_WORDS as _ATTRIBUTE_WORDS
from .object_vocabulary import GENERIC_OBJECTS as _GENERIC_OBJECTS


_SPATIAL_WORDS = {
    "left",
    "right",
    "front",
    "back",
    "behind",
    "top",
    "bottom",
    "upper",
    "lower",
    "center",
    "middle",
    "near",
    "nearest",
    "far",
    "farthest",
}


@dataclass
class DisambiguationResult:
    original_instruction: str
    resolved_instruction: str
    status: str
    needs_clarification: bool = False
    clarification_question: str | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    raw_candidate_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstructionResolver:
    """Instruction disambiguation layer.

    The resolver keeps final ambiguity decisions in code. If a candidate
    extractor is provided, the model is used only to list visible candidates.
    """

    def __init__(self, candidate_extractor: Any | None = None, state_tracker: Any | None = None):
        self.candidate_extractor = candidate_extractor
        self.state_tracker = state_tracker

    def resolve(self, image: Any, instruction: str, mode: str | None = None) -> DisambiguationResult:
        normalized = " ".join(instruction.strip().split())
        if not normalized:
            return DisambiguationResult(
                original_instruction=instruction,
                resolved_instruction=normalized,
                status="invalid",
                needs_clarification=True,
                clarification_question="What object or action should the robot perform?",
                reason="empty_instruction",
            )

        state_candidates = self._retrieve_state_candidates(normalized, mode)
        if state_candidates:
            return self._result_from_candidates(
                instruction=instruction,
                normalized=normalized,
                candidates=state_candidates,
                reason_prefix="mcp_state",
            )

        if self._state_has_single_mentioned_category(normalized):
            return self._result_from_candidates(
                instruction=instruction,
                normalized=normalized,
                candidates=[],
                reason_prefix="mcp_state",
            )

        generic_object = self._find_generic_object(normalized)

        if not self._should_check_candidates(normalized, mode):
            return DisambiguationResult(
                original_instruction=instruction,
                resolved_instruction=normalized,
                status="resolved",
                needs_clarification=False,
                reason="instruction_preserved",
            )

        if self.candidate_extractor is None:
            return DisambiguationResult(
                original_instruction=instruction,
                resolved_instruction=normalized,
                status="ambiguous",
                needs_clarification=True,
                clarification_question=self._generic_question(normalized, generic_object),
                candidates=[],
                reason="candidate_extractor_unavailable",
            )

        extraction = self.candidate_extractor.extract_candidates(image=image, instruction=normalized, mode=mode)
        return self._result_from_candidates(
            instruction=instruction,
            normalized=normalized,
            candidates=extraction.candidates,
            reason_prefix="r1",
            raw_candidate_output=extraction.raw_output,
        )

    def _result_from_candidates(
        self,
        instruction: str,
        normalized: str,
        candidates: list[dict[str, str]],
        reason_prefix: str,
        raw_candidate_output: str = "",
    ) -> DisambiguationResult:
        if len(candidates) == 1:
            return DisambiguationResult(
                original_instruction=instruction,
                resolved_instruction=self._resolved_instruction(normalized, candidates[0]),
                status="resolved",
                needs_clarification=False,
                candidates=candidates,
                reason=f"{reason_prefix}_single_candidate",
                raw_candidate_output=raw_candidate_output,
            )

        if len(candidates) > 1:
            return DisambiguationResult(
                original_instruction=instruction,
                resolved_instruction=normalized,
                status="ambiguous",
                needs_clarification=True,
                clarification_question=self._candidate_question(normalized, candidates),
                candidates=candidates,
                reason=f"{reason_prefix}_multiple_candidates",
                raw_candidate_output=raw_candidate_output,
            )

        return DisambiguationResult(
            original_instruction=instruction,
            resolved_instruction=normalized,
            status="unknown",
            needs_clarification=True,
            clarification_question="I cannot identify the target. Could you describe it more specifically?",
            candidates=[],
            reason=f"{reason_prefix}_no_candidates",
            raw_candidate_output=raw_candidate_output,
        )

    def _retrieve_state_candidates(self, instruction: str, mode: str | None) -> list[dict[str, str]]:
        if self.state_tracker is None:
            return []
        if not hasattr(self.state_tracker, "retrieve_candidates"):
            return []
        candidates = self.state_tracker.retrieve_candidates(instruction=instruction, mode=mode)
        return candidates if isinstance(candidates, list) else []

    def _state_has_mentioned_category(self, instruction: str) -> bool:
        if self.state_tracker is None:
            return False
        if not hasattr(self.state_tracker, "has_mentioned_category"):
            return False
        return bool(self.state_tracker.has_mentioned_category(instruction))

    def _state_has_single_mentioned_category(self, instruction: str) -> bool:
        if self.state_tracker is None:
            return False
        if not hasattr(self.state_tracker, "has_single_mentioned_category"):
            return False
        return bool(self.state_tracker.has_single_mentioned_category(instruction))

    def _should_check_candidates(self, instruction: str, mode: str | None) -> bool:
        generic_object = self._find_generic_object(instruction)
        if generic_object is None:
            return False

        tokens = self._tokens(instruction)
        generic_count = sum(1 for token in tokens if token == generic_object)
        has_disambiguating_cue = self._has_disambiguating_cue(instruction)

        if generic_count >= 2 and not self._has_attribute_cue(instruction):
            return True

        if mode in {"REG", "OFG"} and not has_disambiguating_cue:
            return True

        return False

    def _find_generic_object(self, instruction: str) -> str | None:
        lowered = instruction.lower()
        for obj in sorted(_GENERIC_OBJECTS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(obj)}s?\b", lowered):
                return obj
        return None

    def _has_disambiguating_cue(self, instruction: str) -> bool:
        tokens = set(self._tokens(instruction))
        if self._has_attribute_cue(instruction):
            return True
        if tokens & _SPATIAL_WORDS:
            return True
        return bool(re.search(r"\b(on|in|inside|under|above|below|beside|next to)\b", instruction.lower()))

    def _has_attribute_cue(self, instruction: str) -> bool:
        tokens = set(self._tokens(instruction))
        return bool(tokens & _ATTRIBUTE_WORDS)

    def _tokens(self, instruction: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+", instruction.lower())

    def _generic_question(self, instruction: str, generic_object: str | None) -> str:
        if self._looks_like_relation_task(instruction):
            return "Which object should be moved, and which object should it be placed relative to?"
        if generic_object:
            return f"Which {generic_object} do you mean?"
        return "Which target do you mean?"

    def _candidate_question(self, instruction: str, candidates: list[dict[str, str]]) -> str:
        if self._looks_like_relation_task(instruction):
            return "Which object should be moved, and which object should it be placed relative to?"

        options = [self._candidate_phrase(candidate) for candidate in candidates]
        options = [option for option in options if option]
        if not options:
            return "Which target do you mean?"
        if len(options) == 1:
            return f"Do you mean {options[0]}?"
        if len(options) == 2:
            joined = " or ".join(options)
        else:
            joined = ", ".join(options[:-1]) + f", or {options[-1]}"
        return f"Which target do you mean: {joined}?"

    def _resolved_instruction(self, instruction: str, candidate: dict[str, str]) -> str:
        phrase = self._candidate_phrase(candidate)
        if not phrase:
            return instruction
        return f"{instruction} referring to {phrase}"

    def _candidate_phrase(self, candidate: dict[str, str]) -> str:
        name = candidate.get("name", "").strip()
        attributes = candidate.get("visual_attributes", "").strip()
        location = candidate.get("location", "").strip()

        parts = []
        if attributes:
            name_tokens = set(self._tokens(name))
            attribute_parts = [
                part.strip()
                for part in attributes.split(",")
                if part.strip() and not (set(self._tokens(part.strip())) <= name_tokens)
            ]
            if attribute_parts:
                parts.append(", ".join(attribute_parts))
        if name:
            parts.append(name)

        phrase = " ".join(parts).strip()
        if location:
            phrase = f"{phrase} at {location}" if phrase else f"the target at {location}"
        return phrase

    def _looks_like_relation_task(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return bool(re.search(r"\b(on top of|onto|into|in|on|under|above|below|beside|next to)\b", lowered))
