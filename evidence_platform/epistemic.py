"""Machine-readable epistemic contracts for probe interpretation.

The central distinction is deliberately executable: predictive success may
support probe validity while experimental non-identifiability blocks semantic
interpretation.  These objects are small, dependency-free, and suitable for
JSON publication payloads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class EpistemicStatus(str, Enum):
    SUPPORTED = "supported"
    BLOCKED = "blocked"
    UNIDENTIFIED = "unidentified"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class EvidenceClaim:
    evidence: str
    interpretation: str
    cannot_conclude: str
    status: EpistemicStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class BC1Report:
    """Canonical report for the BC1 construct-identifiability gate."""

    observed_support: str
    construct_identifiable: bool
    model_blind_match: bool
    semantic_interpretation_supported: bool
    blocking_gate: str = "BC1"
    experimental_status: EpistemicStatus = EpistemicStatus.EXPLORATORY
    rationale: str = ""

    @classmethod
    def from_design(
        cls,
        *,
        observed_cells: Iterable[tuple[int, int]],
        model_blind_match: bool,
        rationale: str = "",
    ) -> "BC1Report":
        cells = {(int(i), int(l)) for i, l in observed_cells}
        required = {(0, 0), (0, 1), (1, 0), (1, 1)}
        identifiable = required.issubset(cells)
        supported = identifiable and not model_blind_match
        status = EpistemicStatus.SUPPORTED if supported else EpistemicStatus.UNIDENTIFIED
        support = "fully_crossed" if identifiable else "diagonal_or_incomplete"
        return cls(
            observed_support=support,
            construct_identifiable=identifiable,
            model_blind_match=bool(model_blind_match),
            semantic_interpretation_supported=supported,
            experimental_status=status,
            rationale=rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experimental_status"] = self.experimental_status.value
        payload["probe_validity_may_still_hold"] = True
        return payload

    def strict_interpret(self) -> dict[str, Any]:
        """Return a publication payload only when semantic interpretation passes."""
        if not self.semantic_interpretation_supported:
            raise ValueError(
                "Semantic interpretation is blocked by BC1: the intended construct "
                "is not experimentally identifiable or a model-blind match remains."
            )
        return self.to_dict()


def maritime_counterexample_report() -> BC1Report:
    """Canonical machine-readable summary of the motivating experiment."""
    return BC1Report.from_design(
        observed_cells={(0, 0), (1, 1)},
        model_blind_match=True,
        rationale=(
            "The stimulus design observes intent and lexical form only together; "
            "the repository-computed turn-count witness matches the held-out probe AUC on the preregistered multi-turn attack-family subset."
        ),
    )


def scientific_contract() -> Mapping[str, str]:
    """Repository-wide publication contract shared by result producers."""
    return {
        "estimand": "Declare what each quantity estimates.",
        "assumptions": "Declare the design assumptions required for interpretation.",
        "uncertainty": "Report sampling or resampling uncertainty where estimable.",
        "limits": "State what the evidence does not support.",
        "status": "Expose the final epistemic status in machine-readable form.",
    }
