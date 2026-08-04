"""Dependency-light payload structures for model-blind witness diagnostics.

This module records only the exogenous stimulus properties needed by the
model-blind turn-count witness.  It deliberately imports no model runtime,
tokenizer, or tensor library.  The full :mod:`environment` module owns lexical
realisation and model-facing behavior; this module owns the preregistered
conversation-shape contract used by the witness test.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

ALL_ATTACK_CLASSES = ("fragmentation", "semantic", "encoding", "priming")
MULTI_TURN_ATTACK_CLASSES = ("fragmentation", "priming")
SINGLE_TURN_ATTACK_CLASSES = ("semantic", "encoding")

# Conversation lengths are design properties, not model outputs.
_ATTACK_TURN_COUNTS = {
    "fragmentation": 3,
    "semantic": 1,
    "encoding": 1,
    "priming": 3,
}


@dataclass(frozen=True)
class WitnessPayload:
    """Minimal payload view used by dependency-light witness diagnostics."""

    turns: tuple[str, ...]
    label: int
    attack_class: str
    pair_id: int


def _turns(attack_class: str, label: int, pair_id: int) -> tuple[str, ...]:
    n_turns = 1 if label == 0 else _ATTACK_TURN_COUNTS[attack_class]
    role = "legitimate" if label == 0 else attack_class
    return tuple(f"{role} payload {pair_id}, turn {i + 1}" for i in range(n_turns))


def generate_witness_payloads(
    classes: Iterable[str] = ALL_ATTACK_CLASSES,
    *,
    n_pairs_per_class: int = 50,
    seed: int = 0,
) -> list[WitnessPayload]:
    """Generate the matched payload *shape* used by the turn-count witness.

    Each attack-family pair contains one one-turn legitimate payload and one
    adversarial payload with the preregistered family conversation length.
    Text is intentionally inert: the witness reads only ``len(turns)``.
    """
    if n_pairs_per_class <= 0:
        raise ValueError("n_pairs_per_class must be positive")
    selected = tuple(classes)
    unknown = sorted(set(selected) - set(ALL_ATTACK_CLASSES))
    if unknown:
        raise ValueError(f"unknown attack classes: {unknown}")

    pairs: list[tuple[WitnessPayload, WitnessPayload]] = []
    pair_id = 0
    for attack_class in selected:
        for _ in range(n_pairs_per_class):
            pairs.append((
                WitnessPayload(_turns(attack_class, 0, pair_id), 0, "legitimate", pair_id),
                WitnessPayload(_turns(attack_class, 1, pair_id), 1, attack_class, pair_id),
            ))
            pair_id += 1

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return [payload for pair in pairs for payload in pair]
