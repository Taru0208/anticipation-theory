"""Base game protocol and helpers."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, Sequence, Hashable, Any, runtime_checkable


@runtime_checkable
class Game(Protocol):
    """Protocol that game definitions must implement.

    A game is defined by:
    - A hashable State type
    - An initial_state() class method
    - An is_terminal(state) class method
    - A get_transitions(state, config) class method returning [(probability, next_state), ...]
    - A compute_intrinsic_desire(state) class method
    - An optional tostr(state) class method for display
    """

    @staticmethod
    def initial_state() -> Hashable: ...

    @staticmethod
    def is_terminal(state: Hashable) -> bool: ...

    @staticmethod
    def get_transitions(state: Hashable, config: Any = None) -> list[tuple[float, Hashable]]: ...

    @staticmethod
    def compute_intrinsic_desire(state: Hashable) -> float: ...

    @staticmethod
    def tostr(state: Hashable) -> str: ...


def sanitize_transitions(transitions: list[tuple[float, Any]]) -> list[tuple[float, Any]]:
    """Normalize transition probabilities to sum to 1.0."""
    total = sum(p for p, _ in transitions)
    if total == 0:
        return transitions
    return [(p / total, s) for p, s in transitions]
