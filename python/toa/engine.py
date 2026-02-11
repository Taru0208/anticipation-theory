"""Core analysis engine — Python port of game.ixx."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable
from collections import defaultdict


MAX_ANTICIPATION_NEST_LEVEL = 20


@dataclass
class StateNode:
    """Result node for a single state after analysis."""
    d_global: float = 0.0
    a: list[float] = field(default_factory=lambda: [0.0] * MAX_ANTICIPATION_NEST_LEVEL)

    def sum_a(self) -> float:
        return sum(self.a)


@dataclass
class GameAnalysis:
    """Complete analysis result for a game."""
    states: list  # forward topological order
    states_r: list  # reverse topological order
    state_nodes: dict  # state -> StateNode
    gds_components: list[float] = field(default_factory=lambda: [0.0] * MAX_ANTICIPATION_NEST_LEVEL)
    game_design_score: float = 0.0


def _serialize_r(
    start_state: Hashable,
    get_transitions: Callable,
    config: Any,
) -> list:
    """Post-order DFS traversal — returns states in reverse topological order.

    Mirrors serializeR<game_t> in game.ixx.
    """
    visited = set()
    result = []

    def recurse(s):
        if s in visited:
            return
        visited.add(s)
        for prob, next_s in get_transitions(s, config):
            recurse(next_s)
        result.append(s)

    recurse(start_state)
    return result


def analyze(
    *,
    initial_state: Hashable,
    is_terminal: Callable[[Hashable], bool],
    get_transitions: Callable[[Hashable, Any], list[tuple[float, Hashable]]],
    compute_intrinsic_desire: Callable[[Hashable], float],
    config: Any = None,
    nest_level: int = MAX_ANTICIPATION_NEST_LEVEL,
) -> GameAnalysis:
    """Analyze a game and compute anticipation components + GDS.

    This is a faithful port of game::analyze<game_t> from game.ixx.

    Args:
        initial_state: The starting state of the game.
        is_terminal: Predicate for terminal states.
        get_transitions: Returns [(probability, next_state), ...] for a state.
        compute_intrinsic_desire: Returns intrinsic desire (0 or 1) for terminal states.
        config: Optional game configuration passed to get_transitions.
        nest_level: How many anticipation components to compute (default: 20).

    Returns:
        GameAnalysis with all states, state nodes, GDS components, and total GDS.
    """
    if nest_level > MAX_ANTICIPATION_NEST_LEVEL:
        raise ValueError(f"nest_level {nest_level} exceeds MAX_ANTICIPATION_NEST_LEVEL {MAX_ANTICIPATION_NEST_LEVEL}")

    # Topological ordering
    states_r = _serialize_r(initial_state, get_transitions, config)
    states = list(reversed(states_r))

    # Result storage
    result_nodes: dict[Hashable, StateNode] = {s: StateNode() for s in states}
    gds_components = [0.0] * MAX_ANTICIPATION_NEST_LEVEL

    for component_idx in range(nest_level):
        # --- buildNodes ---
        d_local: dict[Hashable, float] = {}
        d_global: dict[Hashable, float] = {}
        a_values: dict[Hashable, float] = {}

        # --- seedD ---
        for s in states:
            if component_idx == 0:
                d_local[s] = compute_intrinsic_desire(s)
            else:
                d_local[s] = result_nodes[s].a[component_idx - 1]

        # --- propagateD ---
        # Process in reverse topological order (leaves first)
        for s in states_r:
            d_global[s] = d_local[s]
            for prob, next_s in get_transitions(s, config):
                d_global[s] += d_global[next_s] * prob

        # --- compute_A ---
        for s in states_r:
            transitions = get_transitions(s, config)
            if not transitions:
                a_values[s] = 0.0
                continue

            # Perspective desire: D(s') - D(s) for each transition
            # Weighted mean of perspective desires
            sum_pd = 0.0
            for prob, next_s in transitions:
                perspective_d = d_global[next_s] - d_global[s]
                sum_pd += prob * perspective_d

            avg_pd = sum_pd  # Already probability-weighted sum = weighted mean

            # Weighted variance of perspective desires
            weighted_variance = 0.0
            for prob, next_s in transitions:
                perspective_d = d_global[next_s] - d_global[s]
                diff = perspective_d - avg_pd
                weighted_variance += prob * diff * diff

            a_values[s] = math.sqrt(weighted_variance)

        # --- compute_gamedesign_score ---
        # Propagate reach probabilities and accumulated anticipation
        # steps[s] = dict of {step_count: (reach_probability, accumulated_a)}
        steps: dict[Hashable, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        # [0] = reach_probability, [1] = accumulated anticipation

        steps[initial_state][0][0] = 1.0  # reach_probability
        steps[initial_state][0][1] = a_values[initial_state]  # initial A

        # Forward propagation (topological order)
        for s in states:
            for prob, next_s in get_transitions(s, config):
                for step_i, (reach_p, acc_a) in list(steps[s].items()):
                    entry = steps[next_s][step_i + 1]
                    # Propagate reach probability
                    entry[0] += reach_p * prob
                    # Propagate accumulated A: current accumulated * P + target's own A * reach * P
                    entry[1] += acc_a * prob + a_values[next_s] * reach_p * prob

        # Sum at terminal states
        total_end_probability = 0.0
        for s in states:
            if is_terminal(s):
                for step_i, (reach_p, acc_a) in steps[s].items():
                    total_end_probability += reach_p
                    if step_i > 0:
                        gds_components[component_idx] += acc_a / step_i

        # Sanity check
        if abs(total_end_probability - 1.0) > 0.001:
            raise RuntimeError(
                f"Bug in probability propagation: total_end_probability={total_end_probability:.6f}"
            )

        # --- reflect_to_result ---
        for s in states:
            result_nodes[s].a[component_idx] = a_values[s]
            if component_idx == 0:
                result_nodes[s].d_global = d_global[s]

    # Build final result
    result = GameAnalysis(
        states=states,
        states_r=states_r,
        state_nodes=result_nodes,
        gds_components=gds_components,
    )
    result.game_design_score = sum(gds_components[:nest_level])

    return result
