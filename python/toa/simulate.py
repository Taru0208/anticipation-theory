"""Monte Carlo simulation-based GDS computation.

An alternative to the exact dynamic programming approach. Uses random
sampling of game trajectories to estimate GDS.

Useful for:
1. Validating exact computation
2. Analyzing games too large for exact computation
3. Understanding trajectory-level behavior
"""

import math
import random
from typing import Any, Callable, Hashable

from toa.engine import analyze, GameAnalysis


def simulate_gds(
    *,
    initial_state: Hashable,
    is_terminal: Callable[[Hashable], bool],
    get_transitions: Callable[[Hashable, Any], list[tuple[float, Hashable]]],
    compute_intrinsic_desire: Callable[[Hashable], float],
    config: Any = None,
    nest_level: int = 5,
    num_simulations: int = 10000,
    seed: int | None = None,
) -> dict:
    """Estimate GDS via Monte Carlo simulation.

    Plays random games and collects anticipation values along trajectories,
    then averages them the same way the exact method does.

    This requires an exact analysis first (for A values at each state),
    then simulates trajectories to estimate the GDS weighting.

    Returns dict with:
        - gds_sim: simulated GDS
        - gds_components_sim: per-component GDS estimates
        - gds_exact: exact GDS for comparison
        - num_simulations: actual simulation count
        - mean_game_length: average trajectory length
    """
    rng = random.Random(seed)

    # First, compute exact analysis to get A values at each state
    exact = analyze(
        initial_state=initial_state,
        is_terminal=is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=compute_intrinsic_desire,
        config=config,
        nest_level=nest_level,
    )

    # Simulate trajectories
    component_sums = [0.0] * nest_level
    total_length = 0
    completed = 0

    for _ in range(num_simulations):
        state = initial_state
        trajectory_a = [[] for _ in range(nest_level)]  # A values per component

        steps = 0
        max_steps = 1000  # Safety limit

        while not is_terminal(state) and steps < max_steps:
            # Record A values at this state
            node = exact.state_nodes.get(state)
            if node:
                for c in range(nest_level):
                    trajectory_a[c].append(node.a[c])

            # Sample next state
            transitions = get_transitions(state, config)
            if not transitions:
                break

            r = rng.random()
            cumulative = 0.0
            for prob, next_state in transitions:
                cumulative += prob
                if r <= cumulative:
                    state = next_state
                    break
            else:
                state = transitions[-1][1]

            steps += 1

        if steps > 0:
            # GDS for this trajectory: average A over non-terminal states
            for c in range(nest_level):
                if trajectory_a[c]:
                    component_sums[c] += sum(trajectory_a[c]) / len(trajectory_a[c])

            total_length += steps
            completed += 1

    if completed == 0:
        return {
            "gds_sim": 0.0,
            "gds_components_sim": [0.0] * nest_level,
            "gds_exact": exact.game_design_score,
            "num_simulations": 0,
            "mean_game_length": 0,
        }

    gds_components_sim = [s / completed for s in component_sums]
    gds_sim = sum(gds_components_sim)

    return {
        "gds_sim": gds_sim,
        "gds_components_sim": gds_components_sim,
        "gds_exact": exact.game_design_score,
        "gds_exact_components": exact.gds_components[:nest_level],
        "num_simulations": completed,
        "mean_game_length": total_length / completed,
        "error": abs(gds_sim - exact.game_design_score),
        "relative_error": abs(gds_sim - exact.game_design_score) / max(0.001, exact.game_design_score),
    }
