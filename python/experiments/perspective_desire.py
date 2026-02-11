"""Perspective Desire Comparison Experiment

The paper defines anticipation using "local" desire values:
  A(s) = sqrt(Σ P(s→s') · (D(s') - μ)²)
  where D(s') = intrinsic desire (0 or 1 for terminal, 0 for non-terminal)
  and μ = Σ P(s→s') · D(s')

But the actual C++ code uses "perspective desire":
  D_perspective(s→s') = D_global(s') - D_global(s)
  A(s) = sqrt(Σ P(s→s') · (D_p(s→s') - μ_p)²)
  where μ_p = Σ P(s→s') · D_perspective(s→s')

This experiment:
1. Implements the naive (paper) version of anticipation
2. Compares with the perspective (code) version
3. Identifies which games differ most and why
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze, _serialize_r, StateNode, GameAnalysis, MAX_ANTICIPATION_NEST_LEVEL
from toa.games.hpgame import HpGame
from toa.games.hpgame_rage import HpGameRage
from toa.games.lanegame import LaneGame


def analyze_naive(
    *,
    initial_state,
    is_terminal,
    get_transitions,
    compute_intrinsic_desire,
    config=None,
    nest_level=5,
):
    """Analyze using the 'naive' (paper) formulation.

    Differs from the code implementation:
    - Uses D_global(s') directly instead of D_global(s') - D_global(s)
    - This is closer to the paper's original formulation
    """
    states_r = _serialize_r(initial_state, get_transitions, config)
    states = list(reversed(states_r))
    result_nodes = {s: StateNode() for s in states}
    gds_components = [0.0] * MAX_ANTICIPATION_NEST_LEVEL

    for component_idx in range(nest_level):
        d_local = {}
        d_global = {}
        a_values = {}

        # seedD
        for s in states:
            if component_idx == 0:
                d_local[s] = compute_intrinsic_desire(s)
            else:
                d_local[s] = result_nodes[s].a[component_idx - 1]

        # propagateD (same as code)
        for s in states_r:
            d_global[s] = d_local[s]
            for prob, next_s in get_transitions(s, config):
                d_global[s] += d_global[next_s] * prob

        # compute_A — NAIVE VERSION: use D_global(s') directly
        for s in states_r:
            transitions = get_transitions(s, config)
            if not transitions:
                a_values[s] = 0.0
                continue

            # Use absolute D_global of children (not perspective)
            sum_d = 0.0
            for prob, next_s in transitions:
                sum_d += prob * d_global[next_s]
            avg_d = sum_d

            weighted_var = 0.0
            for prob, next_s in transitions:
                diff = d_global[next_s] - avg_d
                weighted_var += prob * diff * diff

            a_values[s] = math.sqrt(weighted_var)

        # GDS computation (same as code)
        from collections import defaultdict
        steps = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        steps[initial_state][0][0] = 1.0
        steps[initial_state][0][1] = a_values[initial_state]

        for s in states:
            for prob, next_s in get_transitions(s, config):
                for step_i, (reach_p, acc_a) in list(steps[s].items()):
                    entry = steps[next_s][step_i + 1]
                    entry[0] += reach_p * prob
                    entry[1] += acc_a * prob + a_values[next_s] * reach_p * prob

        for s in states:
            if is_terminal(s):
                for step_i, (reach_p, acc_a) in steps[s].items():
                    if step_i > 0:
                        gds_components[component_idx] += acc_a / step_i

        for s in states:
            result_nodes[s].a[component_idx] = a_values[s]
            if component_idx == 0:
                result_nodes[s].d_global = d_global[s]

    result = GameAnalysis(
        states=states,
        states_r=states_r,
        state_nodes=result_nodes,
        gds_components=gds_components,
    )
    result.game_design_score = sum(gds_components[:nest_level])
    return result


def compare_games():
    """Compare naive vs perspective desire across multiple games."""
    print("Perspective Desire Comparison Experiment")
    print("=" * 75)
    print()
    print("'Naive' = paper formulation (D_global of children)")
    print("'Perspective' = code implementation (D_global(s') - D_global(s))")
    print()

    games = [
        ("HpGame (5,5)", HpGame, None),
        ("HpGame_Rage (10%)", HpGameRage, HpGameRage.Config(critical_chance=0.10)),
        ("HpGame_Rage (13%)", HpGameRage, HpGameRage.Config(critical_chance=0.13)),
        ("LaneGame", LaneGame, None),
    ]

    print(f"{'Game':<25} {'Naive GDS':>12} {'Persp GDS':>12} {'Diff':>10} {'%Diff':>8}")
    print("-" * 70)

    for name, game_cls, config in games:
        naive = analyze_naive(
            initial_state=game_cls.initial_state(),
            is_terminal=game_cls.is_terminal,
            get_transitions=game_cls.get_transitions,
            compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
            config=config,
            nest_level=5,
        )
        perspective = analyze(
            initial_state=game_cls.initial_state(),
            is_terminal=game_cls.is_terminal,
            get_transitions=game_cls.get_transitions,
            compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
            config=config,
            nest_level=5,
        )

        diff = perspective.game_design_score - naive.game_design_score
        pct = diff / max(0.001, naive.game_design_score) * 100

        print(f"{name:<25} {naive.game_design_score:>12.6f} {perspective.game_design_score:>12.6f} "
              f"{diff:>+10.6f} {pct:>+7.1f}%")

    print()
    print("=" * 75)
    print("Detailed State-by-State Comparison: HpGame")
    print("=" * 75)

    naive = analyze_naive(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    persp = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )

    print(f"\n{'State':<12} {'N_A₁':>8} {'P_A₁':>8} {'Diff':>8} {'N_A₂':>8} {'P_A₂':>8} {'N_total':>8} {'P_total':>8}")
    print("-" * 70)

    non_terminal = [(s, naive.state_nodes[s]) for s in naive.states if not HpGame.is_terminal(s)]
    non_terminal.sort(key=lambda x: (x[0][0], x[0][1]))

    for s, n_node in non_terminal:
        p_node = persp.state_nodes[s]
        na1 = n_node.a[0]
        pa1 = p_node.a[0]
        na2 = n_node.a[1]
        pa2 = p_node.a[1]
        n_total = n_node.sum_a()
        p_total = p_node.sum_a()
        diff = pa1 - na1
        print(f"HP({s[0]},{s[1]})     {na1:>8.4f} {pa1:>8.4f} {diff:>+8.4f} {na2:>8.4f} {pa2:>8.4f} {n_total:>8.4f} {p_total:>8.4f}")

    # Component-by-component comparison
    print()
    print("Component comparison:")
    print(f"{'Comp':<6} {'Naive':>12} {'Perspective':>12} {'Diff':>12}")
    for i in range(5):
        n = naive.gds_components[i]
        p = persp.gds_components[i]
        print(f"A{i+1:<5} {n:>12.6f} {p:>12.6f} {p-n:>+12.6f}")

    print()
    print("=" * 75)
    print("MATHEMATICAL PROOF OF EQUIVALENCE")
    print("=" * 75)
    print("""
FINDING: Naive and Perspective formulations produce IDENTICAL results.

PROOF:
  Let X_i = D_global(s_i) for child states s_i with probabilities p_i.
  Let c = D_global(s) for parent state s.

  Naive anticipation:
    μ_naive = Σ p_i · X_i
    A_naive = sqrt(Σ p_i · (X_i - μ_naive)²)

  Perspective anticipation:
    Y_i = X_i - c  (perspective desire)
    μ_persp = Σ p_i · Y_i = Σ p_i · (X_i - c) = μ_naive - c
    A_persp = sqrt(Σ p_i · (Y_i - μ_persp)²)
            = sqrt(Σ p_i · ((X_i - c) - (μ_naive - c))²)
            = sqrt(Σ p_i · (X_i - μ_naive)²)
            = A_naive

  QED: Subtracting a constant c from all values doesn't change variance.

  This holds for ALL components (A₁, A₂, ...) because:
  - For A₁: D_global(s) is the same constant for all children of s
  - For A₂: the A₁ "desire" propagation uses the same d_global subtraction
  - The propagation step d_global[s] += d_global[next_s] * prob is identical
  - Only the A computation step differs, and it's variance-invariant

  IMPLICATION: Vol 2's "Perspective Desire" is a conceptual framing,
  not a computational change. Both formulations are mathematically equivalent.
  The perspective framing is useful for INTERPRETATION (measuring change
  rather than absolute position), but the numbers are the same.
""")


if __name__ == "__main__":
    compare_games()
