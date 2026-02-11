"""Player Choice and Anticipation — Do meaningful decisions increase GDS?

All existing ToA models are pure chance: each state has fixed transition probabilities.
Real games involve player decisions. This experiment explores what happens when we
add strategic choice to the framework.

Key question: Does player agency increase or decrease measurable engagement?

Approach:
  1. Build a combat game where players choose actions each turn (attack, defend, special)
  2. Model the game under different "play models":
     - Random: uniform distribution over actions → pure chance Markov chain
     - Nash: game-theoretic optimal mixed strategy → also a Markov chain
     - Aggressive: always attack → deterministic selection
  3. Compare GDS under each model
  4. Build a "choice space" game where the number of actions varies, compare GDS

The insight: in ToA's Markov framework, player "choice" manifests as the
*probability distribution* over actions. Different play styles → different
distributions → different GDS scores. The DIFFERENCE in GDS between random
and optimal play may capture something about how much "meaningful choice" exists.
"""

import sys
import os
import math
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


# ─── Game: Tactical Combat ───────────────────────────────────────────────────
#
# Two players with HP. Each turn, both simultaneously choose an action:
#   - ATTACK: Deal damage to opponent. Risky — you take counter-damage if they also attack.
#   - DEFEND: Reduce incoming damage. Safe but no progress.
#   - SPECIAL: High-risk, high-reward. Big damage but leaves you vulnerable.
#
# Outcomes are resolved by the (action1, action2) pair.

# Outcome table: (p1_hp_change, p2_hp_change)
# action_idx: 0=attack, 1=defend, 2=special
OUTCOMES = {
    # attack vs attack: both take damage (mutual exchange)
    (0, 0): [(-1, -1)],
    # attack vs defend: attacker bounces off, takes chip damage
    (0, 1): [(-1, 0)],
    # attack vs special: attacker punishes vulnerable special user
    (0, 2): [(0, -2)],
    # defend vs attack: mirror
    (1, 0): [(0, -1)],
    # defend vs defend: slow attrition (prevents infinite loops)
    (1, 1): [(-1, -1)],
    # defend vs special: special bypasses defense
    (1, 2): [(-1, 0)],
    # special vs attack: mirror
    (2, 0): [(-2, 0)],
    # special vs defend: mirror
    (2, 1): [(0, -1)],
    # special vs special: mutual destruction
    (2, 2): [(-2, -2)],
}

ACTION_NAMES = ["Attack", "Defend", "Special"]
N_ACTIONS = len(ACTION_NAMES)


def build_choice_game(max_hp, play_model_p1, play_model_p2, outcomes=None, n_actions=None):
    """Build a game with given action probability distributions.

    play_model_p1/p2: function(hp1, hp2) → [prob_action_0, ..., prob_action_n]
    outcomes: dict mapping (action1, action2) → [(d1, d2), ...]
    """
    if outcomes is None:
        outcomes = OUTCOMES
    if n_actions is None:
        n_actions = N_ACTIONS

    def initial_state():
        return (max_hp, max_hp)

    def is_terminal(state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        p1_probs = play_model_p1(hp1, hp2)
        p2_probs = play_model_p2(hp1, hp2)

        transitions = []
        for a1 in range(n_actions):
            for a2 in range(n_actions):
                joint_prob = p1_probs[a1] * p2_probs[a2]
                if joint_prob < 1e-10:
                    continue

                outcome_list = outcomes.get((a1, a2), [(0, 0)])
                for d1, d2 in outcome_list:
                    new_hp1 = max(0, min(max_hp, hp1 + d1))
                    new_hp2 = max(0, min(max_hp, hp2 + d2))
                    transitions.append((joint_prob / len(outcome_list), (new_hp1, new_hp2)))

        return sanitize_transitions(transitions)

    def desire(state):
        if state[0] > 0 and state[1] <= 0:
            return 1.0
        return 0.0

    return initial_state, is_terminal, get_transitions, desire


# ─── Play Models ──────────────────────────────────────────────────────────────

def random_play(hp1, hp2):
    """Uniform random over all actions."""
    return [1/N_ACTIONS] * N_ACTIONS


def aggressive_play(hp1, hp2):
    """Always attack."""
    return [1.0, 0.0, 0.0]


def defensive_play(hp1, hp2):
    """Always defend."""
    return [0.0, 1.0, 0.0]


def special_spam(hp1, hp2):
    """Always special."""
    return [0.0, 0.0, 1.0]


def hp_adaptive_play(hp1, hp2):
    """Adapt based on HP difference.
    - Low HP → more defensive
    - High HP → more aggressive
    - Close HP → more special (high-risk to break ties)
    """
    hp_ratio = hp1 / max(1, hp2)
    if hp_ratio > 1.3:
        # Winning → aggressive
        return [0.6, 0.2, 0.2]
    elif hp_ratio < 0.7:
        # Losing → defensive with occasional desperation special
        return [0.2, 0.5, 0.3]
    else:
        # Close → balanced with emphasis on special
        return [0.3, 0.3, 0.4]


def compute_nash_equilibrium(max_hp, outcomes=None, n_actions=None):
    """Compute Nash equilibrium mixed strategies for each HP state.

    For each (hp1, hp2), find the mixed strategy Nash equilibrium
    of the one-shot action game. Use support enumeration for small action space.

    This is a simplified computation — for the one-step game at each state,
    assuming the rest plays out with equal expected value. The real Nash would
    need backward induction, but this captures the essential structure.
    """
    if outcomes is None:
        outcomes = OUTCOMES
    if n_actions is None:
        n_actions = N_ACTIONS

    # For simplicity, compute a state-independent Nash of the one-shot game
    # based on the payoff matrix.
    payoff_matrix = []
    for a1 in range(n_actions):
        row = []
        for a2 in range(n_actions):
            outcome_list = outcomes.get((a1, a2), [(0, 0)])
            # Average net advantage for P1: damage to P2 minus damage to P1
            avg_payoff = sum((-d2) - (-d1) for d1, d2 in outcome_list) / len(outcome_list)
            row.append(avg_payoff)
        payoff_matrix.append(row)

    # Solve for Nash equilibrium of this 3x3 zero-sum game
    # Using linear programming approach for 2-player zero-sum
    nash_probs = _solve_zero_sum_nash(payoff_matrix)

    def nash_play(hp1, hp2):
        return nash_probs

    return nash_play, payoff_matrix


def _solve_zero_sum_nash(payoff_matrix):
    """Solve a zero-sum game Nash equilibrium via support enumeration.

    For a 3x3 game, we can enumerate all possible supports.
    """
    n = len(payoff_matrix)
    best_value = -float('inf')
    best_strategy = [1/n] * n

    # Try all non-empty subsets of actions as support
    for support_size in range(1, n + 1):
        for support in itertools.combinations(range(n), support_size):
            # For the row player in a zero-sum game, find the mixed strategy
            # over the support that maximizes the minimum expected payoff
            strategy = _solve_support(payoff_matrix, support)
            if strategy is None:
                continue

            # Compute expected payoff against best response
            min_payoff = float('inf')
            for j in range(n):
                expected = sum(strategy[i] * payoff_matrix[i][j] for i in range(n))
                min_payoff = min(min_payoff, expected)

            if min_payoff > best_value:
                best_value = min_payoff
                best_strategy = strategy

    return best_strategy


def _solve_support(payoff_matrix, support):
    """Find the equalizing strategy for a given support in a zero-sum game.

    The mixed strategy makes the opponent indifferent among all their actions.
    This means: for all columns j, Σ_i p_i * M[i][j] = v (constant).
    """
    n = len(payoff_matrix)
    m = len(support)

    if m == 1:
        probs = [0.0] * n
        probs[support[0]] = 1.0
        return probs

    if m == 2:
        i, j = support
        # Try to find p such that p*M[i][col] + (1-p)*M[j][col] is constant for all cols
        # p*M[i][0] + (1-p)*M[j][0] = p*M[i][1] + (1-p)*M[j][1]
        # Check all column pairs
        best_probs = None
        for c1 in range(n):
            for c2 in range(c1 + 1, n):
                diff_i = payoff_matrix[i][c1] - payoff_matrix[i][c2]
                diff_j = payoff_matrix[j][c1] - payoff_matrix[j][c2]
                denom = diff_i - diff_j
                if abs(denom) < 1e-10:
                    continue
                p = -diff_j / denom
                if p < -0.01 or p > 1.01:
                    continue
                p = max(0, min(1, p))
                probs = [0.0] * n
                probs[i] = p
                probs[j] = 1 - p
                return probs

    if m == 3:
        # Full support: solve 3x3 system
        # p1*M[0][j] + p2*M[1][j] + p3*M[2][j] = v for all j
        # p1 + p2 + p3 = 1
        # This gives us 4 equations, 4 unknowns (p1, p2, p3, v)
        # From j=0,1: (M[0][0]-M[0][1])p1 + (M[1][0]-M[1][1])p2 + (M[2][0]-M[2][1])p3 = 0
        # From j=0,2: (M[0][0]-M[0][2])p1 + (M[1][0]-M[1][2])p2 + (M[2][0]-M[2][2])p3 = 0
        # p1 + p2 + p3 = 1
        M = payoff_matrix
        a11 = M[0][0] - M[0][1]
        a12 = M[1][0] - M[1][1]
        a13 = M[2][0] - M[2][1]
        a21 = M[0][0] - M[0][2]
        a22 = M[1][0] - M[1][2]
        a23 = M[2][0] - M[2][2]

        # Solve:
        # a11*p1 + a12*p2 + a13*p3 = 0
        # a21*p1 + a22*p2 + a23*p3 = 0
        # p1 + p2 + p3 = 1

        # Express p1 = 1 - p2 - p3, substitute:
        # a11*(1-p2-p3) + a12*p2 + a13*p3 = 0
        # → (a12-a11)*p2 + (a13-a11)*p3 = -a11
        # a21*(1-p2-p3) + a22*p2 + a23*p3 = 0
        # → (a22-a21)*p2 + (a23-a21)*p3 = -a21

        b11 = a12 - a11
        b12 = a13 - a11
        b21 = a22 - a21
        b22 = a23 - a21

        det = b11 * b22 - b12 * b21
        if abs(det) < 1e-10:
            return None

        p2 = (-a11 * b22 - (-a21) * b12) / det
        p3 = (b11 * (-a21) - b21 * (-a11)) / det
        p1 = 1 - p2 - p3

        if p1 < -0.01 or p2 < -0.01 or p3 < -0.01:
            return None

        probs = [max(0, p1), max(0, p2), max(0, p3)]
        total = sum(probs)
        if total < 0.01:
            return None
        probs = [p / total for p in probs]
        return probs

    return None


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_game(max_hp, play_model_p1, play_model_p2, label, nest_level=5, outcomes=None):
    """Analyze a game and return results."""
    init, is_term, trans, desire = build_choice_game(max_hp, play_model_p1, play_model_p2, outcomes=outcomes)
    result = analyze(
        initial_state=init(),
        is_terminal=is_term,
        get_transitions=trans,
        compute_intrinsic_desire=desire,
        nest_level=nest_level,
    )

    d0 = result.state_nodes[init()].d_global
    non_terminal = [s for s in result.states if not is_term(s)]

    comps = result.gds_components[:nest_level]
    comp_str = " ".join(f"A{i+1}={c:.4f}" for i, c in enumerate(comps) if c > 0.001)

    print(f"  {label:<35} GDS={result.game_design_score:.4f}  D₀={d0:.3f}  "
          f"States={len(result.states)}  {comp_str}")

    return result


def measure_choice_impact(max_hp, nash_model, nest_level=5, outcomes=None):
    """Measure the 'choice impact' — GDS difference between random and Nash play.

    A high difference means player decisions significantly change the game's
    engagement structure. A low difference means choices don't matter much.
    """
    init_r, is_term_r, trans_r, desire_r = build_choice_game(max_hp, random_play, random_play, outcomes=outcomes)
    init_n, is_term_n, trans_n, desire_n = build_choice_game(max_hp, nash_model, nash_model, outcomes=outcomes)

    result_r = analyze(
        initial_state=init_r(), is_terminal=is_term_r,
        get_transitions=trans_r, compute_intrinsic_desire=desire_r,
        nest_level=nest_level,
    )
    result_n = analyze(
        initial_state=init_n(), is_terminal=is_term_n,
        get_transitions=trans_n, compute_intrinsic_desire=desire_n,
        nest_level=nest_level,
    )

    return result_r.game_design_score, result_n.game_design_score


# ─── Experiments ──────────────────────────────────────────────────────────────

def experiment_1_play_styles():
    """Compare GDS under different play models for fixed HP."""
    print("=" * 80)
    print("EXPERIMENT 1: Play Styles and GDS (HP=5)")
    print("=" * 80)
    print()

    max_hp = 5
    nash_model, payoff_matrix = compute_nash_equilibrium(max_hp)

    print("  Payoff matrix (P1 net advantage = damage dealt - damage taken):")
    print(f"  {'':>10} {'Attack':>10} {'Defend':>10} {'Special':>10}")
    for i, name in enumerate(ACTION_NAMES):
        row = " ".join(f"{payoff_matrix[i][j]:>10.1f}" for j in range(N_ACTIONS))
        print(f"  {name:>10} {row}")

    nash_probs = nash_model(5, 5)
    print(f"\n  Nash equilibrium: {' '.join(f'{ACTION_NAMES[i]}={p:.3f}' for i, p in enumerate(nash_probs))}")
    print()

    models = [
        ("Random (uniform)", random_play, random_play),
        ("Nash vs Nash", nash_model, nash_model),
        ("Aggressive vs Aggressive", aggressive_play, aggressive_play),
        ("Defensive vs Defensive", defensive_play, defensive_play),
        ("Special vs Special", special_spam, special_spam),
        ("Adaptive vs Adaptive", hp_adaptive_play, hp_adaptive_play),
        ("Nash vs Random", nash_model, random_play),
        ("Nash vs Aggressive", nash_model, aggressive_play),
    ]

    results = {}
    for label, p1, p2 in models:
        result = analyze_game(max_hp, p1, p2, label)
        results[label] = result

    # Highlight key comparison
    gds_random = results["Random (uniform)"].game_design_score
    gds_nash = results["Nash vs Nash"].game_design_score
    delta = (gds_nash - gds_random) / gds_random * 100

    print()
    print(f"  Choice impact: Nash vs Random = {delta:+.1f}% GDS change")
    if gds_nash > gds_random:
        print("  → Strategic play INCREASES engagement!")
    else:
        print("  → Strategic play DECREASES engagement (reduces variance).")

    return results


def experiment_2_hp_sweep():
    """How does HP level interact with choice impact?"""
    print()
    print("=" * 80)
    print("EXPERIMENT 2: HP vs Choice Impact")
    print("=" * 80)
    print()

    print(f"  {'HP':>4}  {'GDS(Random)':>12}  {'GDS(Nash)':>12}  {'Delta':>8}  {'Choice Impact':>14}")
    print(f"  {'-'*58}")

    for hp in [3, 4, 5, 6, 7]:
        nash_model, _ = compute_nash_equilibrium(hp)
        gds_r, gds_n = measure_choice_impact(hp, nash_model)
        delta = (gds_n - gds_r) / max(0.001, gds_r) * 100
        print(f"  {hp:>4}  {gds_r:>12.4f}  {gds_n:>12.4f}  {delta:>+7.1f}%  {'★' * max(0, min(5, int(abs(delta)/5)))}")


def experiment_3_asymmetric_outcomes():
    """Test different outcome tables — how does the action design affect choice impact?"""
    print()
    print("=" * 80)
    print("EXPERIMENT 3: Outcome Design and Engagement")
    print("=" * 80)
    print()

    # Variant 1: Rock-Paper-Scissors-like (strong cyclical dominance)
    # NOTE: Every outcome must reduce someone's HP (no (0,0)) to avoid infinite cycles
    rps_outcomes = {
        (0, 0): [(-1, -1)],    # A vs A: mutual damage
        (0, 1): [(-1, 0)],     # A vs D: attacker bounces
        (0, 2): [(0, -2)],     # A beats S
        (1, 0): [(0, -1)],     # D beats A
        (1, 1): [(-1, -1)],    # D vs D: attrition
        (1, 2): [(-1, 0)],     # S beats D
        (2, 0): [(-2, 0)],     # S loses to A
        (2, 1): [(0, -1)],     # S beats D
        (2, 2): [(-2, -2)],    # S vs S: mutual destruction
    }

    # Variant 2: Dominant strategy exists (attack always best)
    dominant_outcomes = {
        (0, 0): [(-1, -1)],
        (0, 1): [(0, -2)],     # Attack beats defend hard
        (0, 2): [(0, -1)],     # Attack beats special
        (1, 0): [(-2, 0)],
        (1, 1): [(-1, -1)],    # Attrition (was (0,0))
        (1, 2): [(-1, 0)],
        (2, 0): [(-1, 0)],
        (2, 1): [(-1, 0)],     # Both lose with defend/special
        (2, 2): [(-1, -1)],
    }

    # Variant 3: High-variance outcomes (gambling)
    volatile_outcomes = {
        (0, 0): [(-1, -2), (-2, -1)],   # One side gets lucky
        (0, 1): [(-1, -1), (0, -2)],    # Mixed outcomes, always someone takes damage
        (0, 2): [(0, -3), (-2, 0)],     # High stakes
        (1, 0): [(-1, -1), (-2, 0)],
        (1, 1): [(-1, -1)],             # Attrition (was (0,0))
        (1, 2): [(-1, 0), (0, -1)],
        (2, 0): [(-3, 0), (0, -2)],
        (2, 1): [(0, -1), (-1, 0)],
        (2, 2): [(-3, -3), (-1, -1)],
    }

    variants = [
        ("Baseline (standard)", OUTCOMES),
        ("RPS-like (cyclical)", rps_outcomes),
        ("Dominant strategy", dominant_outcomes),
        ("High-variance", volatile_outcomes),
    ]

    for label, var_outcomes in variants:
        print(f"\n  --- {label} ---")

        try:
            nash_model, payoff = compute_nash_equilibrium(5, outcomes=var_outcomes)
            nash_probs = nash_model(5, 5)
            print(f"  Nash: {' '.join(f'{ACTION_NAMES[i]}={p:.3f}' for i, p in enumerate(nash_probs))}")

            analyze_game(5, random_play, random_play, "Random", outcomes=var_outcomes)
            analyze_game(5, nash_model, nash_model, "Nash", outcomes=var_outcomes)

            gds_r, gds_n = measure_choice_impact(5, nash_model, outcomes=var_outcomes)
            delta = (gds_n - gds_r) / max(0.001, gds_r) * 100
            print(f"  Choice impact: {delta:+.1f}%")
        except Exception as e:
            print(f"  Error: {e}")


def experiment_4_action_count():
    """What happens as we vary the number of available actions?"""
    print()
    print("=" * 80)
    print("EXPERIMENT 4: Number of Actions vs Engagement")
    print("=" * 80)
    print()
    print("  Do more choices = more engagement?")
    print()

    max_hp = 5

    for n_actions in [2, 3, 4, 5]:
        # Generate a balanced action set
        actions = []
        for i in range(n_actions):
            for j in range(n_actions):
                # Create outcomes based on relative action indices
                diff = (i - j) % n_actions
                if diff == 0:
                    actions.append(((i, j), [(-1, -1)]))  # Mirror
                elif diff <= n_actions // 2:
                    actions.append(((i, j), [(0, -diff)]))  # i beats j
                else:
                    actions.append(((i, j), [(-1 * (n_actions - diff), 0)]))  # j beats i

        custom_outcomes = dict(actions)

        def random_play_n(hp1, hp2, n=n_actions):
            return [1/n] * n

        def build_n_action_game(max_hp, play_model, outcomes, n):
            def initial_state():
                return (max_hp, max_hp)

            def is_terminal(state):
                return state[0] <= 0 or state[1] <= 0

            def get_transitions(state, config=None):
                hp1, hp2 = state
                if hp1 <= 0 or hp2 <= 0:
                    return []

                p1_probs = play_model(hp1, hp2)
                p2_probs = play_model(hp1, hp2)

                transitions = []
                for a1 in range(n):
                    for a2 in range(n):
                        joint_prob = p1_probs[a1] * p2_probs[a2]
                        if joint_prob < 1e-10:
                            continue
                        outcome_list = outcomes.get((a1, a2), [(0, 0)])
                        for d1, d2 in outcome_list:
                            new_hp1 = max(0, min(max_hp, hp1 + d1))
                            new_hp2 = max(0, min(max_hp, hp2 + d2))
                            transitions.append((joint_prob / len(outcome_list), (new_hp1, new_hp2)))

                return sanitize_transitions(transitions)

            def desire(state):
                return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0

            return initial_state, is_terminal, get_transitions, desire

        init, is_term, trans, desire = build_n_action_game(max_hp, random_play_n, custom_outcomes, n_actions)
        try:
            result = analyze(
                initial_state=init(), is_terminal=is_term,
                get_transitions=trans, compute_intrinsic_desire=desire, nest_level=5,
            )
            d0 = result.state_nodes[init()].d_global
            print(f"  {n_actions} actions:  GDS={result.game_design_score:.4f}  "
                  f"D₀={d0:.3f}  States={len(result.states)}  "
                  f"A₁={result.gds_components[0]:.4f}  A₂={result.gds_components[1]:.4f}")
        except Exception as e:
            print(f"  {n_actions} actions: Error — {e}")


if __name__ == "__main__":
    experiment_1_play_styles()
    experiment_2_hp_sweep()
    experiment_3_asymmetric_outcomes()
    experiment_4_action_count()
