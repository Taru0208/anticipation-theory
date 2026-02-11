"""Deep Analysis of the GA-Optimal Game

The GA (v2 symmetric) found an optimal game with GDS ≈ 0.979:
  HP = 3, 8 outcomes, all asymmetric

This experiment investigates WHY this game is optimal by:
1. Decomposing the engagement by game state
2. Comparing with theoretical upper bounds
3. Understanding the relationship between game length and GDS
4. Testing whether the optimal design is a "disguised coin toss"
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions
from toa.games.coin_toss import CoinToss
from toa.games.hpgame import HpGame


def build_optimal_game():
    """Reconstruct the GA-optimal symmetric game."""
    # From GA v2 output:
    # HP=3, outcomes: mix of (-1,0), (0,-1), (-3,+1), (+1,-3), (-3,0), (0,-3)
    # with symmetric probabilities

    max_hp = 3

    # The GA converges to games with these probability-effect pairs
    # Using the specific values from the run
    outcomes = [
        (0.057, +1, -3),   # P1 heals, P2 takes massive damage
        (0.057, -3, +1),   # Mirror
        (0.248, 0, -1),    # P2 takes damage
        (0.248, -1, 0),    # P1 takes damage
        (0.137, 0, -3),    # P2 takes massive damage
        (0.137, -3, 0),    # P1 takes massive damage
        (0.057, +1, -3),   # Duplicate of first pair
        (0.057, -3, +1),   # Mirror
    ]

    # Merge duplicate outcomes
    merged = {}
    for p, d1, d2 in outcomes:
        key = (d1, d2)
        merged[key] = merged.get(key, 0) + p

    outcomes_clean = [(p, d1, d2) for (d1, d2), p in merged.items()]
    # Normalize
    total_p = sum(p for p, _, _ in outcomes_clean)
    outcomes_clean = [(p / total_p, d1, d2) for p, d1, d2 in outcomes_clean]

    def initial_state():
        return (max_hp, max_hp)

    def is_terminal(state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []
        transitions = []
        for p, d1, d2 in outcomes_clean:
            new1 = max(0, min(max_hp, hp1 + d1))
            new2 = max(0, min(max_hp, hp2 + d2))
            transitions.append((p, (new1, new2)))
        return sanitize_transitions(transitions)

    def desire(state):
        return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0

    return initial_state, is_terminal, get_transitions, desire, outcomes_clean


def analyze_state_contributions():
    """Understand which states contribute most to the total GDS."""
    print("State-by-State Engagement Decomposition")
    print("=" * 75)

    init, is_term, get_trans, desire, outcomes = build_optimal_game()

    result = analyze(
        initial_state=init(),
        is_terminal=is_term,
        get_transitions=get_trans,
        compute_intrinsic_desire=desire,
        nest_level=5,
    )

    print(f"\nGA-Optimal Game: GDS = {result.game_design_score:.6f}")
    print(f"Outcomes: {outcomes}")
    print()

    # For each non-terminal state, show its contribution
    non_terminal = [s for s in result.states if not is_term(s)]
    print(f"{'State':<12} {'D_global':>10} {'A₁':>8} {'A₂':>8} {'A₃':>8} {'Total_A':>10}")
    print("-" * 60)

    for s in sorted(non_terminal):
        node = result.state_nodes[s]
        total = node.sum_a()
        print(f"HP({s[0]},{s[1]})     {node.d_global:>10.4f} "
              f"{node.a[0]:>8.4f} {node.a[1]:>8.4f} {node.a[2]:>8.4f} {total:>10.4f}")

    # Key insight: why is A₁ uniformly high?
    a1_vals = [result.state_nodes[s].a[0] for s in non_terminal]
    print(f"\nA₁ statistics:")
    print(f"  Mean: {sum(a1_vals)/len(a1_vals):.4f}")
    print(f"  Min:  {min(a1_vals):.4f}")
    print(f"  Max:  {max(a1_vals):.4f}")
    print(f"  All > 0.35: {all(a > 0.35 for a in a1_vals)}")


def coin_toss_comparison():
    """Is the optimal game just a 'disguised coin toss'?"""
    print()
    print("=" * 75)
    print("'Disguised Coin Toss' Hypothesis")
    print("=" * 75)
    print("""
If the optimal game is equivalent to a multi-round coin toss,
its A₁ should be ≈ 0.5 at every state and A₂+ ≈ 0.

The coin toss has GDS = 0.5 (single turn).
A 3-round coin toss (best of 3) would have GDS involving both A₁ and A₂.
""")

    # Best of 3 coin toss
    def bo3_initial():
        return (0, 0)  # (p1_wins, p2_wins)

    def bo3_terminal(state):
        return state[0] >= 2 or state[1] >= 2

    def bo3_transitions(state, config=None):
        if state[0] >= 2 or state[1] >= 2:
            return []
        return [
            (0.5, (state[0] + 1, state[1])),
            (0.5, (state[0], state[1] + 1)),
        ]

    def bo3_desire(state):
        return 1.0 if state[0] >= 2 else 0.0

    bo3 = analyze(
        initial_state=bo3_initial(),
        is_terminal=bo3_terminal,
        get_transitions=bo3_transitions,
        compute_intrinsic_desire=bo3_desire,
        nest_level=5,
    )

    print(f"  Best-of-3 Coin Toss GDS: {bo3.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={bo3.gds_components[i]:.4f}' for i in range(5) if bo3.gds_components[i] > 0.001)}")

    # Best of 5 coin toss
    def bo5_terminal(state):
        return state[0] >= 3 or state[1] >= 3

    def bo5_transitions(state, config=None):
        if state[0] >= 3 or state[1] >= 3:
            return []
        return [
            (0.5, (state[0] + 1, state[1])),
            (0.5, (state[0], state[1] + 1)),
        ]

    def bo5_desire(state):
        return 1.0 if state[0] >= 3 else 0.0

    bo5 = analyze(
        initial_state=(0, 0),
        is_terminal=bo5_terminal,
        get_transitions=bo5_transitions,
        compute_intrinsic_desire=bo5_desire,
        nest_level=5,
    )

    print(f"  Best-of-5 Coin Toss GDS: {bo5.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={bo5.gds_components[i]:.4f}' for i in range(5) if bo5.gds_components[i] > 0.001)}")

    # GA optimal
    init, is_term, get_trans, desire, _ = build_optimal_game()
    ga = analyze(
        initial_state=init(),
        is_terminal=is_term,
        get_transitions=get_trans,
        compute_intrinsic_desire=desire,
        nest_level=5,
    )

    print(f"\n  GA-Optimal GDS:         {ga.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={ga.gds_components[i]:.4f}' for i in range(5) if ga.gds_components[i] > 0.001)}")

    # HpGame for reference
    hp = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )

    print(f"\n  HpGame (5,5) GDS:       {hp.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={hp.gds_components[i]:.4f}' for i in range(5) if hp.gds_components[i] > 0.001)}")

    # Analysis
    print(f"""
  ANALYSIS:
  ─────────
  Best-of-3 coin toss:   {bo3.game_design_score:.4f}
  Best-of-5 coin toss:   {bo5.game_design_score:.4f}
  GA-Optimal:            {ga.game_design_score:.4f}
  HpGame:                {hp.game_design_score:.4f}

  The GA-optimal game is NOT a disguised coin toss:
  - Bo3 has GDS {bo3.game_design_score:.3f}, Bo5 has {bo5.game_design_score:.3f}
  - GA-optimal has {ga.game_design_score:.3f} — {"higher" if ga.game_design_score > bo5.game_design_score else "lower"} than Bo5
  """)


def theoretical_bounds():
    """Explore theoretical upper bounds for GDS."""
    print("=" * 75)
    print("Theoretical Upper Bounds")
    print("=" * 75)
    print("""
  For binary (win/loss) games:
  - A₁ ≤ 0.5 per state (from the paper's proof)
  - If a game has N non-terminal states, each with A₁ = 0.5, and each
    state is visited on average once per game, then:
    GDS_A1_upper ≤ 0.5 × N / avg_game_length

  But higher components (A₂, A₃, ...) can add to GDS unboundedly
  (Unbound Conjecture).
  """)

    # Test: what if we make a game where every state has A₁ = 0.5?
    # This requires every state to be a 50/50 fork.
    print("  Best-of-N coin toss series (all states have A₁ = 0.5):")
    print(f"  {'N':>4}  {'GDS':>10}  {'A₁':>8}  {'A₂':>8}  {'Depth%':>8}")
    print(f"  {'-'*45}")

    for n_wins in range(1, 8):
        def make_terminal(nw):
            def t(state):
                return state[0] >= nw or state[1] >= nw
            return t

        def make_desire(nw):
            def d(state):
                return 1.0 if state[0] >= nw else 0.0
            return d

        def make_transitions(nw):
            def t(state, config=None):
                if state[0] >= nw or state[1] >= nw:
                    return []
                return [(0.5, (state[0]+1, state[1])), (0.5, (state[0], state[1]+1))]
            return t

        try:
            r = analyze(
                initial_state=(0, 0),
                is_terminal=make_terminal(n_wins),
                get_transitions=make_transitions(n_wins),
                compute_intrinsic_desire=make_desire(n_wins),
                nest_level=min(10, 2*n_wins),
            )
            total = r.game_design_score
            a1 = r.gds_components[0]
            a2 = r.gds_components[1]
            depth = sum(r.gds_components[1:10]) / max(0.001, total) * 100
            states = len(r.states)
            print(f"  Bo{2*n_wins-1:>2}  {total:>10.6f}  {a1:>8.4f}  {a2:>8.4f}  {depth:>7.1f}%  ({states} states)")
        except Exception as e:
            print(f"  Bo{2*n_wins-1:>2}  [error: {e}]")

    print("""
  OBSERVATION: Best-of-N coin toss GDS grows with N.
  This is the "purest" form of the Unbound Conjecture:
  even with identical transitions at every state (50/50),
  more turns → more engagement. Each turn adds uncertainty
  that compounds through higher-order components.
""")


def kill_probability_analysis():
    """Analyze how the probability of one-turn kills affects engagement."""
    print("=" * 75)
    print("One-Turn Kill Probability Analysis")
    print("=" * 75)
    print("  How does the probability of instant kills affect GDS?")
    print()

    print(f"  {'Kill_p':>8}  {'GDS':>8}  {'A₁':>8}  {'A₂':>8}  {'States':>8}  {'AvgLen':>8}")
    print(f"  {'-'*55}")

    for kill_pct in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        kill_p = kill_pct / 100.0
        safe_p = 1.0 - kill_p

        # Symmetric game: hp=3, outcomes are:
        # kill_p/2: P1 instant-kills P2 (0, -3)
        # kill_p/2: P2 instant-kills P1 (-3, 0)
        # safe_p/2: P2 takes 1 damage (0, -1)
        # safe_p/2: P1 takes 1 damage (-1, 0)
        outcomes = []
        if kill_p > 0:
            outcomes.append((kill_p / 2, 0, -3))
            outcomes.append((kill_p / 2, -3, 0))
        if safe_p > 0:
            outcomes.append((safe_p / 2, 0, -1))
            outcomes.append((safe_p / 2, -1, 0))

        def make_game(hp, outs):
            def initial(): return (hp, hp)
            def terminal(s): return s[0] <= 0 or s[1] <= 0
            def trans(s, c=None):
                if s[0] <= 0 or s[1] <= 0: return []
                t = []
                for p, d1, d2 in outs:
                    t.append((p, (max(0, s[0]+d1), max(0, s[1]+d2))))
                return sanitize_transitions(t)
            def d(s): return 1.0 if s[0] > 0 and s[1] <= 0 else 0.0
            return initial, terminal, trans, d

        init, term, trans, d = make_game(3, outcomes)
        try:
            r = analyze(
                initial_state=init(),
                is_terminal=term,
                get_transitions=trans,
                compute_intrinsic_desire=d,
                nest_level=5,
            )
            # Estimate average game length from simulation
            from toa.simulate import simulate_gds
            sim = simulate_gds(
                initial_state=init(),
                is_terminal=term,
                get_transitions=trans,
                compute_intrinsic_desire=d,
                nest_level=1,
                num_simulations=5000,
                seed=42,
            )
            avg_len = sim["mean_game_length"]

            print(f"  {kill_pct:>6}%  {r.game_design_score:>8.4f}  {r.gds_components[0]:>8.4f}  "
                  f"{r.gds_components[1]:>8.4f}  {len(r.states):>8}  {avg_len:>8.1f}")
        except Exception as e:
            print(f"  {kill_pct:>6}%  [error: {e}]")

    print("""
  INSIGHT: The relationship between instant-kill probability and GDS
  reveals the tension between:
  - Immediate drama (high A₁ from uncertain outcomes)
  - Strategic depth (high A₂+ from multi-turn planning)

  100% kills = single-turn coin toss (GDS = 0.500)
  0% kills = multi-turn gradual combat (like HpGame)
  Somewhere in between = optimal balance
""")


if __name__ == "__main__":
    analyze_state_contributions()
    coin_toss_comparison()
    theoretical_bounds()
    kill_probability_analysis()
