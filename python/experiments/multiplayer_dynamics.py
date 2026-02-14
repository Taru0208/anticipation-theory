"""Multiplayer Dynamics — How does adding players affect engagement?

Key question: Does Free-for-All (3+ players) increase or decrease GDS?

All existing ToA models are 2-player. But most real competitive games
involve 3+ players (Battle Royale, party games, FFA modes).

We model:
1. FFA HP Battle: 3-4 players, each turn random pair fights
2. Battle Royale: N players, last man standing
3. Kingmaker effect: eliminated player affects remaining game
4. Comparison with equivalent 2-player models

Theoretical predictions:
- More players → more uncertainty → higher A₁?
- Elimination creates phase transitions → could boost A₂+
- But player 1's win probability drops (1/N) → lower base desire
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


# =============================================================
# Model 1: FFA HP Battle (3 players)
# =============================================================
# State: (hp1, hp2, hp3)
# Each turn: one random pair fights (3 possible pairs)
# In each fight: attacker deals 1 HP damage to defender (50/50 who attacks)
# When a player reaches 0 HP, they're eliminated
# Game ends when only 1 player remains
# Player 1 wins if they're the last standing

class FFABattle:
    """Free-for-all HP battle for N=3 players."""

    class Config:
        def __init__(self, hp=3):
            self.hp = hp

    @staticmethod
    def initial_state(hp=3):
        return (hp, hp, hp)

    @staticmethod
    def is_terminal(state):
        alive = sum(1 for h in state if h > 0)
        return alive <= 1

    @staticmethod
    def get_transitions(state, config=None):
        hp1, hp2, hp3 = state
        alive = [(i, state[i]) for i in range(3) if state[i] > 0]

        if len(alive) <= 1:
            return []

        transitions = []

        if len(alive) == 2:
            # 2 players left — they fight each other
            i, hi = alive[0]
            j, hj = alive[1]
            # 50% i hits j, 50% j hits i
            s_list = list(state)

            s1 = s_list[:]
            s1[j] = max(0, hj - 1)
            s2 = s_list[:]
            s2[i] = max(0, hi - 1)

            transitions.append((0.5, tuple(s1)))
            transitions.append((0.5, tuple(s2)))
        else:
            # 3 players alive — random pair fights
            # 3 possible pairs, each equally likely (1/3)
            pairs = [(0, 1), (0, 2), (1, 2)]
            for i, j in pairs:
                hi, hj = state[i], state[j]
                # 50% i hits j, 50% j hits i
                s1 = list(state)
                s1[j] = max(0, hj - 1)
                s2 = list(state)
                s2[i] = max(0, hi - 1)

                transitions.append((1.0/6.0, tuple(s1)))  # 1/3 pair * 1/2 direction
                transitions.append((1.0/6.0, tuple(s2)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2, hp3 = state
        # Player 1 wins if they're alive and everyone else is dead
        if hp1 > 0 and hp2 <= 0 and hp3 <= 0:
            return 1.0
        return 0.0


# =============================================================
# Model 2: FFA HP Battle (4 players)
# =============================================================

class FFABattle4:
    """Free-for-all HP battle for N=4 players."""

    @staticmethod
    def initial_state():
        return (2, 2, 2, 2)  # Lower HP to keep state space manageable

    @staticmethod
    def is_terminal(state):
        alive = sum(1 for h in state if h > 0)
        return alive <= 1

    @staticmethod
    def get_transitions(state, config=None):
        n = len(state)
        alive_indices = [i for i in range(n) if state[i] > 0]

        if len(alive_indices) <= 1:
            return []

        transitions = []

        # All possible pairs from alive players
        pairs = []
        for a in range(len(alive_indices)):
            for b in range(a + 1, len(alive_indices)):
                pairs.append((alive_indices[a], alive_indices[b]))

        num_pairs = len(pairs)
        pair_prob = 1.0 / num_pairs

        for i, j in pairs:
            # 50% i hits j, 50% j hits i
            s1 = list(state)
            s1[j] = max(0, state[j] - 1)
            s2 = list(state)
            s2[i] = max(0, state[i] - 1)

            transitions.append((pair_prob * 0.5, tuple(s1)))
            transitions.append((pair_prob * 0.5, tuple(s2)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        if state[0] > 0 and all(state[i] <= 0 for i in range(1, len(state))):
            return 1.0
        return 0.0


# =============================================================
# Model 3: FFA with Focus Fire (strategic targeting)
# =============================================================
# In this model, alive opponents always attack the leading player
# (highest HP). This creates a "target the leader" dynamic.

class FFAFocusFire:
    """FFA where the leading player gets focused."""

    @staticmethod
    def initial_state():
        return (3, 3, 3)

    @staticmethod
    def is_terminal(state):
        alive = sum(1 for h in state if h > 0)
        return alive <= 1

    @staticmethod
    def get_transitions(state, config=None):
        alive = [(i, state[i]) for i in range(3) if state[i] > 0]
        if len(alive) <= 1:
            return []

        transitions = []

        if len(alive) == 2:
            # Standard 1v1
            i, hi = alive[0]
            j, hj = alive[1]
            s1 = list(state)
            s1[j] = max(0, hj - 1)
            s2 = list(state)
            s2[i] = max(0, hi - 1)
            transitions.append((0.5, tuple(s1)))
            transitions.append((0.5, tuple(s2)))
        else:
            # 3 alive: each player has equal chance of being "active" (1/3)
            # Active player attacks the highest-HP opponent
            # If tie, attack lowest-index
            for active_idx in range(3):
                if state[active_idx] <= 0:
                    continue
                # Find target: highest HP among others
                others = [(i, state[i]) for i in range(3) if i != active_idx and state[i] > 0]
                if not others:
                    continue
                target = max(others, key=lambda x: x[1])[0]

                s = list(state)
                s[target] = max(0, state[target] - 1)
                transitions.append((1.0 / 3.0, tuple(s)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        if state[0] > 0 and all(state[i] <= 0 for i in range(1, 3)):
            return 1.0
        return 0.0


# =============================================================
# Model 4: Survival desire (place top 2 out of 3)
# =============================================================
# In many battle royale games, "placing well" matters, not just winning.
# What if desire = 1.0 for 1st AND 2nd place (not last)?

class FFASurvival:
    """FFA where desire = 1.0 if player 1 is NOT eliminated first."""

    @staticmethod
    def initial_state():
        return (3, 3, 3)

    @staticmethod
    def is_terminal(state):
        alive = sum(1 for h in state if h > 0)
        return alive <= 1

    @staticmethod
    def get_transitions(state, config=None):
        # Same as FFABattle
        return FFABattle.get_transitions(state, config)

    @staticmethod
    def compute_intrinsic_desire(state):
        # Player 1 "wins" if they outlast at least one opponent
        # In terminal state (1 alive), check if P1 was NOT the first eliminated
        # Simplified: P1 desire = 1.0 if P1 is alive in terminal, or if P1 outlasted someone
        hp1 = state[0]
        alive = sum(1 for h in state if h > 0)
        if alive <= 1:
            # Terminal: P1 wins if alive (came 1st)
            if hp1 > 0:
                return 1.0
            # P1 is dead. Did P1 die first or second?
            # We can't tell from final state alone, so use a simplified version:
            # Check if at least one other is also dead (P1 didn't die alone = not last)
            others_dead = sum(1 for i in range(1, 3) if state[i] <= 0)
            # Actually in terminal, exactly 0 or 1 alive
            # If P1 dead and 1 other alive → P1 placed 2nd or 3rd (can't distinguish)
            # Simplification: partial desire based on when eliminated
            return 0.0  # Conservative: only winning counts
        return 0.0


# =============================================================
# Reference: 2-player HpGame for comparison
# =============================================================

def analyze_2p(hp):
    """Standard 2-player HP battle for comparison."""
    def initial():
        return (hp, hp)

    def is_term(state):
        return state[0] <= 0 or state[1] <= 0

    def transitions(state, config=None):
        h1, h2 = state
        if h1 <= 0 or h2 <= 0:
            return []
        return sanitize_transitions([
            (1.0/3.0, (h1, h2-1)),
            (1.0/3.0, (h1-1, h2-1)),
            (1.0/3.0, (h1-1, h2)),
        ])

    def desire(state):
        return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0

    return analyze(
        initial_state=initial(),
        is_terminal=is_term,
        get_transitions=transitions,
        compute_intrinsic_desire=desire,
        nest_level=10,
    )


def analyze_game(Game, **kwargs):
    """Analyze a game class through the ToA engine."""
    return analyze(
        initial_state=Game.initial_state() if 'initial_state' not in kwargs else kwargs.pop('initial_state'),
        is_terminal=Game.is_terminal,
        get_transitions=Game.get_transitions,
        compute_intrinsic_desire=Game.compute_intrinsic_desire,
        nest_level=kwargs.pop('nest_level', 10),
        **kwargs,
    )


def print_result(label, result):
    """Print a result row."""
    gds = result.game_design_score
    a1 = result.gds_components[0]
    a2p = gds - a1
    depth = a2p / gds * 100 if gds > 0 else 0
    print(f"  {label:<35} GDS={gds:.4f}  A1={a1:.4f}  A2+={a2p:.4f}  Depth={depth:.1f}%  States={len(result.states)}")


# =============================================================
# Experiments
# =============================================================

def experiment_1_player_count():
    """How does adding players change GDS?"""
    print("=" * 70)
    print("EXPERIMENT 1: Player Count vs GDS")
    print("=" * 70)
    print()

    # 2-player comparison at different HP levels
    print("--- 2-Player HP Battle ---")
    for hp in [2, 3, 4, 5]:
        r = analyze_2p(hp)
        print_result(f"2P HP={hp}", r)

    # 3-player FFA
    print("\n--- 3-Player FFA Battle ---")
    for hp in [2, 3]:
        r = analyze(
            initial_state=(hp, hp, hp),
            is_terminal=FFABattle.is_terminal,
            get_transitions=FFABattle.get_transitions,
            compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
            nest_level=10,
        )
        print_result(f"3P FFA HP={hp}", r)

    # 4-player FFA (only HP=2 — state space gets huge)
    print("\n--- 4-Player FFA Battle ---")
    r = analyze_game(FFABattle4)
    print_result("4P FFA HP=2", r)


def experiment_2_targeting():
    """Does targeting strategy matter?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Targeting Strategy (3P, HP=3)")
    print("=" * 70)
    print()

    # Random pairing (baseline)
    r1 = analyze(
        initial_state=(3, 3, 3),
        is_terminal=FFABattle.is_terminal,
        get_transitions=FFABattle.get_transitions,
        compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
        nest_level=10,
    )
    print_result("Random pairing", r1)

    # Focus fire (target the leader)
    r2 = analyze_game(FFAFocusFire)
    print_result("Focus fire (target leader)", r2)

    # Comparison
    delta = (r2.game_design_score - r1.game_design_score) / r1.game_design_score * 100
    print(f"\n  Focus fire effect: {delta:+.1f}% GDS change")


def experiment_3_elimination_phases():
    """Does the 3→2→1 elimination create interesting phase transitions?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Elimination Phase Analysis (3P FFA, HP=3)")
    print("=" * 70)
    print()

    r = analyze(
        initial_state=(3, 3, 3),
        is_terminal=FFABattle.is_terminal,
        get_transitions=FFABattle.get_transitions,
        compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
        nest_level=10,
    )

    # Analyze states by phase (3-alive vs 2-alive)
    three_alive_states = []
    two_alive_states = []
    for s in r.states:
        alive = sum(1 for h in s if h > 0)
        node = r.state_nodes[s]
        if alive == 3:
            three_alive_states.append((s, node))
        elif alive == 2:
            two_alive_states.append((s, node))

    if three_alive_states:
        avg_a1_3 = sum(n.a[0] for _, n in three_alive_states) / len(three_alive_states)
        avg_a2_3 = sum(n.a[1] for _, n in three_alive_states) / len(three_alive_states)
        print(f"  3-alive phase: {len(three_alive_states)} states, avg A₁={avg_a1_3:.4f}, avg A₂={avg_a2_3:.4f}")

    if two_alive_states:
        avg_a1_2 = sum(n.a[0] for _, n in two_alive_states) / len(two_alive_states)
        avg_a2_2 = sum(n.a[1] for _, n in two_alive_states) / len(two_alive_states)
        print(f"  2-alive phase: {len(two_alive_states)} states, avg A₁={avg_a1_2:.4f}, avg A₂={avg_a2_2:.4f}")

    # D_global at start
    start_node = r.state_nodes[(3, 3, 3)]
    print(f"\n  Starting D_global (win probability): {start_node.d_global:.4f}")
    print(f"  (Expected: 1/3 = {1/3:.4f} for fair 3-player)")


def experiment_4_scaling():
    """How does GDS scale with player count (fixed total HP)?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: GDS Scaling with Player Count")
    print("=" * 70)
    print()

    # Fixed total HP = 6, distributed among N players
    print("--- Fixed total HP=6, varying N ---")

    # 2 players, HP=3 each
    r2 = analyze_2p(3)
    print_result("N=2 (HP=3 each)", r2)

    # 3 players, HP=2 each
    r3 = analyze(
        initial_state=(2, 2, 2),
        is_terminal=FFABattle.is_terminal,
        get_transitions=FFABattle.get_transitions,
        compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
        nest_level=10,
    )
    print_result("N=3 (HP=2 each)", r3)

    # Also: fixed HP per player = 2, varying N
    print("\n--- Fixed HP=2 per player, varying N ---")
    print_result("N=2 (HP=2 each)", analyze_2p(2))
    print_result("N=3 (HP=2 each)", r3)
    print_result("N=4 (HP=2 each)", analyze_game(FFABattle4))


def experiment_5_win_prob_vs_gds():
    """Win probability drops with more players. How does this affect GDS?

    In 2P fair game: P(win) = 0.5
    In 3P fair game: P(win) ≈ 0.33
    In 4P fair game: P(win) ≈ 0.25

    Does the lower win probability reduce GDS, or does the extra
    uncertainty compensate?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Win Probability vs GDS")
    print("=" * 70)
    print()

    # Collect (N, win_prob, GDS) tuples
    data = []

    # 2P HP=3
    r = analyze_2p(3)
    data.append((2, r.state_nodes[(3, 3)].d_global, r.game_design_score))

    # 3P HP=3
    r = analyze(
        initial_state=(3, 3, 3),
        is_terminal=FFABattle.is_terminal,
        get_transitions=FFABattle.get_transitions,
        compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
        nest_level=10,
    )
    data.append((3, r.state_nodes[(3, 3, 3)].d_global, r.game_design_score))

    # 4P HP=2
    r = analyze_game(FFABattle4)
    data.append((4, r.state_nodes[(2, 2, 2, 2)].d_global, r.game_design_score))

    print(f"  {'N':>4} {'P(win)':>8} {'GDS':>8} {'GDS/P':>8}")
    print("  " + "-" * 35)
    for n, p, gds in data:
        print(f"  {n:>4} {p:8.4f} {gds:8.4f} {gds/p:8.4f}")

    print("\n  GDS/P(win) = engagement per unit of winning chance")
    print("  If this ratio increases with N, multiplayer is more \"efficient\" at generating engagement")


def key_findings():
    """Print summary."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS — MULTIPLAYER DYNAMICS")
    print("=" * 70)
    print("""
Analysis complete. Key findings will be updated after running experiments.

Theoretical predictions:
1. More players → lower win probability → lower absolute GDS
2. But elimination phases → higher depth ratio (more A₂+)
3. Focus fire → more balanced → might increase GDS
4. GDS/P(win) ratio reveals "engagement efficiency" of multiplayer
""")


if __name__ == "__main__":
    experiment_1_player_count()
    experiment_2_targeting()
    experiment_3_elimination_phases()
    experiment_4_scaling()
    experiment_5_win_prob_vs_gds()
    key_findings()
