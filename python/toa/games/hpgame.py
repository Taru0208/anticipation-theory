"""HpGame — 1v1 HP battle, benchmark for multi-turn action games."""

from toa.game import sanitize_transitions


class HpGame:
    """Two players with HP. Each turn: win (P2 loses 1 HP), draw (both lose),
    or loss (P1 loses 1 HP), each with 1/3 probability.

    State: (hp1, hp2) tuple.
    Terminal when either HP reaches 0.
    Win condition: P1 alive and P2 dead.

    Expected GDS ≈ 0.430 (with 5 components).
    """

    @staticmethod
    def initial_state():
        return (5, 5)

    @staticmethod
    def is_terminal(state):
        hp1, hp2 = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def get_transitions(state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = [
            (1.0 / 3.0, (hp1, hp2 - 1)),      # P1 attacks, P2 misses
            (1.0 / 3.0, (hp1 - 1, hp2 - 1)),   # Both attack
            (1.0 / 3.0, (hp1 - 1, hp2)),        # P1 misses, P2 attacks
        ]
        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2 = state
        return 1.0 if hp1 > 0 and hp2 <= 0 else 0.0

    @staticmethod
    def tostr(state):
        return f"HP1:{state[0]} HP2:{state[1]}"
