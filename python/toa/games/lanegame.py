"""LaneGame — MOBA-style laning phase model.

Two players compete for minion last-hits over a fixed number of minion waves.
Similar to HpGame but with experience accumulation instead of HP depletion.

State: (exp1, exp2, minions_remaining).
Terminal when minions_remaining <= 0.
Win: player with more experience wins.
"""

from toa.game import sanitize_transitions


class LaneGame:
    """Port of lanegame.ixx — 1v1 lane competition."""

    @staticmethod
    def initial_state():
        return (0, 0, 10)

    @staticmethod
    def is_terminal(state):
        return state[2] <= 0

    @staticmethod
    def get_transitions(state, config=None):
        exp1, exp2, minions = state
        if minions <= 0:
            return []

        next_minions = minions - 2

        # Win: P1 gets minion, P2 doesn't
        win = (exp1 + 1, exp2, next_minions)
        # Draw: both get a minion
        draw = (exp1 + 1, exp2 + 1, next_minions)
        # Loss: P2 gets minion, P1 doesn't
        loss = (exp1, exp2 + 1, next_minions)

        transitions = [
            (1.0 / 3.0, win),
            (1.0 / 3.0, draw),
            (1.0 / 3.0, loss),
        ]
        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        exp1, exp2, minions = state
        if minions > 0:
            return 0.0
        return 1.0 if exp1 > exp2 else 0.0

    @staticmethod
    def tostr(state):
        return f"Exp1:{state[0]} Exp2:{state[1]} Minions:{state[2]}"
