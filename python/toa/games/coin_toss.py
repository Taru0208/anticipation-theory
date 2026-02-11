"""Coin toss game — theoretical optimum for single-turn binary games."""


class CoinToss:
    """50/50 win/loss. A₁ should equal exactly 0.5."""

    @staticmethod
    def initial_state():
        return "initial"

    @staticmethod
    def is_terminal(state):
        return state != "initial"

    @staticmethod
    def get_transitions(state, config=None):
        if state != "initial":
            return []
        return [(0.5, "win"), (0.5, "loss")]

    @staticmethod
    def compute_intrinsic_desire(state):
        return 1.0 if state == "win" else 0.0

    @staticmethod
    def tostr(state):
        return state.capitalize()
