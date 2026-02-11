"""Rock-Paper-Scissors game — benchmark for single-turn 3-outcome games."""

from toa.game import sanitize_transitions


class RPS:
    """Win/Draw/Loss with equal probability. A₁ ≈ 0.471."""

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
        transitions = [
            (1.0 / 3.0, "win"),
            (1.0 / 3.0, "draw"),
            (1.0 / 3.0, "loss"),
        ]
        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        return 1.0 if state == "win" else 0.0

    @staticmethod
    def tostr(state):
        return state.capitalize()
