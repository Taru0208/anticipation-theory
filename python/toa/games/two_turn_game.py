"""TwoTurnGame — minimal multi-turn game for parameter optimization.

A simple tree: turn 0 splits into two branches with configurable
probabilities, each branch then splits into win/loss.

Used in the paper for demonstrating GDS optimization via random search.

State: (turn, node_idx_in_turn).
"""

from toa.game import sanitize_transitions


class TwoTurnGame:
    """Port of two_turn_game.ixx."""

    class Config:
        def __init__(self, probability1=0.5, probability2=0.5, probability3=0.5):
            self.probability1 = probability1
            self.probability2 = probability2
            self.probability3 = probability3

    @staticmethod
    def initial_state():
        return (0, 0)

    @staticmethod
    def is_terminal(state):
        return state[0] >= 2

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = TwoTurnGame.Config()

        turn, node = state

        if turn == 0 and node == 0:
            transitions = [
                (config.probability1, (1, 0)),
                (1.0 - config.probability1, (1, 1)),
            ]
        elif turn == 1:
            if node == 0:
                transitions = [
                    (config.probability2, (2, 0)),
                    (1.0 - config.probability2, (2, 1)),
                ]
            elif node == 1:
                transitions = [
                    (config.probability3, (2, 0)),
                    (1.0 - config.probability3, (2, 1)),
                ]
            else:
                return []
        else:
            return []

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        return 1.0 if state[0] == 2 and state[1] == 0 else 0.0

    @staticmethod
    def tostr(state):
        return f"Turn:{state[0]} Node:{state[1]}"
