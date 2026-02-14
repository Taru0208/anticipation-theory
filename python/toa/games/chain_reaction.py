"""ChainReaction — Abstract model of territory-control with cascade mechanics.

Instead of a full spatial grid (too many states), this models the *essence*
of chain reaction games: placing a piece can trigger cascading flips.

Abstract model:
- State: (territory1, territory2, turn_number) where territories sum to TOTAL
- Each turn, the current player "places" and a cascade may or may not trigger
- Cascade probability increases with territory size (more pieces = more chain potential)
- A cascade flips 1-3 enemy pieces to friendly

This captures the key ToA properties:
- Multi-turn depth (Unbound potential)
- Comeback mechanics via cascades (large territory flips)
- Agency: placement quality affects cascade probability

State: (t1, t2, turn, whose_turn) where:
  t1 + t2 + empty = BOARD_SIZE
  whose_turn: 1 or 2

Terminal: all cells filled (turn >= max_turns) or one player has 0 territory
"""

from toa.game import sanitize_transitions


class ChainReaction:
    """Abstract chain reaction territory game."""

    class Config:
        def __init__(
            self,
            board_size=9,       # total cells
            max_turns=9,        # game length
            cascade_base=0.15,  # base cascade probability
            cascade_scale=0.08, # extra cascade prob per owned territory
            cascade_flip_1=0.6, # P(flip 1 piece | cascade)
            cascade_flip_2=0.3, # P(flip 2 pieces | cascade)
            cascade_flip_3=0.1, # P(flip 3 pieces | cascade)
        ):
            self.board_size = board_size
            self.max_turns = max_turns
            self.cascade_base = cascade_base
            self.cascade_scale = cascade_scale
            self.cascade_flip_1 = cascade_flip_1
            self.cascade_flip_2 = cascade_flip_2
            self.cascade_flip_3 = cascade_flip_3

    @staticmethod
    def initial_state():
        # (t1, t2, turn) — P1 starts
        return (0, 0, 0)

    @staticmethod
    def is_terminal(state):
        t1, t2, turn = state
        return turn >= 9

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = ChainReaction.Config()

        t1, t2, turn = state
        if turn >= config.max_turns:
            return []

        empty = config.board_size - t1 - t2
        if empty <= 0:
            return []

        is_p1_turn = (turn % 2 == 0)
        my_t = t1 if is_p1_turn else t2
        opp_t = t2 if is_p1_turn else t1

        transitions = []

        # Basic placement: always claims 1 empty cell
        # Cascade probability depends on how much territory you control
        cascade_prob = min(0.9, config.cascade_base + my_t * config.cascade_scale)
        no_cascade_prob = 1.0 - cascade_prob

        # No cascade: just place one piece
        new_t1_nc = t1 + (1 if is_p1_turn else 0)
        new_t2_nc = t2 + (0 if is_p1_turn else 1)
        transitions.append((no_cascade_prob, (new_t1_nc, new_t2_nc, turn + 1)))

        # Cascade: place one piece AND flip some opponent pieces
        if opp_t > 0:
            for flip_n, flip_p in [(1, config.cascade_flip_1),
                                    (2, config.cascade_flip_2),
                                    (3, config.cascade_flip_3)]:
                actual_flip = min(flip_n, opp_t)
                if is_p1_turn:
                    nt1 = t1 + 1 + actual_flip
                    nt2 = t2 - actual_flip
                else:
                    nt1 = t1 - actual_flip
                    nt2 = t2 + 1 + actual_flip
                # Ensure non-negative
                nt1 = max(0, nt1)
                nt2 = max(0, nt2)
                transitions.append((cascade_prob * flip_p, (nt1, nt2, turn + 1)))
        else:
            # No opponent pieces to flip — cascade still just places
            transitions.append((cascade_prob, (new_t1_nc, new_t2_nc, turn + 1)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        t1, t2, turn = state
        if turn < 9:
            return 0.0
        # P1 wins if more territory
        if t1 > t2:
            return 1.0
        return 0.0  # draw or loss

    @staticmethod
    def tostr(state):
        return f"P1:{state[0]} P2:{state[1]} T:{state[2]}"
