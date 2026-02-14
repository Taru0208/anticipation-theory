"""CoinDuel_Rage — Coin duel with comeback mechanics.

Extends CoinDuel with a "desperation" system:
- When a player loses a round, they get bonus coins (desperation bonus)
- This creates comeback potential similar to the rage mechanic in HpGameRage

The losing player's bank grows faster, giving them bigger wagers → more
volatile outcomes → higher GDS from A₂+ anticipation components.

State: (score1, score2, bank1, bank2) tuple.
Terminal when score1 or score2 >= 3.
"""

from math import comb
from toa.game import sanitize_transitions


class CoinDuelRage:
    """Coin duel with desperation/comeback mechanics."""

    class Config:
        def __init__(
            self,
            rounds_to_win=3,
            initial_bank=4,
            max_bank=8,
            max_wager=3,
            refill_per_turn=1,
            desperation_bonus=2,  # extra coins when losing a round
        ):
            self.rounds_to_win = rounds_to_win
            self.initial_bank = initial_bank
            self.max_bank = max_bank
            self.max_wager = max_wager
            self.refill_per_turn = refill_per_turn
            self.desperation_bonus = desperation_bonus

    @staticmethod
    def initial_state():
        return (0, 0, 4, 4)

    @staticmethod
    def is_terminal(state):
        score1, score2, _, _ = state
        return score1 >= 3 or score2 >= 3

    @staticmethod
    def _flip_outcomes(n):
        outcomes = []
        for h in range(n + 1):
            prob = comb(n, h) / (2 ** n)
            outcomes.append((prob, h))
        return outcomes

    @staticmethod
    def _round_result(n1, n2):
        outcomes1 = CoinDuelRage._flip_outcomes(n1)
        outcomes2 = CoinDuelRage._flip_outcomes(n2)
        p_win1 = p_draw = p_win2 = 0.0
        for prob1, h1 in outcomes1:
            for prob2, h2 in outcomes2:
                joint = prob1 * prob2
                if h1 > h2:
                    p_win1 += joint
                elif h1 == h2:
                    p_draw += joint
                else:
                    p_win2 += joint
        return (p_win1, p_draw, p_win2)

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = CoinDuelRage.Config()

        score1, score2, bank1, bank2 = state
        if score1 >= config.rounds_to_win or score2 >= config.rounds_to_win:
            return []

        max_w1 = min(config.max_wager, bank1) if bank1 > 0 else 0
        max_w2 = min(config.max_wager, bank2) if bank2 > 0 else 0

        if max_w1 == 0 or max_w2 == 0:
            if max_w1 == 0 and max_w2 == 0:
                new_b1 = min(bank1 + config.refill_per_turn, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn, config.max_bank)
                return [(1.0, (score1, score2, new_b1, new_b2))]
            elif max_w1 == 0:
                # P1 has no coins → P2 wins round, P1 gets desperation bonus
                new_b1 = min(bank1 + config.refill_per_turn + config.desperation_bonus, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn, config.max_bank)
                return [(1.0, (score1, score2 + 1, new_b1, new_b2))]
            else:
                new_b1 = min(bank1 + config.refill_per_turn, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn + config.desperation_bonus, config.max_bank)
                return [(1.0, (score1 + 1, score2, new_b1, new_b2))]

        wager_pairs = []
        for w1 in range(1, max_w1 + 1):
            for w2 in range(1, max_w2 + 1):
                wager_pairs.append((w1, w2))

        pair_prob = 1.0 / len(wager_pairs)
        transitions = []

        for w1, w2 in wager_pairs:
            p_win1, p_draw, p_win2 = CoinDuelRage._round_result(w1, w2)

            decisive = p_win1 + p_win2
            if decisive > 0:
                p_win1_adj = p_win1 / decisive
                p_win2_adj = p_win2 / decisive
            else:
                p_win1_adj = 0.5
                p_win2_adj = 0.5

            base_b1 = bank1 - w1 + config.refill_per_turn
            base_b2 = bank2 - w2 + config.refill_per_turn

            # P1 wins: P2 gets desperation bonus
            new_b1_w1 = min(base_b1, config.max_bank)
            new_b2_w1 = min(base_b2 + config.desperation_bonus, config.max_bank)
            transitions.append((
                pair_prob * p_win1_adj,
                (score1 + 1, score2, new_b1_w1, new_b2_w1),
            ))

            # P2 wins: P1 gets desperation bonus
            new_b1_w2 = min(base_b1 + config.desperation_bonus, config.max_bank)
            new_b2_w2 = min(base_b2, config.max_bank)
            transitions.append((
                pair_prob * p_win2_adj,
                (score1, score2 + 1, new_b1_w2, new_b2_w2),
            ))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        score1, score2, _, _ = state
        return 1.0 if score1 >= 3 and score2 < 3 else 0.0

    @staticmethod
    def tostr(state):
        return f"S1:{state[0]} S2:{state[1]} B1:{state[2]} B2:{state[3]}"
