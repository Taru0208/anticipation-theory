"""CoinDuel — Turn-based coin-flipping duel with resource management.

Each turn, both players simultaneously choose how many coins to wager (1 to max_wager).
All coins are flipped; whoever gets more heads wins the round.
First to reach `rounds_to_win` round wins takes the game.

The interesting tension: more coins = better odds of winning the round,
but the bank is limited. Banking coins early means bigger wagers later.

State: (score1, score2, bank1, bank2) tuple.
- score: rounds won so far (0 to rounds_to_win)
- bank: coins available (0 to max_bank)

For ToA analysis, we average over all possible wager combinations uniformly,
representing the "typical play experience" across strategies.

Designed as a game concept for Unity implementation with ToA optimization.
"""

from math import comb
from toa.game import sanitize_transitions


class CoinDuel:
    """Two-player coin duel with resource management."""

    class Config:
        def __init__(
            self,
            rounds_to_win=3,
            initial_bank=5,
            max_bank=8,
            max_wager=3,
            refill_per_turn=1,
        ):
            self.rounds_to_win = rounds_to_win
            self.initial_bank = initial_bank
            self.max_bank = max_bank
            self.max_wager = max_wager
            self.refill_per_turn = refill_per_turn

    @staticmethod
    def initial_state():
        return (0, 0, 5, 5)  # (score1, score2, bank1, bank2)

    @staticmethod
    def is_terminal(state):
        score1, score2, _, _ = state
        return score1 >= 3 or score2 >= 3

    @staticmethod
    def _flip_outcomes(n):
        """Return [(probability, heads_count)] for flipping n coins."""
        outcomes = []
        for h in range(n + 1):
            prob = comb(n, h) / (2 ** n)
            outcomes.append((prob, h))
        return outcomes

    @staticmethod
    def _round_result(n1, n2):
        """Given wager sizes n1 and n2, return (p_win1, p_draw, p_win2).

        Player wins the round if they get strictly more heads.
        """
        outcomes1 = CoinDuel._flip_outcomes(n1)
        outcomes2 = CoinDuel._flip_outcomes(n2)

        p_win1 = 0.0
        p_draw = 0.0
        p_win2 = 0.0

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
            config = CoinDuel.Config()

        score1, score2, bank1, bank2 = state
        if score1 >= config.rounds_to_win or score2 >= config.rounds_to_win:
            return []

        # Available wager choices for each player (1 to min(max_wager, bank))
        max_w1 = min(config.max_wager, bank1) if bank1 > 0 else 0
        max_w2 = min(config.max_wager, bank2) if bank2 > 0 else 0

        if max_w1 == 0 or max_w2 == 0:
            # Edge case: a player has no coins. Force wager of 0 = auto-lose.
            # Give the round to the other player.
            if max_w1 == 0 and max_w2 == 0:
                # Both empty — draw, just refill
                new_b1 = min(bank1 + config.refill_per_turn, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn, config.max_bank)
                return [(1.0, (score1, score2, new_b1, new_b2))]
            elif max_w1 == 0:
                new_b1 = min(bank1 + config.refill_per_turn, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn, config.max_bank)
                return [(1.0, (score1, score2 + 1, new_b1, new_b2))]
            else:
                new_b1 = min(bank1 + config.refill_per_turn, config.max_bank)
                new_b2 = min(bank2 + config.refill_per_turn, config.max_bank)
                return [(1.0, (score1 + 1, score2, new_b1, new_b2))]

        # Enumerate all wager combinations uniformly
        wager_pairs = []
        for w1 in range(1, max_w1 + 1):
            for w2 in range(1, max_w2 + 1):
                wager_pairs.append((w1, w2))

        pair_prob = 1.0 / len(wager_pairs)
        transitions = []

        for w1, w2 in wager_pairs:
            p_win1, p_draw, p_win2 = CoinDuel._round_result(w1, w2)

            # Resolve draws by redistributing proportionally (= re-roll until decisive).
            # This eliminates self-loops so the state graph is a DAG.
            decisive = p_win1 + p_win2
            if decisive > 0:
                p_win1_adj = p_win1 / decisive
                p_win2_adj = p_win2 / decisive
            else:
                # Both wager 0 somehow — shouldn't happen with bank > 0
                p_win1_adj = 0.5
                p_win2_adj = 0.5

            # After wager, coins are spent and bank refills
            new_b1 = min(bank1 - w1 + config.refill_per_turn, config.max_bank)
            new_b2 = min(bank2 - w2 + config.refill_per_turn, config.max_bank)

            transitions.append((
                pair_prob * p_win1_adj,
                (score1 + 1, score2, new_b1, new_b2),
            ))
            transitions.append((
                pair_prob * p_win2_adj,
                (score1, score2 + 1, new_b1, new_b2),
            ))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        score1, score2, _, _ = state
        return 1.0 if score1 >= 3 and score2 < 3 else 0.0

    @staticmethod
    def tostr(state):
        return f"S1:{state[0]} S2:{state[1]} B1:{state[2]} B2:{state[3]}"
