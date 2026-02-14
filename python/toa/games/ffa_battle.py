"""FFA Battle — Free-for-all multiplayer HP battle.

N players, each with HP. Each turn a random pair fights (50/50 who attacks).
When a player reaches 0 HP, they're eliminated. Last player standing wins.

Key findings:
- GDS drops slightly with more players, but depth ratio surges (32% → 70%)
- "Engagement efficiency" (GDS/P(win)) increases with N: 1.02 → 1.29 → 1.64
- Elimination creates phase transitions: 3-alive phase has lower A₁ but higher A₂
- Focus fire mechanics halve GDS (-51.5%)

State: tuple of HP values, one per player.
"""

from toa.game import sanitize_transitions


class FFABattle:
    """Free-for-all HP battle for N players.

    Default: 3 players, HP=3 each.
    GDS ≈ 0.429 (HP=3, N=3), depth ratio 67.2%.
    """

    class Config:
        def __init__(self, num_players=3, hp=3):
            self.num_players = num_players
            self.hp = hp

    @staticmethod
    def initial_state(num_players=3, hp=3):
        return tuple([hp] * num_players)

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
        # Player 1 (index 0) wins if they're alive and everyone else is dead
        if state[0] > 0 and all(state[i] <= 0 for i in range(1, len(state))):
            return 1.0
        return 0.0

    @staticmethod
    def tostr(state):
        return " ".join(f"P{i+1}:{h}" for i, h in enumerate(state))
