"""GoldGame — economic competition with geometric rewards."""


class GoldGame:
    """Two players accumulate gold over turns. Success multiplies by 1.2x,
    failure divides by 1.2x. Winner determined by final gold comparison.

    State: (player1_gold, player2_gold, turn) tuple.
    Terminal when turn >= max_turns.

    This is a port of goldgame_clean.ixx — the simplest gold game variant.
    """

    class Config:
        def __init__(
            self,
            max_turns=10,
            success_chance=0.68,
            geometric_multiplier=1.2,
            geometric_penalty=1.0 / 1.2,
        ):
            self.max_turns = max_turns
            self.success_chance = success_chance
            self.geometric_multiplier = geometric_multiplier
            self.geometric_penalty = geometric_penalty

    @staticmethod
    def initial_state():
        return (1000, 1000, 0)

    @staticmethod
    def is_terminal(state):
        _, _, turn = state
        # Must check against config, but we use default here
        # The caller should use a closure or partial for config-dependent terminal check
        return turn >= 10

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = GoldGame.Config()

        p1_gold, p2_gold, turn = state
        if turn >= config.max_turns:
            return []

        hit = config.success_chance
        miss = 1.0 - hit

        transitions = []
        for p1_result in range(2):  # 0=miss, 1=hit
            for p2_result in range(2):
                new_p1 = int(p1_gold * (config.geometric_multiplier if p1_result else config.geometric_penalty))
                new_p2 = int(p2_gold * (config.geometric_multiplier if p2_result else config.geometric_penalty))

                prob = (hit if p1_result else miss) * (hit if p2_result else miss)
                transitions.append((prob, (new_p1, new_p2, turn + 1)))

        return transitions

    @staticmethod
    def compute_intrinsic_desire(state):
        p1_gold, p2_gold, turn = state
        if turn < 10:  # not terminal
            return 0.0
        return 1.0 if p1_gold > p2_gold else 0.0

    @staticmethod
    def tostr(state):
        return f"T{state[2]} P1:{state[0]} P2:{state[1]}"
