"""GoldGameCritical — gold competition with critical hits and stealing.

Extension of GoldGame with three possible outcomes per player per turn:
miss, normal hit, critical hit (with gold stealing from opponent).

The terminal condition uses a cubed-ratio win probability, making gold
advantages matter more (cubic function amplifies small leads).

State: (p1_gold, p2_gold, turn, has_won).
"""


class GoldGameCritical:
    """Port of goldgame_critical.ixx."""

    MAX_TURNS = 5

    class Config:
        def __init__(
            self,
            reward_type="linear",  # "linear" or "geometric"
            success_chance=0.5,
            base_reward=100,
            reward_growth=0.0,
            critical_chance=0.0,
            steal_percentage=0.0,
            geometric_multiplier=1.2,
            geometric_penalty=1.0 / 1.2,
        ):
            self.reward_type = reward_type
            self.success_chance = success_chance
            self.base_reward = base_reward
            self.reward_growth = reward_growth
            self.critical_chance = critical_chance
            self.steal_percentage = steal_percentage
            self.geometric_multiplier = geometric_multiplier
            self.geometric_penalty = geometric_penalty

    @staticmethod
    def initial_state():
        return (1000, 1000, 1, False)

    @staticmethod
    def is_terminal(state):
        return state[2] > GoldGameCritical.MAX_TURNS

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = GoldGameCritical.Config()

        p1_gold, p2_gold, turn, has_won = state
        if turn > GoldGameCritical.MAX_TURNS:
            return []

        # Last turn: resolve winner based on gold ratio
        if turn == GoldGameCritical.MAX_TURNS:
            total = p1_gold + p2_gold
            if total == 0:
                return [(1.0, (p1_gold, p2_gold, turn + 1, False))]

            p1_ratio = p1_gold / total
            p1_cubed = p1_ratio ** 3
            p2_cubed = (1.0 - p1_ratio) ** 3
            p1_win = p1_cubed / (p1_cubed + p2_cubed)

            return [
                (p1_win, (p1_gold, p2_gold, turn + 1, True)),
                (1.0 - p1_win, (p1_gold, p2_gold, turn + 1, False)),
            ]

        # Regular gameplay
        miss = 1.0 - config.success_chance
        normal_hit = config.success_chance * (1.0 - config.critical_chance)
        critical_hit = config.success_chance * config.critical_chance

        probs = [miss, normal_hit, critical_hit]
        base_reward = config.base_reward + int(config.reward_growth * turn * config.base_reward)

        transitions = []
        for p1_result in range(3):  # 0=miss, 1=hit, 2=crit
            for p2_result in range(3):
                new_p1, new_p2 = p1_gold, p2_gold

                if config.reward_type == "linear":
                    if p1_result == 1:
                        new_p1 += base_reward
                    elif p1_result == 2:
                        new_p1 += base_reward
                        stolen = int(new_p2 * config.steal_percentage)
                        new_p1 += stolen
                        new_p2 -= stolen

                    if p2_result == 1:
                        new_p2 += base_reward
                    elif p2_result == 2:
                        new_p2 += base_reward
                        stolen = int(new_p1 * config.steal_percentage)
                        new_p2 += stolen
                        new_p1 -= stolen
                else:  # geometric
                    if p1_result == 0:
                        new_p1 = int(new_p1 * config.geometric_penalty)
                    elif p1_result == 1:
                        bonus = int(new_p1 * (config.geometric_multiplier - 1.0))
                        new_p1 += base_reward + bonus
                    elif p1_result == 2:
                        bonus = int(new_p1 * (config.geometric_multiplier - 1.0))
                        new_p1 += base_reward + bonus
                        stolen = int(new_p2 * config.steal_percentage)
                        new_p1 += stolen
                        new_p2 -= stolen

                    if p2_result == 0:
                        new_p2 = int(new_p2 * config.geometric_penalty)
                    elif p2_result == 1:
                        bonus = int(new_p2 * (config.geometric_multiplier - 1.0))
                        new_p2 += base_reward + bonus
                    elif p2_result == 2:
                        bonus = int(new_p2 * (config.geometric_multiplier - 1.0))
                        new_p2 += base_reward + bonus
                        stolen = int(new_p1 * config.steal_percentage)
                        new_p2 += stolen
                        new_p1 -= stolen

                prob = probs[p1_result] * probs[p2_result]
                if prob > 1e-12:
                    transitions.append((prob, (new_p1, new_p2, turn + 1, False)))

        return transitions

    @staticmethod
    def compute_intrinsic_desire(state):
        if not GoldGameCritical.is_terminal(state):
            return 0.0
        return 1.0 if state[3] else 0.0

    @staticmethod
    def tostr(state):
        p1, p2, turn, won = state
        result = f"T{turn} P1:{p1} P2:{p2}"
        if GoldGameCritical.is_terminal(state):
            result += " [P1 WINS]" if won else " [P2 WINS]"
        return result
