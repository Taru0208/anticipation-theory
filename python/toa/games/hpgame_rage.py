"""HpGame_Rage — 1v1 HP battle with rage/critical mechanics.

Port of hpgame_rage.ixx. This game model adds a rage system to the
base HpGame, where:
- Rage accumulates when dealing or receiving damage
- Critical hits deal bonus damage scaled by accumulated rage
- Rage persists (does not spend) by default, only increasing over time

This model demonstrates ToA's power: the rage mechanics yield a +26.5%
improvement in Game Design Score over baseline HpGame (0.544 vs 0.430).

State: (hp1, hp2, rage1, rage2) tuple.
Terminal when either HP <= 0.
Win condition: P1 alive and P2 dead.
"""

from toa.game import sanitize_transitions


class HpGameRage:
    """HpGame with rage/critical hit system."""

    class Config:
        def __init__(
            self,
            critical_chance=0.10,
            rage_spendable=False,
            rage_dmg_multiplier=1,
            rage_increase_on_attack_dmg=True,
            rage_increase_on_received_dmg=True,
        ):
            self.critical_chance = critical_chance
            self.hit_chance = (1.0 - critical_chance) / 2.0
            self.miss_chance = (1.0 - critical_chance) / 2.0
            self.rage_spendable = rage_spendable
            self.rage_dmg_multiplier = rage_dmg_multiplier
            self.rage_increase_on_attack_dmg = rage_increase_on_attack_dmg
            self.rage_increase_on_received_dmg = rage_increase_on_received_dmg

    @staticmethod
    def initial_state():
        return (5, 5, 0, 0)

    @staticmethod
    def is_terminal(state):
        hp1, hp2, _, _ = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = HpGameRage.Config()

        hp1, hp2, rage1, rage2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        crit = config.critical_chance
        hit = config.hit_chance
        miss = config.miss_chance

        def next_rage(current_rage, dealt_damage, received_damage):
            r = current_rage
            if dealt_damage and config.rage_increase_on_attack_dmg:
                r += 1
            if received_damage and config.rage_increase_on_received_dmg:
                r += 1
            return r

        def crit_damage(attacker_rage):
            return 1 + attacker_rage * config.rage_dmg_multiplier

        def clamp_hp(hp):
            return max(0, hp)

        def rage_after_crit(current_rage, dealt_dmg, recv_dmg):
            if config.rage_spendable:
                return 0
            return next_rage(current_rage, dealt_dmg, recv_dmg)

        transitions = []

        # 1. P1 attacks, P2 misses
        transitions.append((
            hit * miss,
            (
                hp1,
                clamp_hp(hp2 - 1),
                next_rage(rage1, True, False),
                next_rage(rage2, False, True),
            )
        ))

        # 2. P1 crits, P2 misses
        transitions.append((
            crit * miss,
            (
                hp1,
                clamp_hp(hp2 - crit_damage(rage1)),
                rage_after_crit(rage1, True, False),
                next_rage(rage2, False, True),
            )
        ))

        # 3. P1 misses, P2 attacks
        transitions.append((
            miss * hit,
            (
                clamp_hp(hp1 - 1),
                hp2,
                next_rage(rage1, False, True),
                next_rage(rage2, True, False),
            )
        ))

        # 4. P1 misses, P2 crits
        transitions.append((
            miss * crit,
            (
                clamp_hp(hp1 - crit_damage(rage2)),
                hp2,
                next_rage(rage1, False, True),
                rage_after_crit(rage2, True, False),
            )
        ))

        # 5. Both attack
        transitions.append((
            hit * hit,
            (
                clamp_hp(hp1 - 1),
                clamp_hp(hp2 - 1),
                next_rage(rage1, True, True),
                next_rage(rage2, True, True),
            )
        ))

        # 6. P1 attacks, P2 crits
        transitions.append((
            hit * crit,
            (
                clamp_hp(hp1 - crit_damage(rage2)),
                clamp_hp(hp2 - 1),
                next_rage(rage1, True, True),
                rage_after_crit(rage2, True, False),
            )
        ))

        # 7. P1 crits, P2 attacks
        transitions.append((
            crit * hit,
            (
                clamp_hp(hp1 - 1),
                clamp_hp(hp2 - crit_damage(rage1)),
                rage_after_crit(rage1, True, False),
                next_rage(rage2, True, True),
            )
        ))

        # 8. Both crit
        transitions.append((
            crit * crit,
            (
                clamp_hp(hp1 - crit_damage(rage2)),
                clamp_hp(hp2 - crit_damage(rage1)),
                rage_after_crit(rage1, True, False),
                rage_after_crit(rage2, True, False),
            )
        ))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2, _, _ = state
        return 1.0 if hp1 > 0 and hp2 <= 0 else 0.0

    @staticmethod
    def tostr(state):
        return f"HP1:{state[0]} HP2:{state[1]} R1:{state[2]} R2:{state[3]}"
