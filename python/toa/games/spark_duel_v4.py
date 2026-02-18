"""SparkDuel v4 — Symmetric sequential combat with Dodge counter-attack.

Builds on v3's symmetric structure (both attack+defend each round).
Key change: successful Dodge deals counter-damage to the attacker.

This addresses v3's CPG problem (0.206): Brace was safe+winning, Dodge was
exciting+losing. By adding counter-damage, Dodge becomes risky-but-rewarding
for the defender, aligning fun play with winning play.

State: (hp1, hp2, cd1, cd2, phase)
  phase 0: P1 attacks (start of round, after chip damage)
  phase 1: P2 attacks (mid-round)
"""

from toa.game import sanitize_transitions


class SparkDuelV4:
    """Symmetric sequential combat with Dodge counter-attack."""

    class Config:
        def __init__(
            self,
            max_hp=7,
            chip_damage=1,
            blast_damage=4,
            blast_hit_rate=0.60,
            zap_damage=2,
            brace_reduction=1,
            dodge_chance=0.40,
            dodge_counter=1,  # damage dealt to attacker on successful dodge
        ):
            self.max_hp = max_hp
            self.chip_damage = chip_damage
            self.blast_damage = blast_damage
            self.blast_hit_rate = blast_hit_rate
            self.zap_damage = zap_damage
            self.brace_reduction = brace_reduction
            self.dodge_chance = dodge_chance
            self.dodge_counter = dodge_counter

    BLAST = 0
    ZAP = 1
    BRACE = 2
    DODGE = 3

    @staticmethod
    def initial_state(config=None):
        if config is None:
            config = SparkDuelV4.Config()
        return (config.max_hp, config.max_hp, 0, 0, 0)

    @staticmethod
    def is_terminal(state):
        hp1, hp2, _, _, _ = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def _available_attacks(cooldown):
        if cooldown > 0:
            return [SparkDuelV4.ZAP]
        return [SparkDuelV4.BLAST, SparkDuelV4.ZAP]

    @staticmethod
    def _resolve_attack(attack, defend, config):
        """Resolve one attack-defend exchange.
        Returns [(probability, damage_to_defender, damage_to_attacker)]
        Does NOT include chip damage.
        """
        if attack == SparkDuelV4.BLAST:
            hit_dmg = config.blast_damage

            if defend == SparkDuelV4.BRACE:
                reduced = max(0, hit_dmg - config.brace_reduction)
                return [
                    (config.blast_hit_rate, reduced, 0),         # hit, braced
                    (1.0 - config.blast_hit_rate, 0, 0),         # miss
                ]
            else:  # DODGE
                return [
                    # hit + dodge success: no damage to defender, counter to attacker
                    (config.blast_hit_rate * config.dodge_chance, 0, config.dodge_counter),
                    # hit + dodge fail: full damage, no counter
                    (config.blast_hit_rate * (1 - config.dodge_chance), hit_dmg, 0),
                    # miss: no damage either way (miss = Blast didn't fire, nothing to dodge)
                    (1.0 - config.blast_hit_rate, 0, 0),
                ]

        else:  # ZAP
            zap_dmg = config.zap_damage

            if defend == SparkDuelV4.BRACE:
                reduced = max(0, zap_dmg - config.brace_reduction)
                return [(1.0, reduced, 0)]
            else:  # DODGE
                return [
                    # dodge success: no damage + counter
                    (config.dodge_chance, 0, config.dodge_counter),
                    # dodge fail: full damage
                    (1.0 - config.dodge_chance, zap_dmg, 0),
                ]

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = SparkDuelV4.Config()

        hp1, hp2, cd1, cd2, phase = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        if phase == 0:
            # Start of round: apply chip damage to both first
            hp1_chipped = max(0, hp1 - config.chip_damage)
            hp2_chipped = max(0, hp2 - config.chip_damage)

            # Check if chip damage killed anyone
            if hp1_chipped <= 0 or hp2_chipped <= 0:
                return [(1.0, (hp1_chipped, hp2_chipped, cd1, cd2, 0))]

            # P1 attacks, P2 defends
            attacks = SparkDuelV4._available_attacks(cd1)
            defends = [SparkDuelV4.BRACE, SparkDuelV4.DODGE]

            pairs = [(a, d) for a in attacks for d in defends]
            pair_prob = 1.0 / len(pairs)

            transitions = []
            for attack, defend in pairs:
                new_cd1 = 1 if attack == SparkDuelV4.BLAST else max(0, cd1 - 1)
                outcomes = SparkDuelV4._resolve_attack(attack, defend, config)
                for prob, dmg_to_def, dmg_to_atk in outcomes:
                    new_hp2 = max(0, hp2_chipped - dmg_to_def)
                    new_hp1 = max(0, hp1_chipped - dmg_to_atk)
                    next_state = (new_hp1, new_hp2, new_cd1, cd2, 1)
                    transitions.append((pair_prob * prob, next_state))

            return sanitize_transitions(transitions)

        else:  # phase == 1
            # P2 attacks, P1 defends. No chip here (already applied at phase 0).
            attacks = SparkDuelV4._available_attacks(cd2)
            defends = [SparkDuelV4.BRACE, SparkDuelV4.DODGE]

            pairs = [(a, d) for a in attacks for d in defends]
            pair_prob = 1.0 / len(pairs)

            transitions = []
            for attack, defend in pairs:
                new_cd2 = 1 if attack == SparkDuelV4.BLAST else max(0, cd2 - 1)
                outcomes = SparkDuelV4._resolve_attack(attack, defend, config)
                for prob, dmg_to_def, dmg_to_atk in outcomes:
                    new_hp1 = max(0, hp1 - dmg_to_def)  # P1 is defender
                    new_hp2 = max(0, hp2 - dmg_to_atk)   # P2 is attacker, takes counter
                    # Back to phase 0 (next round)
                    next_state = (new_hp1, new_hp2, cd1, new_cd2, 0)
                    transitions.append((pair_prob * prob, next_state))

            return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2, _, _, _ = state
        if hp1 <= 0 and hp2 <= 0:
            return 0.5
        if hp2 <= 0:
            return 1.0
        if hp1 <= 0:
            return 0.0
        return 0.0

    @staticmethod
    def tostr(state):
        phase = "P1atk" if state[4] == 0 else "P2atk"
        return f"HP:{state[0]}v{state[1]} CD:{state[2]}v{state[3]} {phase}"
