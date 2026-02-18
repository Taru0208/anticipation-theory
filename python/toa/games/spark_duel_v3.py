"""SparkDuel v3 — Symmetric sequential combat.

Each ROUND:
  1. P1 chooses attack (Blast or Zap), P2 sees it and chooses defense (Brace or Dodge)
  2. P2 chooses attack (Blast or Zap), P1 sees it and chooses defense (Brace or Dodge)
  3. Both take chip damage

This is structurally symmetric — both players attack and defend each round.
The sequential element (defender sees attacker's choice) creates strategic depth.

State: (hp1, hp2, cd1, cd2, phase)
  phase 0: P1 attacks (start of round)
  phase 1: P2 attacks (mid-round, after P1's attack resolved)

Chip damage is applied once per round (at end of phase 1).
"""

from toa.game import sanitize_transitions


class SparkDuelV3:
    """Symmetric sequential combat — both attack and defend each round."""

    class Config:
        def __init__(
            self,
            max_hp=7,
            chip_damage=1,
            blast_damage=3,
            blast_hit_rate=0.60,
            zap_damage=1,
            brace_reduction=1,
            dodge_chance=0.40,
        ):
            self.max_hp = max_hp
            self.chip_damage = chip_damage
            self.blast_damage = blast_damage
            self.blast_hit_rate = blast_hit_rate
            self.zap_damage = zap_damage
            self.brace_reduction = brace_reduction
            self.dodge_chance = dodge_chance

    BLAST = 0
    ZAP = 1
    BRACE = 2
    DODGE = 3

    @staticmethod
    def initial_state(config=None):
        if config is None:
            config = SparkDuelV3.Config()
        return (config.max_hp, config.max_hp, 0, 0, 0)

    @staticmethod
    def is_terminal(state):
        hp1, hp2, _, _, _ = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def _available_attacks(cooldown):
        if cooldown > 0:
            return [SparkDuelV3.ZAP]
        return [SparkDuelV3.BLAST, SparkDuelV3.ZAP]

    @staticmethod
    def _resolve_attack(attack, defend, config):
        """Resolve one attack-defend exchange.
        Returns [(probability, extra_damage_to_defender)]
        Does NOT include chip damage.
        """
        if attack == SparkDuelV3.BLAST:
            hit_dmg = config.blast_damage

            if defend == SparkDuelV3.BRACE:
                reduced = max(0, hit_dmg - config.brace_reduction)
                return [
                    (config.blast_hit_rate, reduced),
                    (1.0 - config.blast_hit_rate, 0),
                ]
            else:  # DODGE
                return [
                    (config.blast_hit_rate * config.dodge_chance, 0),           # hit but dodged
                    (config.blast_hit_rate * (1 - config.dodge_chance), hit_dmg),  # hit, no dodge
                    (1.0 - config.blast_hit_rate, 0),                           # miss
                ]

        else:  # ZAP
            zap_dmg = config.zap_damage

            if defend == SparkDuelV3.BRACE:
                reduced = max(0, zap_dmg - config.brace_reduction)
                return [(1.0, reduced)]
            else:  # DODGE
                return [
                    (config.dodge_chance, 0),
                    (1.0 - config.dodge_chance, zap_dmg),
                ]

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = SparkDuelV3.Config()

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
            attacks = SparkDuelV3._available_attacks(cd1)
            defends = [SparkDuelV3.BRACE, SparkDuelV3.DODGE]

            pairs = [(a, d) for a in attacks for d in defends]
            pair_prob = 1.0 / len(pairs)

            transitions = []
            for attack, defend in pairs:
                new_cd1 = 1 if attack == SparkDuelV3.BLAST else max(0, cd1 - 1)
                outcomes = SparkDuelV3._resolve_attack(attack, defend, config)
                for prob, dmg_to_p2 in outcomes:
                    new_hp2 = max(0, hp2_chipped - dmg_to_p2)
                    next_state = (hp1_chipped, new_hp2, new_cd1, cd2, 1)
                    transitions.append((pair_prob * prob, next_state))

            return sanitize_transitions(transitions)

        else:  # phase == 1
            # P2 attacks, P1 defends. No chip here (already applied at phase 0).
            attacks = SparkDuelV3._available_attacks(cd2)
            defends = [SparkDuelV3.BRACE, SparkDuelV3.DODGE]

            pairs = [(a, d) for a in attacks for d in defends]
            pair_prob = 1.0 / len(pairs)

            transitions = []
            for attack, defend in pairs:
                new_cd2 = 1 if attack == SparkDuelV3.BLAST else max(0, cd2 - 1)
                outcomes = SparkDuelV3._resolve_attack(attack, defend, config)
                for prob, dmg_to_p1 in outcomes:
                    new_hp1 = max(0, hp1 - dmg_to_p1)
                    # Back to phase 0 (next round)
                    next_state = (new_hp1, hp2, cd1, new_cd2, 0)
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
