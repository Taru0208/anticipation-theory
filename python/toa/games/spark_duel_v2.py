"""SparkDuel v2 — Sequential-action combat with visible choices.

Key change from v1: Turn-based instead of simultaneous.
Players alternate turns. The acting player chooses, then the defender responds.
This creates information visibility + sequentiality = deeper strategy (DraftWars principle).

Mechanics:
  - Each turn: attacker chooses action, defender sees it and responds.
  - Attacker actions: Blast (60% chance, 3 dmg, 1-turn cooldown) or Zap (100%, 1 dmg)
  - Defender responses: Brace (reduce incoming by 1) or Dodge (40% chance to avoid ALL damage)
  - Both players take 1 chip damage per turn (ensures progress).
  - After resolution, roles swap.

State: (hp1, hp2, cd1, cd2, turn) where turn=0 means P1 attacks, turn=1 means P2 attacks.

Design rationale:
  - Sequential choice → defender adapts to attacker → strategic depth
  - Dodge (40% avoid all) vs Brace (guaranteed -1) is a risk/reward decision for defender too
  - Both roles have meaningful choices → high PI expected
  - Blast (high damage, probabilistic) > Zap (low damage, certain) → CPG ≈ 0 for attacker
  - Dodge vs Brace gives defender agency without making defense dominant
"""

from toa.game import sanitize_transitions


class SparkDuelV2:
    """Turn-based combat with attacker/defender roles."""

    class Config:
        def __init__(
            self,
            max_hp=5,
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

    # Attacker actions
    BLAST = 0
    ZAP = 1
    # Defender actions
    BRACE = 2
    DODGE = 3

    ATTACK_NAMES = ["Blast", "Zap"]
    DEFEND_NAMES = ["Brace", "Dodge"]

    @staticmethod
    def initial_state(config=None):
        if config is None:
            config = SparkDuelV2.Config()
        return (config.max_hp, config.max_hp, 0, 0, 0)  # P1 attacks first

    @staticmethod
    def is_terminal(state):
        hp1, hp2, _, _, _ = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def _available_attacks(cooldown):
        if cooldown > 0:
            return [SparkDuelV2.ZAP]
        return [SparkDuelV2.BLAST, SparkDuelV2.ZAP]

    @staticmethod
    def _resolve_turn(hp_atk, hp_def, cd_atk, attack, defend, config):
        """Resolve one attack-defend exchange.
        Returns [(probability, (new_hp_atk, new_hp_def, new_cd_atk))]

        Both players take chip damage. Attacker's extra damage is modified by defense.
        """
        new_cd_atk = 1 if attack == SparkDuelV2.BLAST else max(0, cd_atk - 1)

        # Chip damage to both (unavoidable)
        chip_atk = config.chip_damage
        chip_def = config.chip_damage

        results = []

        if attack == SparkDuelV2.BLAST:
            hit_dmg = config.blast_damage
            hit_prob = config.blast_hit_rate
            miss_prob = 1.0 - hit_prob

            if defend == SparkDuelV2.BRACE:
                # Hit: damage reduced by brace
                reduced = max(0, hit_dmg - config.brace_reduction)
                results.append((hit_prob, chip_atk, chip_def + reduced))
                # Miss: no extra damage
                results.append((miss_prob, chip_atk, chip_def))

            elif defend == SparkDuelV2.DODGE:
                # Hit + dodge success: no extra damage
                results.append((hit_prob * config.dodge_chance, chip_atk, chip_def))
                # Hit + dodge fail: full damage
                results.append((hit_prob * (1.0 - config.dodge_chance), chip_atk, chip_def + hit_dmg))
                # Miss: no extra damage regardless
                results.append((miss_prob, chip_atk, chip_def))

        elif attack == SparkDuelV2.ZAP:
            zap_dmg = config.zap_damage

            if defend == SparkDuelV2.BRACE:
                reduced = max(0, zap_dmg - config.brace_reduction)
                results.append((1.0, chip_atk, chip_def + reduced))

            elif defend == SparkDuelV2.DODGE:
                # Dodge success: no extra damage
                results.append((config.dodge_chance, chip_atk, chip_def))
                # Dodge fail: full zap damage
                results.append((1.0 - config.dodge_chance, chip_atk, chip_def + zap_dmg))

        # Convert to HP changes
        final = []
        for prob, dmg_to_atk, dmg_to_def in results:
            new_hp_atk = max(0, hp_atk - dmg_to_atk)
            new_hp_def = max(0, hp_def - dmg_to_def)
            final.append((prob, (new_hp_atk, new_hp_def, new_cd_atk)))

        return final

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = SparkDuelV2.Config()

        hp1, hp2, cd1, cd2, turn = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        if turn == 0:
            # P1 attacks, P2 defends
            attacks = SparkDuelV2._available_attacks(cd1)
            defends = [SparkDuelV2.BRACE, SparkDuelV2.DODGE]
        else:
            # P2 attacks, P1 defends
            attacks = SparkDuelV2._available_attacks(cd2)
            defends = [SparkDuelV2.BRACE, SparkDuelV2.DODGE]

        # Enumerate all attack-defend pairs uniformly
        pairs = [(a, d) for a in attacks for d in defends]
        pair_prob = 1.0 / len(pairs)

        transitions = []
        for attack, defend in pairs:
            if turn == 0:
                outcomes = SparkDuelV2._resolve_turn(hp1, hp2, cd1, attack, defend, config)
                for prob, (new_hp_atk, new_hp_def, new_cd) in outcomes:
                    # After P1's attack, P2 attacks next (turn=1)
                    next_state = (new_hp_atk, new_hp_def, new_cd, cd2, 1)
                    transitions.append((pair_prob * prob, next_state))
            else:
                outcomes = SparkDuelV2._resolve_turn(hp2, hp1, cd2, attack, defend, config)
                for prob, (new_hp_atk, new_hp_def, new_cd) in outcomes:
                    # After P2's attack, P1 attacks next (turn=0)
                    # Note: resolve_turn returns (atk_hp, def_hp) = (P2, P1)
                    next_state = (new_hp_def, new_hp_atk, cd1, new_cd, 0)
                    transitions.append((pair_prob * prob, next_state))

        return sanitize_transitions(transitions)

    @staticmethod
    def get_transitions_for_policy(state, atk_policy, def_policy, config=None):
        """Get transitions for specific attacker/defender policies.

        atk_policy: [p_blast, p_zap] (attacker's mixed strategy)
        def_policy: [p_brace, p_dodge] (defender's mixed strategy)
        """
        if config is None:
            config = SparkDuelV2.Config()

        hp1, hp2, cd1, cd2, turn = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        if turn == 0:
            attacks = SparkDuelV2._available_attacks(cd1)
        else:
            attacks = SparkDuelV2._available_attacks(cd2)

        # Normalize attack policy over available actions
        atk_total = sum(atk_policy[a] for a in attacks if a < len(atk_policy))
        if atk_total <= 0:
            atk_weights = {a: 1.0 / len(attacks) for a in attacks}
        else:
            atk_weights = {a: atk_policy[a] / atk_total for a in attacks if a < len(atk_policy)}

        # Defender always has both options
        def_total = def_policy[0] + def_policy[1]
        if def_total <= 0:
            def_weights = {SparkDuelV2.BRACE: 0.5, SparkDuelV2.DODGE: 0.5}
        else:
            def_weights = {
                SparkDuelV2.BRACE: def_policy[0] / def_total,
                SparkDuelV2.DODGE: def_policy[1] / def_total,
            }

        transitions = []
        for attack, aw in atk_weights.items():
            if aw <= 0:
                continue
            for defend, dw in def_weights.items():
                if dw <= 0:
                    continue
                if turn == 0:
                    outcomes = SparkDuelV2._resolve_turn(hp1, hp2, cd1, attack, defend, config)
                    for prob, (new_hp_atk, new_hp_def, new_cd) in outcomes:
                        next_state = (new_hp_atk, new_hp_def, new_cd, cd2, 1)
                        transitions.append((aw * dw * prob, next_state))
                else:
                    outcomes = SparkDuelV2._resolve_turn(hp2, hp1, cd2, attack, defend, config)
                    for prob, (new_hp_atk, new_hp_def, new_cd) in outcomes:
                        next_state = (new_hp_def, new_hp_atk, cd1, new_cd, 0)
                        transitions.append((aw * dw * prob, next_state))

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
        turn_str = "P1atk" if state[4] == 0 else "P2atk"
        return f"HP:{state[0]}v{state[1]} CD:{state[2]}v{state[3]} {turn_str}"
