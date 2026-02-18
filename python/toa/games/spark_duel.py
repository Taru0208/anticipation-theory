"""SparkDuel — Simultaneous-action combat with guaranteed progress.

Two players start with HP. Each turn:
  1. Both take 1 chip damage (the "spark" — ensures game always progresses).
  2. Both simultaneously choose an action:
     - Blast: 70% chance to deal 2 extra damage. Goes on 1-turn cooldown.
     - Zap:   Deal 1 extra damage (guaranteed).
     - Guard: Reduce incoming damage by 1 this turn.
  3. When Blast is on cooldown, only Zap and Guard are available.

First to reach 0 HP loses. Simultaneous death = draw.

State: (hp1, hp2, cd1, cd2) where cd is cooldown turns remaining.
The chip damage ensures every turn reduces total HP → DAG guaranteed.
"""

from toa.game import sanitize_transitions


class SparkDuel:
    """Two-player simultaneous combat with chip damage and cooldown."""

    class Config:
        def __init__(
            self,
            max_hp=5,
            chip_damage=1,       # automatic damage each turn to both
            blast_damage=2,      # extra damage on top of chip
            blast_hit_rate=0.70,
            zap_damage=1,        # extra damage on top of chip
            guard_reduction=1,   # reduce incoming damage
        ):
            self.max_hp = max_hp
            self.chip_damage = chip_damage
            self.blast_damage = blast_damage
            self.blast_hit_rate = blast_hit_rate
            self.zap_damage = zap_damage
            self.guard_reduction = guard_reduction

    BLAST = 0
    ZAP = 1
    GUARD = 2
    ACTION_NAMES = ["Blast", "Zap", "Guard"]

    @staticmethod
    def initial_state(config=None):
        if config is None:
            config = SparkDuel.Config()
        return (config.max_hp, config.max_hp, 0, 0)

    @staticmethod
    def is_terminal(state):
        hp1, hp2, _, _ = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def _available_actions(cooldown):
        """Return list of action indices available given cooldown state."""
        if cooldown > 0:
            return [SparkDuel.ZAP, SparkDuel.GUARD]
        return [SparkDuel.BLAST, SparkDuel.ZAP, SparkDuel.GUARD]

    @staticmethod
    def _resolve_pair(hp1, hp2, cd1, cd2, a1, a2, config):
        """Resolve one action pair. Returns [(probability, next_state), ...]."""
        # Next turn cooldowns
        new_cd1 = 1 if a1 == SparkDuel.BLAST else max(0, cd1 - 1)
        new_cd2 = 1 if a2 == SparkDuel.BLAST else max(0, cd2 - 1)

        # Base damage: chip damage to both
        base_dmg_to_p1 = config.chip_damage
        base_dmg_to_p2 = config.chip_damage

        # Guard reduction
        guard1 = config.guard_reduction if a1 == SparkDuel.GUARD else 0
        guard2 = config.guard_reduction if a2 == SparkDuel.GUARD else 0

        # Extra damage from attacks (probabilistic for Blast)
        def extra_damage_outcomes(action, config):
            """Returns [(probability, extra_damage)]"""
            if action == SparkDuel.BLAST:
                return [
                    (config.blast_hit_rate, config.blast_damage),
                    (1.0 - config.blast_hit_rate, 0),
                ]
            elif action == SparkDuel.ZAP:
                return [(1.0, config.zap_damage)]
            else:
                return [(1.0, 0)]

        p1_extra = extra_damage_outcomes(a1, config)  # P1's attack → extra dmg to P2
        p2_extra = extra_damage_outcomes(a2, config)  # P2's attack → extra dmg to P1

        results = []
        for p1_prob, extra_to_p2 in p1_extra:
            for p2_prob, extra_to_p1 in p2_extra:
                joint = p1_prob * p2_prob
                # Chip damage is unavoidable. Guard only reduces extra attack damage.
                reduced_extra_to_p1 = max(0, extra_to_p1 - guard1)
                reduced_extra_to_p2 = max(0, extra_to_p2 - guard2)
                total_to_p1 = base_dmg_to_p1 + reduced_extra_to_p1
                total_to_p2 = base_dmg_to_p2 + reduced_extra_to_p2
                new_hp1 = max(0, hp1 - total_to_p1)
                new_hp2 = max(0, hp2 - total_to_p2)
                results.append((joint, (new_hp1, new_hp2, new_cd1, new_cd2)))

        return results

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = SparkDuel.Config()

        hp1, hp2, cd1, cd2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        actions1 = SparkDuel._available_actions(cd1)
        actions2 = SparkDuel._available_actions(cd2)

        action_pairs = [(a1, a2) for a1 in actions1 for a2 in actions2]
        pair_prob = 1.0 / len(action_pairs)

        transitions = []
        for a1, a2 in action_pairs:
            outcomes = SparkDuel._resolve_pair(hp1, hp2, cd1, cd2, a1, a2, config)
            for prob, next_state in outcomes:
                transitions.append((pair_prob * prob, next_state))

        return sanitize_transitions(transitions)

    @staticmethod
    def get_transitions_for_policy(state, policy1, policy2, config=None):
        """Get transitions for specific action distributions.

        policy1, policy2: [p_blast, p_zap, p_guard]
        Unavailable actions are skipped (their probability redistributed).
        """
        if config is None:
            config = SparkDuel.Config()

        hp1, hp2, cd1, cd2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        avail1 = SparkDuel._available_actions(cd1)
        avail2 = SparkDuel._available_actions(cd2)

        # Redistribute policy over available actions
        def normalize_policy(policy, available):
            total = sum(policy[a] for a in available)
            if total <= 0:
                return {a: 1.0 / len(available) for a in available}
            return {a: policy[a] / total for a in available}

        p1 = normalize_policy(policy1, avail1)
        p2 = normalize_policy(policy2, avail2)

        transitions = []
        for a1, w1 in p1.items():
            if w1 <= 0:
                continue
            for a2, w2 in p2.items():
                if w2 <= 0:
                    continue
                outcomes = SparkDuel._resolve_pair(hp1, hp2, cd1, cd2, a1, a2, config)
                for prob, next_state in outcomes:
                    transitions.append((w1 * w2 * prob, next_state))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2, _, _ = state
        if hp1 <= 0 and hp2 <= 0:
            return 0.5  # Mutual destruction = draw
        if hp2 <= 0:
            return 1.0  # Player 1 wins
        if hp1 <= 0:
            return 0.0  # Player 2 wins
        return 0.0  # Non-terminal shouldn't be called

    @staticmethod
    def tostr(state):
        return f"HP:{state[0]}v{state[1]} CD:{state[2]}v{state[3]}"
