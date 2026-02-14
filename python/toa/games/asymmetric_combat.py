"""Asymmetric Combat — Ultra-High GDS Game Structure.

A symmetric 1v1 combat game where each turn randomly hurts ONE player
but not both. This produces asymmetric outcomes (one gains, other loses)
which maximize per-turn uncertainty.

Key discovery: GDS grows superlinearly with max_hp.
- HP=3:  GDS ~0.68 (nest=5)
- HP=5:  GDS ~0.90 (nest=5)
- HP=10: GDS ~2.05 (nest=5), ~3.00 (nest=10)
- HP=20: GDS ~34.49 (nest=10)

This is the Unbound Conjecture in its most extreme playable form.
"""

from toa.game import sanitize_transitions


class AsymmetricCombat:
    """Asymmetric combat game with configurable HP."""

    class Config:
        def __init__(self, max_hp=10):
            self.max_hp = max_hp

    OUTCOMES = [
        (0.25, 0, -1),    # P2 takes 1 damage
        (0.25, -1, 0),    # P1 takes 1 damage
        (0.14, 0, -2),    # P2 takes 2 damage
        (0.14, -2, 0),    # P1 takes 2 damage
        (0.06, +1, -2),   # P1 heals 1, P2 takes 2
        (0.06, -2, +1),   # P2 heals 1, P1 takes 2
        (0.05, 0, None),  # P2 one-shot (lethal, uses max_hp)
        (0.05, None, 0),  # P1 one-shot (lethal, uses max_hp)
    ]

    @staticmethod
    def initial_state(config=None):
        hp = config.max_hp if config else 10
        return (hp, hp)

    @staticmethod
    def is_terminal(state):
        return state[0] <= 0 or state[1] <= 0

    @staticmethod
    def get_transitions(state, config=None):
        hp1, hp2 = state
        max_hp = config.max_hp if config else 10

        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = []
        for prob, d1, d2 in AsymmetricCombat.OUTCOMES:
            actual_d1 = -max_hp if d1 is None else d1
            actual_d2 = -max_hp if d2 is None else d2
            new1 = max(0, min(max_hp, hp1 + actual_d1))
            new2 = max(0, min(max_hp, hp2 + actual_d2))
            if prob > 1e-9:
                transitions.append((prob, (new1, new2)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        """P1 wins if P1 alive and P2 dead."""
        return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0
