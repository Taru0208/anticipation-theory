"""HpGame_Heal — 1v1 HP battle with comeback mechanics.

Two comeback variants that avoid self-loops (engine requires DAG):

Variant A — "Damage Shield": When the trailing player (lower HP) is hit,
there's a chance the damage is negated. To avoid self-loops, the shield
only activates when HP difference >= 2 (so at least one unit of damage
was already "absorbed" by the gap). When activated, the hit simply does
0 damage to the trailing player, but their HP is capped at current value
(no healing above current).

Actually, let's use a cleaner approach:

Variant B — "Comeback Bonus": The trailing player (lower HP) gets bonus
damage on their hits. When trailing by 1+, hits deal 2 damage with
heal_chance probability. This ALWAYS reduces HP (never increases),
so no cycles possible. This is a rubber-band mechanic: falling behind
makes your attacks stronger, softening the consequences of being behind.

State: (hp1, hp2) tuple.
Terminal when either HP reaches 0.
Win condition: P1 alive and P2 dead.
"""

from toa.game import sanitize_transitions


class HpGameHeal:
    """HpGame with comeback bonus — trailing player's hits deal bonus damage."""

    class Config:
        def __init__(self, max_hp=5, heal_chance=0.3):
            """
            heal_chance: probability that the trailing player's hit deals
            2 damage instead of 1 (comeback bonus).
            """
            self.max_hp = max_hp
            self.comeback_chance = heal_chance  # reusing param name for API compat

    @staticmethod
    def initial_state():
        return (5, 5)

    @staticmethod
    def is_terminal(state):
        hp1, hp2 = state
        return hp1 <= 0 or hp2 <= 0

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = HpGameHeal.Config()

        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        p_cb = config.comeback_chance
        p_normal = 1.0 - p_cb

        # Determine who is trailing
        p1_trailing = hp1 < hp2
        p2_trailing = hp2 < hp1
        # If tied, nobody gets comeback bonus

        # Base damage for each player's hit
        # Trailing player: deal 2 with p_cb, deal 1 with (1-p_cb)
        # Non-trailing: always deal 1

        def clamp(hp):
            return max(0, hp)

        transitions = []

        # --- Outcome 1: P1 hits, P2 misses (base prob 1/3) ---
        base_p = 1.0 / 3.0
        if p1_trailing:
            # P1 trailing → P1's hit might deal 2 damage
            transitions.append((base_p * p_normal, (hp1, clamp(hp2 - 1))))
            transitions.append((base_p * p_cb, (hp1, clamp(hp2 - 2))))
        else:
            transitions.append((base_p, (hp1, clamp(hp2 - 1))))

        # --- Outcome 2: Both hit (base prob 1/3) ---
        base_p = 1.0 / 3.0
        if p1_trailing and not p2_trailing:
            # P1 trailing: P1 deals 1 or 2, P2 deals 1
            transitions.append((base_p * p_normal, (clamp(hp1 - 1), clamp(hp2 - 1))))
            transitions.append((base_p * p_cb, (clamp(hp1 - 1), clamp(hp2 - 2))))
        elif p2_trailing and not p1_trailing:
            # P2 trailing: P2 deals 1 or 2, P1 deals 1
            transitions.append((base_p * p_normal, (clamp(hp1 - 1), clamp(hp2 - 1))))
            transitions.append((base_p * p_cb, (clamp(hp1 - 2), clamp(hp2 - 1))))
        else:
            # Tied: both deal 1
            transitions.append((base_p, (clamp(hp1 - 1), clamp(hp2 - 1))))

        # --- Outcome 3: P2 hits, P1 misses (base prob 1/3) ---
        base_p = 1.0 / 3.0
        if p2_trailing:
            # P2 trailing → P2's hit might deal 2 damage
            transitions.append((base_p * p_normal, (clamp(hp1 - 1), hp2)))
            transitions.append((base_p * p_cb, (clamp(hp1 - 2), hp2)))
        else:
            transitions.append((base_p, (clamp(hp1 - 1), hp2)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hp1, hp2 = state
        return 1.0 if hp1 > 0 and hp2 <= 0 else 0.0

    @staticmethod
    def tostr(state):
        return f"HP1:{state[0]} HP2:{state[1]}"
