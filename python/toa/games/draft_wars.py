"""DraftWars — Sequential card draft + auto-battle.

6 cards in a shared pool. Players alternate picking cards (3 each).
Each card has an attack value and a special effect.
After draft, automatic battle: cards played in draft order.

The key insight: draft order IS the game. Each pick changes the
remaining pool, creating cascading uncertainty (A₂+ contribution).

State: (hand1_mask, hand2_mask, turn, whose_pick)
- hand1_mask: bitmask of cards player 1 has drafted
- hand2_mask: bitmask of cards player 2 has drafted
- turn: pick number (0-5)

Terminal after turn 6 (all cards drafted). Battle resolved instantly.

For ToA analysis, we model each player's pick as uniform random over
available cards (average across all strategies).
"""

from toa.game import sanitize_transitions


# Card definitions: (attack, defense_bonus)
# 6 cards with varied power levels to create interesting draft decisions
CARDS = [
    (4, 0),  # Card 0: Heavy hitter
    (3, 1),  # Card 1: Balanced attacker
    (2, 2),  # Card 2: Tank
    (3, 0),  # Card 3: Light attacker
    (1, 3),  # Card 4: Wall
    (5, -1), # Card 5: Glass cannon (negative defense = fragile)
]
NUM_CARDS = 6


def simulate_battle(hand1_mask, hand2_mask):
    """Simulate auto-battle. Returns P(player1 wins).

    Cards are "played" simultaneously. Total attack vs total defense.
    Player with higher net damage wins.
    """
    atk1 = def1 = atk2 = def2 = 0
    for i in range(NUM_CARDS):
        if hand1_mask & (1 << i):
            atk1 += CARDS[i][0]
            def1 += CARDS[i][1]
        if hand2_mask & (1 << i):
            atk2 += CARDS[i][0]
            def2 += CARDS[i][1]

    # Net damage: attack minus opponent's defense bonus
    dmg1 = max(0, atk1 - def2)  # P1's damage to P2
    dmg2 = max(0, atk2 - def1)  # P2's damage to P1

    if dmg1 > dmg2:
        return 1.0
    elif dmg2 > dmg1:
        return 0.0
    else:
        return 0.5  # draw


class DraftWars:
    """Sequential card draft with auto-battle resolution."""

    @staticmethod
    def initial_state():
        # (hand1_mask, hand2_mask, turn)
        # P1 picks on even turns (0, 2, 4), P2 on odd turns (1, 3, 5)
        return (0, 0, 0)

    @staticmethod
    def is_terminal(state):
        _, _, turn = state
        return turn >= NUM_CARDS

    @staticmethod
    def get_transitions(state, config=None):
        hand1, hand2, turn = state
        if turn >= NUM_CARDS:
            return []

        is_p1_turn = (turn % 2 == 0)
        taken = hand1 | hand2
        available = []
        for i in range(NUM_CARDS):
            if not (taken & (1 << i)):
                available.append(i)

        if not available:
            return []

        # Uniform over available picks
        pick_prob = 1.0 / len(available)
        transitions = []

        for card_idx in available:
            if is_p1_turn:
                new_hand1 = hand1 | (1 << card_idx)
                new_hand2 = hand2
            else:
                new_hand1 = hand1
                new_hand2 = hand2 | (1 << card_idx)
            transitions.append((pick_prob, (new_hand1, new_hand2, turn + 1)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        hand1, hand2, turn = state
        if turn < NUM_CARDS:
            return 0.0
        # Battle result
        return simulate_battle(hand1, hand2)

    @staticmethod
    def tostr(state):
        hand1, hand2, turn = state
        cards1 = [i for i in range(NUM_CARDS) if hand1 & (1 << i)]
        cards2 = [i for i in range(NUM_CARDS) if hand2 & (1 << i)]
        return f"P1:{cards1} P2:{cards2} T:{turn}"
