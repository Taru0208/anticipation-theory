"""FFA CoinDuel — Multiplayer coin-flipping game analysis.

Extends CoinDuel to 3-4 players. Each round, all players simultaneously
wager coins and flip them. Player with most heads wins the round.
First to score_target wins takes the game.

Simplified version: fixed wager size (1 coin each) to keep state space manageable.
This isolates the effect of multiplayer dynamics from the resource management layer.

Key question: Does the multiplayer engagement efficiency finding from FFABattle
carry over to the CoinDuel mechanics?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from math import comb
from toa.engine import analyze
from toa.game import sanitize_transitions


def flip_distribution(n_coins):
    """Return [(prob, heads)] for flipping n coins."""
    return [(comb(n_coins, h) / (2 ** n_coins), h) for h in range(n_coins + 1)]


def ffa_coin_round(num_players, coins_per_player=1):
    """Compute round outcomes for N players each flipping coins_per_player coins.

    Returns list of (probability, winner_index) where winner_index=-1 means draw.
    In case of ties among top scorers, it's a draw (re-rolled).
    """
    dist = flip_distribution(coins_per_player)

    # Build all combinations recursively
    def _combos(remaining):
        if remaining == 0:
            return [(1.0, [])]
        result = []
        for prob, heads in dist:
            for sub_prob, sub_heads in _combos(remaining - 1):
                result.append((prob * sub_prob, [heads] + sub_heads))
        return result

    combos = _combos(num_players)

    outcomes = {}  # winner_index -> probability
    for prob, heads_list in combos:
        max_heads = max(heads_list)
        winners = [i for i, h in enumerate(heads_list) if h == max_heads]
        if len(winners) == 1:
            winner = winners[0]
        else:
            winner = -1  # draw
        outcomes[winner] = outcomes.get(winner, 0.0) + prob

    return list(outcomes.items())


class FFACoinDuel:
    """N-player coin duel. Fixed 1-coin wagers for tractability."""

    @staticmethod
    def make_initial(num_players=3, score_target=3):
        """Return (initial_state, config)."""
        state = tuple([0] * num_players)  # all scores start at 0
        config = {'num_players': num_players, 'score_target': score_target}
        return state, config

    @staticmethod
    def is_terminal(state, config):
        return max(state) >= config['score_target']

    @staticmethod
    def get_transitions(state, config):
        if max(state) >= config['score_target']:
            return []

        n = len(state)
        alive = list(range(n))  # all alive in score-based game
        round_outcomes = ffa_coin_round(n)

        transitions = []
        draw_prob = 0.0

        for winner, prob in round_outcomes:
            if winner == -1:
                draw_prob += prob
            else:
                s = list(state)
                s[winner] = s[winner] + 1
                transitions.append((prob, tuple(s)))

        # Redistribute draw probability proportionally (re-roll on draw)
        if draw_prob > 0 and transitions:
            decisive = 1.0 - draw_prob
            if decisive > 0:
                transitions = [(p / decisive, s) for p, s in transitions]

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state, config):
        # Player 0 wins if they reach score_target first
        if state[0] >= config['score_target'] and all(
            state[i] < config['score_target'] for i in range(1, len(state))
        ):
            return 1.0
        return 0.0


def analyze_ffa_coin(num_players, score_target, coins=1):
    """Analyze FFA CoinDuel and return result."""
    state, config = FFACoinDuel.make_initial(num_players, score_target)
    return analyze(
        initial_state=state,
        is_terminal=lambda s: FFACoinDuel.is_terminal(s, config),
        get_transitions=lambda s, c=None: FFACoinDuel.get_transitions(s, config),
        compute_intrinsic_desire=lambda s: FFACoinDuel.compute_intrinsic_desire(s, config),
        nest_level=10,
    )


def print_result(label, result, initial_state):
    gds = result.game_design_score
    a1 = result.gds_components[0]
    a2p = gds - a1
    depth = a2p / gds * 100 if gds > 0 else 0
    d = result.state_nodes[initial_state].d_global
    eff = gds / d if d > 0 else 0
    print(f"  {label:<35} GDS={gds:.4f}  A1={a1:.4f}  Depth={depth:.1f}%  P(win)={d:.3f}  Eff={eff:.3f}  States={len(result.states)}")


def experiment_1_player_count():
    """How does player count affect FFA CoinDuel?"""
    print("=" * 80)
    print("EXPERIMENT 1: FFA CoinDuel — Player Count Scaling")
    print("=" * 80)
    print()

    for target in [2, 3]:
        print(f"--- Score target = {target} ---")
        for n in [2, 3, 4]:
            r = analyze_ffa_coin(n, target)
            initial = tuple([0] * n)
            print_result(f"N={n}, target={target}", r, initial)
        print()


def experiment_2_score_target():
    """How does increasing target (game length) affect engagement?"""
    print("=" * 80)
    print("EXPERIMENT 2: FFA CoinDuel — Score Target Scaling (N=3)")
    print("=" * 80)
    print()

    for target in [2, 3, 4, 5]:
        r = analyze_ffa_coin(3, target)
        initial = tuple([0] * 3)
        print_result(f"3P target={target}", r, initial)


def experiment_3_comparison_with_hp_battle():
    """Compare CoinDuel FFA with HP Battle FFA for same player count."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CoinDuel FFA vs HP Battle FFA")
    print("=" * 80)
    print()

    from toa.games.ffa_battle import FFABattle

    # 3-player comparison
    print("--- 3 Players ---")
    r_coin = analyze_ffa_coin(3, 3)
    print_result("CoinDuel FFA (target=3)", r_coin, (0, 0, 0))

    r_hp = analyze(
        initial_state=FFABattle.initial_state(num_players=3, hp=3),
        is_terminal=FFABattle.is_terminal,
        get_transitions=FFABattle.get_transitions,
        compute_intrinsic_desire=FFABattle.compute_intrinsic_desire,
        nest_level=10,
    )
    print_result("HP Battle FFA (hp=3)", r_hp, (3, 3, 3))


if __name__ == "__main__":
    experiment_1_player_count()
    experiment_2_score_target()
    experiment_3_comparison_with_hp_battle()
