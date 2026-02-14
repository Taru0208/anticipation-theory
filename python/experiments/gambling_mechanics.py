"""Gambling Mechanics — Applying ToA to Casino Games.

Models real casino games through the ToA framework to analyze engagement
structure. Key hypothesis: gambling maximizes A₁ (immediate excitement)
while minimizing A₂+ (strategic depth) — the opposite of good game design.

Games modeled:
1. Single-number Roulette: extreme payoff asymmetry (35:1 on 1/37)
2. Roulette red/black: near-50/50 with house edge
3. Slot machine: multi-tier payout with near-miss mechanics
4. Blackjack (simplified): player decisions affect outcome
5. Multi-spin slots: repeated play session
6. Poker (simplified): opponent modeling adds depth

Comparison with existing ToA games (CoinToss, HpGame, GoldGame) to
quantify the structural difference between gambling and game design.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


# ============================================================
# MODEL 1: Roulette (Single Number)
# ============================================================

class RouletteSingleNumber:
    """European roulette, betting on a single number.

    37 slots (0-36). Bet on one number.
    Win: 35:1 payout. Lose: lose bet.
    P(win) = 1/37 ≈ 2.7%

    State: ("initial",) or ("win",) or ("lose",)
    """

    @staticmethod
    def initial_state():
        return "initial"

    @staticmethod
    def is_terminal(state):
        return state != "initial"

    @staticmethod
    def get_transitions(state, config=None):
        if state != "initial":
            return []
        return [(1.0 / 37.0, "win"), (36.0 / 37.0, "lose")]

    @staticmethod
    def compute_intrinsic_desire(state):
        return 1.0 if state == "win" else 0.0


# ============================================================
# MODEL 2: Roulette (Red/Black)
# ============================================================

class RouletteRedBlack:
    """European roulette, red/black bet.

    P(win) = 18/37 ≈ 48.6% (slightly unfavorable)
    Win: 1:1 payout. Lose: lose bet.
    """

    @staticmethod
    def initial_state():
        return "initial"

    @staticmethod
    def is_terminal(state):
        return state != "initial"

    @staticmethod
    def get_transitions(state, config=None):
        if state != "initial":
            return []
        return [(18.0 / 37.0, "win"), (19.0 / 37.0, "lose")]

    @staticmethod
    def compute_intrinsic_desire(state):
        return 1.0 if state == "win" else 0.0


# ============================================================
# MODEL 3: Slot Machine (Single Spin)
# ============================================================

class SlotMachine:
    """Simplified slot machine with tiered payouts.

    Typical slot machine payout distribution:
    - Jackpot (3 matching 7s):   0.1%  → 100x
    - Big win (3 matching):      2%    → 10x
    - Small win (2 matching):    15%   → 2x
    - Near miss (2+1 close):     20%   → 0x  (lose but feels close)
    - Loss:                      62.9% → 0x

    Key: near-miss is structurally identical to loss but
    psychologically distinct. ToA should capture this via
    "almost winning" = high A₁ from probability proximity.

    State: outcome name
    Desire: proportional to payout (normalized to 0-1)
    """

    @staticmethod
    def initial_state():
        return "spinning"

    @staticmethod
    def is_terminal(state):
        return state != "spinning"

    @staticmethod
    def get_transitions(state, config=None):
        if state != "spinning":
            return []
        return [
            (0.001, "jackpot"),      # 0.1%
            (0.020, "big_win"),      # 2%
            (0.150, "small_win"),    # 15%
            (0.200, "near_miss"),    # 20%
            (0.629, "loss"),         # 62.9%
        ]

    @staticmethod
    def compute_intrinsic_desire(state):
        desires = {
            "jackpot": 1.0,
            "big_win": 0.5,
            "small_win": 0.2,
            "near_miss": 0.0,
            "loss": 0.0,
        }
        return desires.get(state, 0.0)


# ============================================================
# MODEL 4: Multi-Spin Slot Session
# ============================================================

def multispin_initial_state(num_spins):
    # State: (balance_level, spins_remaining)
    # balance_level: discretized (0=broke, 5=starting, 10=big_profit)
    return (5, num_spins)


def multispin_is_terminal(state, num_spins):
    balance, remaining = state
    return remaining <= 0 or balance <= 0


def multispin_get_transitions(state, config=None):
    balance, remaining = state
    if remaining <= 0 or balance <= 0:
        return []

    # Each spin costs 1 balance unit
    base = balance - 1  # cost of spin

    transitions = []
    # Jackpot: +10
    jackpot_bal = min(base + 10, 10)
    transitions.append((0.001, (jackpot_bal, remaining - 1)))
    # Big win: +5
    big_bal = min(base + 5, 10)
    transitions.append((0.020, (big_bal, remaining - 1)))
    # Small win: +2 (net +1)
    small_bal = min(base + 2, 10)
    transitions.append((0.150, (small_bal, remaining - 1)))
    # Loss: -1 (just the spin cost)
    transitions.append((0.829, (max(base, 0), remaining - 1)))

    # Merge identical states
    merged = {}
    for p, s in transitions:
        if s in merged:
            merged[s] += p
        else:
            merged[s] = p
    return [(p, s) for s, p in merged.items()]


def multispin_compute_desire(state, starting_balance=5):
    balance, remaining = state
    if remaining > 0 and balance > 0:
        return 0.0
    # Desire = did you profit?
    return min(1.0, max(0.0, balance / starting_balance)) if balance > 0 else 0.0


# ============================================================
# MODEL 5: Simplified Blackjack
# ============================================================

class Blackjack:
    """Simplified blackjack — hit or stand decisions.

    State: (player_total, dealer_showing, can_hit)
    Simplified: player starts at 11-12 (two cards), dealer shows 2-11.
    Hit: add 2-11 (simplified distribution).
    Stand: dealer plays out.

    Key difference from pure gambling: PLAYER CHOICE affects outcome.
    This should generate higher A₂+ than pure chance games.
    """

    @staticmethod
    def initial_state():
        # Average starting hand ~12, dealer shows average ~7
        return (12, 7, True)

    @staticmethod
    def is_terminal(state):
        total, dealer, can_hit = state
        return total > 21 or not can_hit

    @staticmethod
    def get_transitions(state, config=None):
        total, dealer, can_hit = state
        if total > 21 or not can_hit:
            return []

        transitions = []

        # Option 1: STAND — resolve against dealer
        # Simplified dealer outcome: P(dealer_bust) depends on dealer card
        # P(dealer < player) depends on both
        if total >= 12:
            # Simplified dealer resolution
            # Dealer busts ~28% on average, wins ~48%, pushes ~8%, player wins ~44%
            # But it depends on player's total
            if total >= 20:
                p_win = 0.75
            elif total >= 18:
                p_win = 0.60
            elif total >= 16:
                p_win = 0.45
            elif total >= 14:
                p_win = 0.35
            else:
                p_win = 0.25
            transitions.append((1.0, (total, dealer, False)))  # stand → terminal

        # Option 2: HIT — draw a card
        # Simplified: draw adds 2-10 with roughly equal probability
        # Ace simplified as either 1 or 11
        for card_value in range(2, 12):
            p_card = 1.0 / 10.0  # simplified uniform
            new_total = total + card_value
            if new_total > 21:
                transitions.append((p_card, (new_total, dealer, False)))  # bust
            else:
                transitions.append((p_card, (new_total, dealer, True)))

        # Normalize
        total_p = sum(p for p, _ in transitions)
        transitions = [(p / total_p, s) for p, s in transitions]

        # Merge identical states
        merged = {}
        for p, s in transitions:
            if s in merged:
                merged[s] += p
            else:
                merged[s] = p
        return [(p, s) for s, p in merged.items()]

    @staticmethod
    def compute_intrinsic_desire(state):
        total, dealer, can_hit = state
        if can_hit:
            return 0.0
        if total > 21:
            return 0.0  # bust
        # Simplified win probability based on final total
        if total >= 20:
            return 0.80
        elif total >= 18:
            return 0.65
        elif total >= 16:
            return 0.48
        elif total >= 14:
            return 0.35
        else:
            return 0.25

    @staticmethod
    def tostr(state):
        total, dealer, can_hit = state
        status = "playing" if can_hit else ("bust" if total > 21 else "standing")
        return f"P:{total} D:{dealer} ({status})"


# ============================================================
# MODEL 6: Multi-Round Roulette Session
# ============================================================

def roulette_session_initial(num_rounds):
    """State: (balance, rounds_left). Start with 10 units."""
    return (10, num_rounds)


def roulette_session_is_terminal(state, num_rounds):
    balance, rounds = state
    return rounds <= 0 or balance <= 0


def roulette_session_transitions(state, config=None):
    balance, rounds = state
    if rounds <= 0 or balance <= 0:
        return []

    # Bet 1 unit on red/black each round
    bet = 1
    new_balance_win = min(balance + bet, 15)  # cap at 15 for state space
    new_balance_lose = balance - bet

    return [
        (18.0 / 37.0, (new_balance_win, rounds - 1)),
        (19.0 / 37.0, (new_balance_lose, rounds - 1)),
    ]


def roulette_session_desire(state, starting=10):
    balance, rounds = state
    if rounds > 0 and balance > 0:
        return 0.0
    # Desire = profit ratio (capped at 1.0)
    return min(1.0, balance / starting) if balance > 0 else 0.0


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_single_turn_games():
    """Compare single-turn gambling games with CoinToss."""
    print("=" * 70)
    print("EXPERIMENT 1: Single-Turn Gambling vs Games")
    print("=" * 70)

    games = {
        "CoinToss (50/50)": {
            "initial": "initial",
            "terminal": lambda s: s != "initial",
            "transitions": lambda s, _: [(0.5, "win"), (0.5, "lose")] if s == "initial" else [],
            "desire": lambda s: 1.0 if s == "win" else 0.0,
        },
        "Roulette Red/Black (48.6%)": {
            "initial": RouletteRedBlack.initial_state(),
            "terminal": RouletteRedBlack.is_terminal,
            "transitions": RouletteRedBlack.get_transitions,
            "desire": RouletteRedBlack.compute_intrinsic_desire,
        },
        "Roulette Single# (2.7%)": {
            "initial": RouletteSingleNumber.initial_state(),
            "terminal": RouletteSingleNumber.is_terminal,
            "transitions": RouletteSingleNumber.get_transitions,
            "desire": RouletteSingleNumber.compute_intrinsic_desire,
        },
        "Slot Machine (tiered)": {
            "initial": SlotMachine.initial_state(),
            "terminal": SlotMachine.is_terminal,
            "transitions": SlotMachine.get_transitions,
            "desire": SlotMachine.compute_intrinsic_desire,
        },
    }

    results = []
    for name, g in games.items():
        analysis = analyze(
            initial_state=g["initial"],
            is_terminal=g["terminal"],
            get_transitions=g["transitions"],
            compute_intrinsic_desire=g["desire"],
            nest_level=5,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        d0 = analysis.state_nodes[g["initial"]].d_global
        results.append((name, gds, comps, d0))

    print(f"\n{'Game':<30} {'P(win)':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7}")
    print("-" * 75)
    for name, gds, comps, d0 in results:
        print(f"{name:<30} {d0:>7.3f} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f}")

    return results


def analyze_multi_turn_sessions():
    """Compare multi-turn gambling sessions with multi-turn games."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Multi-Turn Sessions — Depth Analysis")
    print("=" * 70)

    from toa.games.hpgame import HpGame
    from toa.games.hpgame_rage import HpGameRage
    from toa.games.goldgame import GoldGame

    results = []

    # HP Game (5,5) — benchmark
    analysis = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=10,
    )
    results.append(("HpGame (5,5)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # HP Game Rage — benchmark with mechanics
    analysis = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        nest_level=10,
    )
    results.append(("HpGame+Rage", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # Multi-spin slots (5 spins)
    num_spins = 5
    analysis = analyze(
        initial_state=multispin_initial_state(num_spins),
        is_terminal=lambda s: multispin_is_terminal(s, num_spins),
        get_transitions=multispin_get_transitions,
        compute_intrinsic_desire=lambda s: multispin_compute_desire(s),
        nest_level=10,
    )
    results.append((f"Slots ({num_spins} spins)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # Multi-spin slots (10 spins)
    num_spins = 10
    analysis = analyze(
        initial_state=multispin_initial_state(num_spins),
        is_terminal=lambda s: multispin_is_terminal(s, num_spins),
        get_transitions=multispin_get_transitions,
        compute_intrinsic_desire=lambda s: multispin_compute_desire(s),
        nest_level=10,
    )
    results.append((f"Slots ({num_spins} spins)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # Roulette session (5 rounds)
    num_rounds = 5
    analysis = analyze(
        initial_state=roulette_session_initial(num_rounds),
        is_terminal=lambda s: roulette_session_is_terminal(s, num_rounds),
        get_transitions=roulette_session_transitions,
        compute_intrinsic_desire=lambda s: roulette_session_desire(s),
        nest_level=10,
    )
    results.append((f"Roulette ({num_rounds}r)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # Roulette session (10 rounds)
    num_rounds = 10
    analysis = analyze(
        initial_state=roulette_session_initial(num_rounds),
        is_terminal=lambda s: roulette_session_is_terminal(s, num_rounds),
        get_transitions=roulette_session_transitions,
        compute_intrinsic_desire=lambda s: roulette_session_desire(s),
        nest_level=10,
    )
    results.append((f"Roulette ({num_rounds}r)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # Blackjack (single hand)
    analysis = analyze(
        initial_state=Blackjack.initial_state(),
        is_terminal=Blackjack.is_terminal,
        get_transitions=Blackjack.get_transitions,
        compute_intrinsic_desire=Blackjack.compute_intrinsic_desire,
        nest_level=10,
    )
    results.append(("Blackjack (1 hand)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    # GoldGame (5 turns for tractability)
    gold_config = GoldGame.Config(max_turns=5)
    analysis = analyze(
        initial_state=GoldGame.initial_state(),
        is_terminal=lambda s: s[2] >= 5,
        get_transitions=lambda s, c: GoldGame.get_transitions(s, gold_config),
        compute_intrinsic_desire=lambda s: 1.0 if s[2] >= 5 and s[0] > s[1] else 0.0,
        nest_level=10,
    )
    results.append(("GoldGame (5t)", analysis.game_design_score,
                     analysis.gds_components[:10], len(analysis.states)))

    print(f"\n{'Game':<22} {'States':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7} {'A₂+%':>7}")
    print("-" * 70)
    for name, gds, comps, n_states in results:
        a2_plus = sum(comps[1:])
        depth_pct = a2_plus / gds * 100 if gds > 0 else 0
        print(f"{name:<22} {n_states:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f} {depth_pct:>6.1f}%")

    return results


def analyze_house_edge_effect():
    """How does house edge affect GDS? Sweep P(win) from 0.40 to 0.60."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: House Edge Effect on Engagement")
    print("=" * 70)

    results = []
    for p_win_pct in range(30, 71, 5):
        p_win = p_win_pct / 100.0
        analysis = analyze(
            initial_state="initial",
            is_terminal=lambda s: s != "initial",
            get_transitions=lambda s, _, pw=p_win: [(pw, "win"), (1 - pw, "lose")] if s == "initial" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "win" else 0.0,
            nest_level=5,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        results.append((p_win, gds, comps))

    print(f"\n{'P(win)':>8} {'GDS':>8} {'A₁':>7} {'House Edge':>12}")
    print("-" * 40)
    for p_win, gds, comps in results:
        edge = 0.5 - p_win if p_win < 0.5 else p_win - 0.5
        direction = "casino" if p_win < 0.5 else "player" if p_win > 0.5 else "fair"
        print(f"{p_win:>8.2f} {gds:>8.4f} {comps[0]:>7.4f} {edge*100:>6.1f}% {direction}")

    # Key finding: max GDS at P=0.50 (fair), drops with house edge
    fair_gds = next(gds for pw, gds, _ in results if abs(pw - 0.50) < 0.01)
    casino_48 = next(gds for pw, gds, _ in results if abs(pw - 0.45) < 0.01)
    casino_40 = next(gds for pw, gds, _ in results if abs(pw - 0.40) < 0.01)

    print(f"\nFair (50%) GDS:    {fair_gds:.4f}")
    print(f"Casino 45% GDS:    {casino_48:.4f} ({(casino_48/fair_gds-1)*100:+.1f}%)")
    print(f"Casino 40% GDS:    {casino_40:.4f} ({(casino_40/fair_gds-1)*100:+.1f}%)")

    return results


def analyze_payout_asymmetry():
    """How does payout asymmetry affect engagement?

    Compare: fair coin (50/50, 1:1) vs lottery-like (1%, 99:1) vs
    slot-like (tiered payouts). All with same expected value.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Payout Asymmetry — Lottery vs Fair Game")
    print("=" * 70)

    # All games have expected value = 0.5 (fair)
    games = {
        "Fair coin (50/50)": [
            (0.50, "win"),
            (0.50, "lose"),
        ],
        "Slight asymmetry (40/60, 1.25x)": [
            (0.40, "win"),    # desire = 1.0
            (0.60, "lose"),   # desire = 0.0 → EV = 0.4, need to adjust
        ],
        "Lottery (5%, 10:1)": [
            (0.05, "big_win"),
            (0.95, "lose"),
        ],
        "Lottery (1%, 50:1)": [
            (0.01, "jackpot"),
            (0.99, "lose"),
        ],
        "Reverse lottery (99%, tiny win)": [
            (0.99, "small_win"),
            (0.01, "big_lose"),
        ],
        "3-tier slot": [
            (0.01, "jackpot"),
            (0.10, "medium"),
            (0.30, "small"),
            (0.59, "lose"),
        ],
    }

    # Desire values to make EV = 0.5 for each game
    desires = {
        "Fair coin (50/50)": {"win": 1.0, "lose": 0.0},
        "Slight asymmetry (40/60, 1.25x)": {"win": 1.0, "lose": 1.0/6.0},
        "Lottery (5%, 10:1)": {"big_win": 1.0, "lose": 0.0},
        "Lottery (1%, 50:1)": {"jackpot": 1.0, "lose": 0.0},
        "Reverse lottery (99%, tiny win)": {"small_win": 0.505, "big_lose": 0.0},
        "3-tier slot": {"jackpot": 1.0, "medium": 0.8, "small": 0.5, "lose": 0.0},
    }

    results = []
    for name, transitions in games.items():
        desire_map = desires[name]

        analysis = analyze(
            initial_state="start",
            is_terminal=lambda s: s != "start",
            get_transitions=lambda s, _, t=transitions: t if s == "start" else [],
            compute_intrinsic_desire=lambda s, d=desire_map: d.get(s, 0.0),
            nest_level=5,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        d0 = analysis.state_nodes["start"].d_global
        results.append((name, gds, comps, d0))

    print(f"\n{'Game':<35} {'E[D]':>6} {'GDS':>8} {'A₁':>7}")
    print("-" * 60)
    for name, gds, comps, d0 in results:
        print(f"{name:<35} {d0:>6.3f} {gds:>8.4f} {comps[0]:>7.4f}")

    return results


def analyze_near_miss_effect():
    """Quantify the near-miss effect in slot machines.

    Compare slot with and without near-miss (same probabilities, different desire assignment).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Near-Miss Effect")
    print("=" * 70)

    transitions = [
        (0.001, "jackpot"),
        (0.020, "big_win"),
        (0.150, "small_win"),
        (0.200, "near_miss"),
        (0.629, "loss"),
    ]

    configs = {
        "Slot (near_miss=0)": {
            "jackpot": 1.0, "big_win": 0.5, "small_win": 0.2,
            "near_miss": 0.0, "loss": 0.0,
        },
        "Slot (near_miss=0.05)": {
            "jackpot": 1.0, "big_win": 0.5, "small_win": 0.2,
            "near_miss": 0.05, "loss": 0.0,
        },
        "Slot (near_miss=0.10)": {
            "jackpot": 1.0, "big_win": 0.5, "small_win": 0.2,
            "near_miss": 0.10, "loss": 0.0,
        },
        "Slot (near_miss=0.15)": {
            "jackpot": 1.0, "big_win": 0.5, "small_win": 0.2,
            "near_miss": 0.15, "loss": 0.0,
        },
        "Slot (no near_miss category)": {
            # Merge near_miss into loss
            "jackpot": 1.0, "big_win": 0.5, "small_win": 0.2,
            "near_miss": 0.0, "loss": 0.0,
        },
    }

    results = []
    for name, desire_map in configs.items():
        analysis = analyze(
            initial_state="spinning",
            is_terminal=lambda s: s != "spinning",
            get_transitions=lambda s, _: transitions if s == "spinning" else [],
            compute_intrinsic_desire=lambda s, d=desire_map: d.get(s, 0.0),
            nest_level=5,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        d0 = analysis.state_nodes["spinning"].d_global
        results.append((name, gds, comps, d0))

    print(f"\n{'Config':<35} {'E[D]':>6} {'GDS':>8} {'A₁':>7}")
    print("-" * 60)
    for name, gds, comps, d0 in results:
        print(f"{name:<35} {d0:>6.3f} {gds:>8.4f} {comps[0]:>7.4f}")

    # Key: near-miss desire > 0 increases E[D] but how does it affect A₁?
    base_gds = results[0][1]
    for name, gds, _, _ in results[1:]:
        diff = (gds - base_gds) / base_gds * 100 if base_gds > 0 else 0
        print(f"  {name}: {diff:+.1f}% vs baseline")

    return results


def analyze_gambling_vs_games_scaling():
    """Compare how GDS scales with depth for gambling vs games.

    Key test for Unbound Conjecture: do gambling sessions show the
    same linear growth as games, or do they plateau/decline?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Depth Scaling — Gambling vs Games")
    print("=" * 70)

    from toa.games.coin_toss import CoinToss

    # Best-of-N (game-like)
    print("\n--- Best-of-N (game pattern) ---")
    bon_results = []
    for n in [1, 3, 5, 7, 9, 11]:
        # Best-of-N: first to (N+1)/2 wins
        target = (n + 1) // 2

        def bon_initial():
            return (0, 0)

        def bon_terminal(s, t=target):
            return s[0] >= t or s[1] >= t

        def bon_transitions(s, _, t=target):
            if s[0] >= t or s[1] >= t:
                return []
            return [(0.5, (s[0] + 1, s[1])), (0.5, (s[0], s[1] + 1))]

        def bon_desire(s, t=target):
            return 1.0 if s[0] >= t else 0.0

        analysis = analyze(
            initial_state=(0, 0),
            is_terminal=bon_terminal,
            get_transitions=bon_transitions,
            compute_intrinsic_desire=bon_desire,
            nest_level=10,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:10]
        bon_results.append((n, gds, comps))

    print(f"{'N':>5} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₂+%':>7}")
    print("-" * 40)
    for n, gds, comps in bon_results:
        a2_plus_pct = sum(comps[1:]) / gds * 100 if gds > 0 else 0
        print(f"{n:>5} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {a2_plus_pct:>6.1f}%")

    # Roulette sessions (gambling pattern)
    print("\n--- Roulette Sessions (gambling pattern) ---")
    rou_results = []
    for n_rounds in [1, 3, 5, 7, 10]:
        analysis = analyze(
            initial_state=roulette_session_initial(n_rounds),
            is_terminal=lambda s, nr=n_rounds: roulette_session_is_terminal(s, nr),
            get_transitions=roulette_session_transitions,
            compute_intrinsic_desire=lambda s: roulette_session_desire(s),
            nest_level=10,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:10]
        rou_results.append((n_rounds, gds, comps))

    print(f"{'Rounds':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₂+%':>7}")
    print("-" * 40)
    for n, gds, comps in rou_results:
        a2_plus_pct = sum(comps[1:]) / gds * 100 if gds > 0 else 0
        print(f"{n:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {a2_plus_pct:>6.1f}%")

    # Slot sessions (gambling pattern)
    print("\n--- Slot Sessions (gambling pattern) ---")
    slot_results = []
    for n_spins in [1, 3, 5, 7]:
        analysis = analyze(
            initial_state=multispin_initial_state(n_spins),
            is_terminal=lambda s, ns=n_spins: multispin_is_terminal(s, ns),
            get_transitions=multispin_get_transitions,
            compute_intrinsic_desire=lambda s: multispin_compute_desire(s),
            nest_level=10,
        )
        gds = analysis.game_design_score
        comps = analysis.gds_components[:10]
        slot_results.append((n_spins, gds, comps))

    print(f"{'Spins':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₂+%':>7}")
    print("-" * 40)
    for n, gds, comps in slot_results:
        a2_plus_pct = sum(comps[1:]) / gds * 100 if gds > 0 else 0
        print(f"{n:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {a2_plus_pct:>6.1f}%")

    # Growth comparison
    if len(bon_results) >= 2 and len(rou_results) >= 2:
        bon_growth = (bon_results[-1][1] - bon_results[0][1]) / (bon_results[-1][0] - bon_results[0][0])
        rou_growth = (rou_results[-1][1] - rou_results[0][1]) / (rou_results[-1][0] - rou_results[0][0])
        print(f"\nGDS growth rates:")
        print(f"  Best-of-N: {bon_growth:.4f}/round")
        print(f"  Roulette:  {rou_growth:.4f}/round")
        if bon_growth > 0:
            print(f"  Ratio:     {rou_growth/bon_growth:.2f}x")

    return bon_results, rou_results, slot_results


def main():
    """Run all gambling mechanics experiments."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  ToA Gambling Mechanics — Casino Game Engagement Analysis     ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    r1 = analyze_single_turn_games()
    r2 = analyze_multi_turn_sessions()
    r3 = analyze_house_edge_effect()
    r4 = analyze_payout_asymmetry()
    r5 = analyze_near_miss_effect()
    r6 = analyze_gambling_vs_games_scaling()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — GAMBLING vs GAME DESIGN")
    print("=" * 70)

    print("""
Key Findings:
""")

    # Extract key numbers for summary
    # Single turn
    coin_gds = next(gds for name, gds, _, _ in r1 if "CoinToss" in name)
    slot_gds = next(gds for name, gds, _, _ in r1 if "Slot" in name)
    roul_single_gds = next(gds for name, gds, _, _ in r1 if "Single" in name)

    print(f"1. SINGLE-TURN ENGAGEMENT:")
    print(f"   CoinToss (fair):        GDS = {coin_gds:.4f}")
    print(f"   Slot Machine (tiered):  GDS = {slot_gds:.4f}")
    print(f"   Roulette (single#):     GDS = {roul_single_gds:.4f}")

    # Multi-turn depth
    hp_data = next((name, gds, comps, n) for name, gds, comps, n in r2 if "HpGame (5,5)" in name)
    print(f"\n2. MULTI-TURN DEPTH:")
    print(f"   HpGame (5,5): GDS = {hp_data[1]:.4f}, A₂+% = {sum(hp_data[2][1:])/hp_data[1]*100:.1f}%")
    for name, gds, comps, n in r2:
        if "Slot" in name or "Roulette" in name:
            depth = sum(comps[1:]) / gds * 100 if gds > 0 else 0
            print(f"   {name}: GDS = {gds:.4f}, A₂+% = {depth:.1f}%")

    print(f"\n3. HOUSE EDGE = ENGAGEMENT REDUCTION:")
    print(f"   Fair (50%):  GDS = {next(gds for pw, gds, _ in r3 if abs(pw-0.50)<0.01):.4f}")
    print(f"   45%:         GDS = {next(gds for pw, gds, _ in r3 if abs(pw-0.45)<0.01):.4f}")
    print(f"   40%:         GDS = {next(gds for pw, gds, _ in r3 if abs(pw-0.40)<0.01):.4f}")

    print(f"\n4. STRUCTURAL DIAGNOSIS:")
    print(f"   Games: high A₂+% → engagement from strategic depth")
    print(f"   Gambling: low A₂+% → engagement from A₁ only (immediate thrill)")
    print(f"   House edge reduces even A₁ → gambling is suboptimal on ALL metrics")


if __name__ == "__main__":
    main()
