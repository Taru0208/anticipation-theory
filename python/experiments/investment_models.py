"""Investment & Trading Models — Applying ToA to Financial Decision-Making.

Why is day trading addictive? Why do people check their portfolio 50x/day?
This experiment applies ToA to financial instruments, placing them on the
gambling → game spectrum.

Models:
1. Index fund (buy & hold): no decisions, pure time-based returns
2. Stock pick: binary outcome per pick (beat market or not)
3. Day trading session: sequential trades with compounding P&L
4. Options trade: asymmetric payoff (limited loss, uncapped gain)
5. Dollar-cost averaging: periodic investment, reducing variance
6. Portfolio rebalancing: periodic decisions to restore target allocation

Key hypothesis: trading is addictive because it hits a sweet spot —
enough randomness for A₁ (immediate excitement like gambling) combined
with enough perceived agency for A₂+ (strategic depth like games).
The "perceived" is crucial — most retail traders don't actually have edge.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze
from toa.game import sanitize_transitions


# ============================================================
# MODEL 1: Index Fund (Buy & Hold)
# ============================================================

class IndexFund:
    """Buy & hold index fund for N periods.

    Each period: market goes up (+10%) with P=0.55, down (-10%) with P=0.45.
    No decisions — just time passing.
    Desire = final portfolio value / initial value (normalized).

    State: (period, value_level) where value_level tracks cumulative returns.
    """

    PERIODS = 5  # number of periods to hold
    P_UP = 0.55
    P_DOWN = 0.45

    @staticmethod
    def initial_state():
        return (0, 0)  # (period, cumulative_gain_level)

    @staticmethod
    def is_terminal(state):
        return state[0] >= IndexFund.PERIODS

    @staticmethod
    def get_transitions(state, config=None):
        period, level = state
        if period >= IndexFund.PERIODS:
            return []
        return sanitize_transitions([
            (IndexFund.P_UP, (period + 1, level + 1)),
            (IndexFund.P_DOWN, (period + 1, level - 1)),
        ])

    @staticmethod
    def compute_intrinsic_desire(state):
        period, level = state
        if period < IndexFund.PERIODS:
            return 0.0
        # Normalize: level ranges from -PERIODS to +PERIODS
        # Map to [0, 1] with 0 at worst, 1 at best
        return (level + IndexFund.PERIODS) / (2 * IndexFund.PERIODS)


# ============================================================
# MODEL 2: Stock Picker
# ============================================================

class StockPicker:
    """Pick stocks over multiple rounds.

    Each round: pick one of 3 stocks (aggressive, moderate, conservative).
    - Aggressive: 40% chance +3, 60% chance -2
    - Moderate: 50% chance +2, 50% chance -1
    - Conservative: 70% chance +1, 30% chance -1

    State: (round, portfolio_value)
    Player has CHOICES — this is what distinguishes it from pure gambling.
    But since we model all choices, the analysis captures the strategic depth.
    """

    ROUNDS = 3
    MAX_VALUE = 10  # cap to keep state space finite

    @staticmethod
    def initial_state():
        return (0, 5)  # start with value 5

    @staticmethod
    def is_terminal(state):
        rnd, val = state
        return rnd >= StockPicker.ROUNDS or val <= 0

    @staticmethod
    def get_transitions(state, config=None):
        rnd, val = state
        if rnd >= StockPicker.ROUNDS or val <= 0:
            return []

        transitions = []
        cap = StockPicker.MAX_VALUE

        # Aggressive pick
        transitions.append((0.40 / 3, (rnd + 1, min(cap, val + 3))))
        transitions.append((0.60 / 3, (rnd + 1, max(0, val - 2))))

        # Moderate pick
        transitions.append((0.50 / 3, (rnd + 1, min(cap, val + 2))))
        transitions.append((0.50 / 3, (rnd + 1, max(0, val - 1))))

        # Conservative pick
        transitions.append((0.70 / 3, (rnd + 1, min(cap, val + 1))))
        transitions.append((0.30 / 3, (rnd + 1, max(0, val - 1))))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        rnd, val = state
        return val / StockPicker.MAX_VALUE


# ============================================================
# MODEL 3: Day Trading Session
# ============================================================

class DayTrader:
    """Day trading session with sequential trades.

    Each trade: choose position size (small/large) and direction is random.
    - Small position: 50% chance +1, 50% chance -1
    - Large position: 50% chance +2, 50% chance -2

    Can also choose to close the session (stop trading).
    State: (trade_number, p&l)
    Bankrupt at p&l <= -5 (margin call).

    Key: the OPTION to stop creates a "double or nothing" dynamic.
    """

    MAX_TRADES = 5
    BUST_THRESHOLD = -5

    @staticmethod
    def initial_state():
        return (0, 0, 1)  # (trade_num, p&l, still_trading)

    @staticmethod
    def is_terminal(state):
        trade, pnl, trading = state
        return trading == 0 or trade >= DayTrader.MAX_TRADES or pnl <= DayTrader.BUST_THRESHOLD

    @staticmethod
    def get_transitions(state, config=None):
        trade, pnl, trading = state
        if trading == 0 or trade >= DayTrader.MAX_TRADES or pnl <= DayTrader.BUST_THRESHOLD:
            return []

        transitions = []

        # Option 1: Close session (stop trading) — 1/3 chance we stop
        transitions.append((1 / 3, (trade, pnl, 0)))

        # Option 2: Small trade — 1/3 chance
        transitions.append((1 / 6, (trade + 1, pnl + 1, 1)))
        transitions.append((1 / 6, (trade + 1, max(DayTrader.BUST_THRESHOLD, pnl - 1), 1)))

        # Option 3: Large trade — 1/3 chance
        transitions.append((1 / 6, (trade + 1, pnl + 2, 1)))
        transitions.append((1 / 6, (trade + 1, max(DayTrader.BUST_THRESHOLD, pnl - 2), 1)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        trade, pnl, trading = state
        # Desire based on final P&L — normalized to [0, 1]
        # Range: -5 to +10 roughly
        return max(0.0, min(1.0, (pnl + 5) / 15))


# ============================================================
# MODEL 4: Options Trade
# ============================================================

class OptionsTrade:
    """Buying a call option — limited downside, uncapped upside.

    Premium paid upfront (lose if wrong). If right, payoff depends on
    how far the underlying moves.

    State: single trade decision.
    Probabilities model underlying movement:
    - 60% expires worthless (lose premium = 1 unit)
    - 25% small profit (gain 1 unit, net 0)
    - 10% moderate profit (gain 3 units, net 2)
    - 4% large profit (gain 8 units, net 7)
    - 1% jackpot (gain 20 units, net 19)

    This is structurally similar to slots but with different payoff ratios.
    """

    @staticmethod
    def initial_state():
        return "open"

    @staticmethod
    def is_terminal(state):
        return state != "open"

    @staticmethod
    def get_transitions(state, config=None):
        if state != "open":
            return []
        return sanitize_transitions([
            (0.60, "worthless"),
            (0.25, "small_profit"),
            (0.10, "moderate_profit"),
            (0.04, "large_profit"),
            (0.01, "jackpot"),
        ])

    @staticmethod
    def compute_intrinsic_desire(state):
        desires = {
            "open": 0.0,
            "worthless": 0.0,
            "small_profit": 0.1,
            "moderate_profit": 0.4,
            "large_profit": 0.7,
            "jackpot": 1.0,
        }
        return desires.get(state, 0.0)


# ============================================================
# MODEL 5: Options Trading Session (multi-round)
# ============================================================

class OptionsSession:
    """Multiple options trades in sequence.

    Each round: buy an option or stop.
    Budget starts at 5 units. Each option costs 1 unit.
    Payoff per option same as OptionsTrade model.

    State: (round, budget, still_trading)
    """

    MAX_ROUNDS = 5

    @staticmethod
    def initial_state():
        return (0, 5, 1)  # round, budget, trading

    @staticmethod
    def is_terminal(state):
        rnd, budget, trading = state
        return trading == 0 or rnd >= OptionsSession.MAX_ROUNDS or budget <= 0

    @staticmethod
    def get_transitions(state, config=None):
        rnd, budget, trading = state
        if trading == 0 or rnd >= OptionsSession.MAX_ROUNDS or budget <= 0:
            return []

        transitions = []
        # 20% chance stop trading this round (modeling the choice to exit)
        transitions.append((0.20, (rnd, budget, 0)))

        # 80% chance buy an option (cost 1 unit)
        remaining = budget - 1  # pay premium
        transitions.append((0.80 * 0.60, (rnd + 1, remaining, 1)))           # worthless
        transitions.append((0.80 * 0.25, (rnd + 1, remaining + 1, 1)))       # small
        transitions.append((0.80 * 0.10, (rnd + 1, remaining + 3, 1)))       # moderate
        transitions.append((0.80 * 0.04, (rnd + 1, remaining + 8, 1)))       # large
        transitions.append((0.80 * 0.01, (rnd + 1, remaining + 20, 1)))      # jackpot

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        rnd, budget, trading = state
        # Normalize budget: started at 5, could go up to 25+ with jackpots
        return max(0.0, min(1.0, budget / 15))


# ============================================================
# MODEL 6: DCA (Dollar-Cost Averaging)
# ============================================================

class DollarCostAverage:
    """Dollar-cost averaging: invest fixed amount each period.

    No decisions — automatic periodic investment.
    Market goes up/down each period.
    State: (period, shares_accumulated, current_price_level)

    Desire = total portfolio value at end.
    """

    PERIODS = 5
    P_UP = 0.55
    P_DOWN = 0.45

    @staticmethod
    def initial_state():
        return (0, 0, 3)  # period, total_shares*100, price_level (start at 3)

    @staticmethod
    def is_terminal(state):
        return state[0] >= DollarCostAverage.PERIODS

    @staticmethod
    def get_transitions(state, config=None):
        period, shares100, price = state
        if period >= DollarCostAverage.PERIODS:
            return []

        # Each period: invest 1 unit, buy shares at current price
        # Shares bought = 100 / price (scaled by 100 to keep integer)
        new_shares = int(100 / max(1, price))

        return sanitize_transitions([
            (DollarCostAverage.P_UP, (period + 1, shares100 + new_shares, min(6, price + 1))),
            (DollarCostAverage.P_DOWN, (period + 1, shares100 + new_shares, max(1, price - 1))),
        ])

    @staticmethod
    def compute_intrinsic_desire(state):
        period, shares100, price = state
        if period < DollarCostAverage.PERIODS:
            return 0.0
        # Portfolio value = shares * current price
        value = (shares100 * price) / 100
        # Normalize: invested 5 units total, value could range 0-15ish
        return max(0.0, min(1.0, value / 10))


# ============================================================
# EXPERIMENTS
# ============================================================

def run_analysis(name, game_cls, nest_level=5):
    """Run ToA analysis on a financial model."""
    analysis = analyze(
        initial_state=game_cls.initial_state(),
        is_terminal=game_cls.is_terminal,
        get_transitions=game_cls.get_transitions,
        compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
        nest_level=nest_level,
    )
    return analysis


def experiment_1_single_trade_comparison():
    """Compare single-decision financial instruments."""
    print("=" * 60)
    print("EXPERIMENT 1: Single-trade GDS comparison")
    print("=" * 60)

    # Single-turn models
    from toa.games.coin_toss import CoinToss

    from experiments.gambling_mechanics import RouletteSingleNumber

    models = [
        ("Roulette (single #)", RouletteSingleNumber),
        ("Options (single)", OptionsTrade),
        ("Fair Coin Flip", CoinToss),
    ]

    for name, cls in models:
        a = run_analysis(name, cls, nest_level=5)
        depth = sum(a.gds_components[1:5]) / a.game_design_score if a.game_design_score > 0 else 0
        print(f"  {name:30s} GDS={a.game_design_score:.4f}  A₁={a.gds_components[0]:.4f}  depth={depth:.1%}")
    print()


def experiment_2_multi_round_comparison():
    """Compare multi-round financial models with games."""
    print("=" * 60)
    print("EXPERIMENT 2: Multi-round models vs games")
    print("=" * 60)

    from toa.games.hpgame import HpGame
    from toa.games.hpgame_rage import HpGameRage

    models = [
        ("Index Fund (5 periods)", IndexFund),
        ("DCA (5 periods)", DollarCostAverage),
        ("Stock Picker (3 rounds)", StockPicker),
        ("Day Trader (5 trades)", DayTrader),
        ("Options Session (5 rounds)", OptionsSession),
        ("HpGame (5,5)", HpGame),
        ("HpGame Rage (5,5)", HpGameRage),
    ]

    results = []
    for name, cls in models:
        a = run_analysis(name, cls, nest_level=5)
        comps = a.gds_components[:5]
        depth = sum(comps[1:]) / a.game_design_score if a.game_design_score > 0 else 0
        results.append((name, a.game_design_score, comps, depth, len(a.state_nodes)))
        print(f"  {name:30s} GDS={a.game_design_score:.4f}  "
              f"[{', '.join(f'{c:.3f}' for c in comps)}]  "
              f"depth={depth:.1%}  states={len(a.state_nodes)}")

    print()
    return results


def experiment_3_trading_vs_gambling_spectrum():
    """Place financial instruments on gambling-game spectrum."""
    print("=" * 60)
    print("EXPERIMENT 3: Gambling → Trading → Game Spectrum")
    print("=" * 60)

    # Import gambling models
    from experiments.gambling_mechanics import RouletteSingleNumber, RouletteRedBlack, SlotMachine
    from toa.games.coin_toss import CoinToss
    from toa.games.hpgame import HpGame

    spectrum = [
        ("Roulette single #", RouletteSingleNumber),
        ("Slot Machine", SlotMachine),
        ("Roulette red/black", RouletteRedBlack),
        ("Options (single)", OptionsTrade),
        ("Fair Coin", CoinToss),
        ("Index Fund (5yr)", IndexFund),
        ("DCA (5yr)", DollarCostAverage),
        ("Stock Picker", StockPicker),
        ("Day Trader", DayTrader),
        ("Options Session", OptionsSession),
        ("HpGame", HpGame),
    ]

    print(f"  {'Model':30s} {'GDS':>7s}  {'A₁':>6s}  {'Depth':>6s}  Category")
    print(f"  {'-'*30} {'-'*7}  {'-'*6}  {'-'*6}  {'-'*12}")

    for name, cls in spectrum:
        a = run_analysis(name, cls, nest_level=5)
        comps = a.gds_components[:5]
        depth = sum(comps[1:]) / a.game_design_score if a.game_design_score > 0 else 0

        # Categorize by depth ratio, not just GDS
        if depth < 0.05:
            cat = "pure chance"
        elif depth < 0.20:
            cat = "low depth"
        elif depth < 0.50:
            cat = "moderate depth"
        else:
            cat = "high depth"

        print(f"  {name:30s} {a.game_design_score:7.4f}  {comps[0]:6.4f}  {depth:5.1%}  {cat}")

    print()


def experiment_4_agency_effect():
    """How does player agency affect GDS?

    Compare models with and without choices.
    Index Fund (no choices) vs Stock Picker (choices).
    """
    print("=" * 60)
    print("EXPERIMENT 4: Agency effect on GDS")
    print("=" * 60)

    a_index = run_analysis("Index Fund", IndexFund, nest_level=5)
    a_stock = run_analysis("Stock Picker", StockPicker, nest_level=5)
    a_day = run_analysis("Day Trader", DayTrader, nest_level=5)

    print(f"  No choices  (Index Fund):   GDS={a_index.game_design_score:.4f}")
    print(f"  3 choices   (Stock Picker): GDS={a_stock.game_design_score:.4f}")
    print(f"  2 choices+stop (Day Trader): GDS={a_day.game_design_score:.4f}")

    if a_stock.game_design_score > a_index.game_design_score:
        boost = (a_stock.game_design_score - a_index.game_design_score) / a_index.game_design_score
        print(f"\n  Agency boost: +{boost:.1%} (Stock Picker vs Index Fund)")
    if a_day.game_design_score > a_index.game_design_score:
        boost = (a_day.game_design_score - a_index.game_design_score) / a_index.game_design_score
        print(f"  Agency boost: +{boost:.1%} (Day Trader vs Index Fund)")

    print()


def experiment_5_stopping_rule():
    """The effect of "option to stop" on engagement.

    Day trading lets you stop anytime. Does this increase GDS?
    Compare: forced 5 trades vs optional stop.
    """
    print("=" * 60)
    print("EXPERIMENT 5: Stopping rule effect")
    print("=" * 60)

    # Day trader has stop option built in
    a_stop = run_analysis("Day Trader (with stop)", DayTrader, nest_level=5)

    # Create forced version — no stop option
    class ForcedTrader:
        MAX_TRADES = 5
        BUST_THRESHOLD = -5

        @staticmethod
        def initial_state():
            return (0, 0)

        @staticmethod
        def is_terminal(state):
            trade, pnl = state
            return trade >= ForcedTrader.MAX_TRADES or pnl <= ForcedTrader.BUST_THRESHOLD

        @staticmethod
        def get_transitions(state, config=None):
            trade, pnl = state
            if trade >= ForcedTrader.MAX_TRADES or pnl <= ForcedTrader.BUST_THRESHOLD:
                return []
            return sanitize_transitions([
                (0.25, (trade + 1, pnl + 1)),  # small win
                (0.25, (trade + 1, max(ForcedTrader.BUST_THRESHOLD, pnl - 1))),  # small loss
                (0.25, (trade + 1, pnl + 2)),  # big win
                (0.25, (trade + 1, max(ForcedTrader.BUST_THRESHOLD, pnl - 2))),  # big loss
            ])

        @staticmethod
        def compute_intrinsic_desire(state):
            trade, pnl = state
            return max(0.0, min(1.0, (pnl + 5) / 15))

    a_forced = run_analysis("Forced Trader (no stop)", ForcedTrader, nest_level=5)

    print(f"  With stop option:    GDS={a_stop.game_design_score:.4f}")
    print(f"  Without stop option: GDS={a_forced.game_design_score:.4f}")

    if a_stop.game_design_score != a_forced.game_design_score:
        diff = a_stop.game_design_score - a_forced.game_design_score
        print(f"  Stop option effect: {'+' if diff > 0 else ''}{diff:.4f} ({diff/a_forced.game_design_score:+.1%})")
    print()


def experiment_6_addiction_vs_engagement():
    """Why is trading addictive? Structural comparison.

    Hypothesis: Trading sits at a dangerous sweet spot where:
    1. A₁ is high enough to feel exciting (like gambling)
    2. A₂+ are non-zero, creating an illusion of skill/depth
    3. But the "depth" is largely illusory for retail traders
    """
    print("=" * 60)
    print("EXPERIMENT 6: Addiction structure analysis")
    print("=" * 60)

    from experiments.gambling_mechanics import SlotMachine, RouletteRedBlack
    from toa.games.hpgame import HpGame

    models = [
        ("Slot Machine", SlotMachine, "gambling"),
        ("Roulette R/B", RouletteRedBlack, "gambling"),
        ("Options (single)", OptionsTrade, "investment"),
        ("Day Trader", DayTrader, "trading"),
        ("Stock Picker", StockPicker, "trading"),
        ("Options Session", OptionsSession, "trading"),
        ("HpGame", HpGame, "game"),
    ]

    print(f"  {'Model':25s} {'GDS':>6s}  {'A₁%':>5s}  {'A₂+%':>5s}  {'Cat':>10s}  Addiction risk")
    print(f"  {'-'*25} {'-'*6}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*20}")

    for name, cls, cat in models:
        a = run_analysis(name, cls, nest_level=5)
        comps = a.gds_components[:5]
        gds = a.game_design_score
        a1_pct = comps[0] / gds * 100 if gds > 0 else 0
        a2p_pct = sum(comps[1:]) / gds * 100 if gds > 0 else 0

        # Addiction risk heuristic:
        # High GDS + high A₁% = most addictive (exciting + feels skillful)
        # Low GDS = boring (not addictive)
        # High GDS + low A₁% = engaging but satisfying (healthy)
        if gds < 0.15:
            risk = "low (boring)"
        elif a1_pct > 60:
            risk = "moderate (exciting)"
        elif gds > 0.3 and a1_pct < 50:
            risk = "healthy engagement"
        else:
            risk = "HIGH (illusion of skill)"

        print(f"  {name:25s} {gds:6.3f}  {a1_pct:4.0f}%  {a2p_pct:4.0f}%  {cat:>10s}  {risk}")

    print()
    print("  Key insight: Trading models show moderate-high GDS with")
    print("  A₂+ contribution, creating an 'illusion of skill' that")
    print("  makes them more addictive than pure gambling.")
    print()


if __name__ == "__main__":
    experiment_1_single_trade_comparison()
    experiment_2_multi_round_comparison()
    experiment_3_trading_vs_gambling_spectrum()
    experiment_4_agency_effect()
    experiment_5_stopping_rule()
    experiment_6_addiction_vs_engagement()
