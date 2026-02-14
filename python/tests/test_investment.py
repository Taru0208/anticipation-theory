"""Tests for investment/trading ToA models."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze
from experiments.investment_models import (
    IndexFund, StockPicker, DayTrader,
    OptionsTrade, OptionsSession, DollarCostAverage,
)


def run(cls, nest=5):
    return analyze(
        initial_state=cls.initial_state(),
        is_terminal=cls.is_terminal,
        get_transitions=cls.get_transitions,
        compute_intrinsic_desire=cls.compute_intrinsic_desire,
        nest_level=nest,
    )


# ============================================================
# Model correctness
# ============================================================

class TestIndexFund:
    def test_initial_state(self):
        assert IndexFund.initial_state() == (0, 0)

    def test_terminal_at_period_5(self):
        assert IndexFund.is_terminal((5, 0))
        assert IndexFund.is_terminal((5, 3))
        assert not IndexFund.is_terminal((2, 0))

    def test_transitions_sum_to_one(self):
        trans = IndexFund.get_transitions((0, 0))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_two_outcomes_per_step(self):
        trans = IndexFund.get_transitions((0, 0))
        assert len(trans) == 2

    def test_up_probability_55(self):
        trans = IndexFund.get_transitions((0, 0))
        # Should have 0.55 up and 0.45 down
        probs = sorted([p for p, _ in trans])
        assert abs(probs[0] - 0.45) < 1e-10
        assert abs(probs[1] - 0.55) < 1e-10

    def test_desire_normalized(self):
        # Best case: all up, level = +5
        assert abs(IndexFund.compute_intrinsic_desire((5, 5)) - 1.0) < 1e-10
        # Worst case: all down, level = -5
        assert abs(IndexFund.compute_intrinsic_desire((5, -5)) - 0.0) < 1e-10
        # Neutral
        assert abs(IndexFund.compute_intrinsic_desire((5, 0)) - 0.5) < 1e-10

    def test_state_count(self):
        a = run(IndexFund)
        # 5 periods, levels from -5 to +5 (11 values at terminal)
        # Plus intermediate states
        assert len(a.state_nodes) > 10


class TestStockPicker:
    def test_initial_state(self):
        assert StockPicker.initial_state() == (0, 5)

    def test_terminal_at_round_3(self):
        assert StockPicker.is_terminal((3, 5))
        assert not StockPicker.is_terminal((1, 5))

    def test_terminal_at_zero_value(self):
        assert StockPicker.is_terminal((1, 0))

    def test_transitions_sum_to_one(self):
        trans = StockPicker.get_transitions((0, 5))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_six_outcomes(self):
        # 3 choices × 2 outcomes = 6 transitions (before merging)
        trans = StockPicker.get_transitions((0, 5))
        assert len(trans) >= 4  # some may merge

    def test_choices_create_depth(self):
        a = run(StockPicker)
        # Should have non-zero depth ratio (choices matter)
        comps = a.gds_components[:5]
        assert comps[0] > 0  # A₁ exists
        assert a.game_design_score > 0.1


class TestDayTrader:
    def test_initial_state(self):
        assert DayTrader.initial_state() == (0, 0, 1)

    def test_terminal_when_stopped(self):
        assert DayTrader.is_terminal((0, 0, 0))

    def test_terminal_at_max_trades(self):
        assert DayTrader.is_terminal((5, 0, 1))

    def test_terminal_at_bust(self):
        assert DayTrader.is_terminal((1, -5, 1))

    def test_transitions_sum_to_one(self):
        trans = DayTrader.get_transitions((0, 0, 1))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_stop_option_exists(self):
        trans = DayTrader.get_transitions((0, 0, 1))
        # One transition should lead to stopped state
        stopped = [s for p, s in trans if s[2] == 0]
        assert len(stopped) >= 1

    def test_high_gds(self):
        a = run(DayTrader)
        # Day trading should have high GDS due to choices + stop option
        assert a.game_design_score > 0.5


class TestOptionsTrade:
    def test_initial_state(self):
        assert OptionsTrade.initial_state() == "open"

    def test_terminal_states(self):
        assert OptionsTrade.is_terminal("worthless")
        assert OptionsTrade.is_terminal("jackpot")
        assert not OptionsTrade.is_terminal("open")

    def test_probabilities_sum_to_one(self):
        trans = OptionsTrade.get_transitions("open")
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_five_outcomes(self):
        trans = OptionsTrade.get_transitions("open")
        assert len(trans) == 5

    def test_single_turn_zero_depth(self):
        a = run(OptionsTrade)
        # Single turn = zero depth
        comps = a.gds_components[:5]
        assert abs(sum(comps[1:])) < 1e-10

    def test_gds_between_slot_and_coin(self):
        a = run(OptionsTrade)
        # Options should be between slot machine (0.10) and fair coin (0.50)
        assert 0.10 < a.game_design_score < 0.50


class TestOptionsSession:
    def test_initial_state(self):
        assert OptionsSession.initial_state() == (0, 5, 1)

    def test_terminal_states(self):
        assert OptionsSession.is_terminal((0, 0, 1))  # no budget
        assert OptionsSession.is_terminal((5, 5, 1))  # max rounds
        assert OptionsSession.is_terminal((0, 5, 0))  # stopped

    def test_transitions_sum_to_one(self):
        trans = OptionsSession.get_transitions((0, 5, 1))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_has_depth(self):
        a = run(OptionsSession)
        comps = a.gds_components[:5]
        depth_ratio = sum(comps[1:]) / a.game_design_score if a.game_design_score > 0 else 0
        assert depth_ratio > 0.3  # meaningful depth


class TestDollarCostAverage:
    def test_initial_state(self):
        assert DollarCostAverage.initial_state() == (0, 0, 3)

    def test_terminal_at_period_5(self):
        assert DollarCostAverage.is_terminal((5, 100, 3))
        assert not DollarCostAverage.is_terminal((3, 100, 3))

    def test_transitions_sum_to_one(self):
        trans = DollarCostAverage.get_transitions((0, 0, 3))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_shares_accumulate(self):
        trans = DollarCostAverage.get_transitions((0, 0, 3))
        for p, (period, shares, price) in trans:
            assert shares > 0  # bought some shares
            assert period == 1


# ============================================================
# Key findings
# ============================================================

class TestKeyFindings:
    """Verify the main research findings."""

    def test_index_fund_lowest_gds(self):
        """Index fund (no decisions) should have lowest GDS among multi-round."""
        a_index = run(IndexFund)
        a_stock = run(StockPicker)
        a_day = run(DayTrader)
        assert a_index.game_design_score < a_stock.game_design_score
        assert a_index.game_design_score < a_day.game_design_score

    def test_agency_increases_gds(self):
        """Adding player choices increases GDS (agency effect)."""
        a_index = run(IndexFund)  # no choices
        a_stock = run(StockPicker)  # choices
        boost = (a_stock.game_design_score - a_index.game_design_score) / a_index.game_design_score
        assert boost > 1.0  # at least 100% improvement

    def test_stop_option_increases_gds(self):
        """The option to stop trading dramatically increases GDS."""
        from toa.game import sanitize_transitions

        class ForcedTrader:
            MAX_TRADES = 5
            BUST_THRESHOLD = -5

            @staticmethod
            def initial_state():
                return (0, 0)

            @staticmethod
            def is_terminal(state):
                return state[0] >= 5 or state[1] <= -5

            @staticmethod
            def get_transitions(state, config=None):
                trade, pnl = state
                if trade >= 5 or pnl <= -5:
                    return []
                return sanitize_transitions([
                    (0.25, (trade + 1, pnl + 1)),
                    (0.25, (trade + 1, max(-5, pnl - 1))),
                    (0.25, (trade + 1, pnl + 2)),
                    (0.25, (trade + 1, max(-5, pnl - 2))),
                ])

            @staticmethod
            def compute_intrinsic_desire(state):
                return max(0.0, min(1.0, (state[1] + 5) / 15))

        a_stop = run(DayTrader)
        a_forced = run(ForcedTrader)
        assert a_stop.game_design_score > a_forced.game_design_score * 1.5

    def test_day_trading_higher_than_hpgame(self):
        """Day trading GDS exceeds HpGame — explains addiction."""
        from toa.games.hpgame import HpGame
        a_day = run(DayTrader)
        a_hp = run(HpGame)
        assert a_day.game_design_score > a_hp.game_design_score

    def test_options_asymmetry_like_slots(self):
        """Single options trade has GDS similar to asymmetric gambling."""
        a_opt = run(OptionsTrade)
        # GDS should be low (< 0.25) due to payoff asymmetry
        assert a_opt.game_design_score < 0.25

    def test_trading_has_depth(self):
        """Trading models show non-trivial depth ratio."""
        a_day = run(DayTrader)
        comps = a_day.gds_components[:5]
        depth = sum(comps[1:]) / a_day.game_design_score
        assert depth > 0.4  # at least 40% depth

    def test_dca_more_engaging_than_index(self):
        """DCA (variance reduction) is slightly more engaging than index."""
        a_index = run(IndexFund)
        a_dca = run(DollarCostAverage)
        assert a_dca.game_design_score > a_index.game_design_score

    def test_options_session_high_gds(self):
        """Options session achieves high GDS through multi-round + stopping."""
        a_opt = run(OptionsSession)
        assert a_opt.game_design_score > 0.7

    def test_addiction_spectrum(self):
        """GDS ordering: Index < Options < StockPicker < HpGame < DayTrader."""
        a_index = run(IndexFund)
        a_opt = run(OptionsTrade)
        a_stock = run(StockPicker)
        a_day = run(DayTrader)

        assert a_index.game_design_score < a_opt.game_design_score
        assert a_opt.game_design_score < a_stock.game_design_score
        assert a_stock.game_design_score < a_day.game_design_score
