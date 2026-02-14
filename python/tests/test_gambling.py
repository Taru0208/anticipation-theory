"""Tests for gambling mechanics models and key findings."""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze
from experiments.gambling_mechanics import (
    RouletteSingleNumber,
    RouletteRedBlack,
    SlotMachine,
    Blackjack,
    multispin_initial_state,
    multispin_is_terminal,
    multispin_get_transitions,
    multispin_compute_desire,
    roulette_session_initial,
    roulette_session_is_terminal,
    roulette_session_transitions,
    roulette_session_desire,
)


# ============================================================
# Model Validation
# ============================================================

class TestRouletteModels:
    def test_roulette_single_probabilities_sum_to_1(self):
        trans = RouletteSingleNumber.get_transitions("initial")
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_roulette_single_win_probability(self):
        trans = RouletteSingleNumber.get_transitions("initial")
        p_win = next(p for p, s in trans if s == "win")
        assert abs(p_win - 1.0 / 37.0) < 1e-10

    def test_roulette_redblack_probabilities(self):
        trans = RouletteRedBlack.get_transitions("initial")
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10
        p_win = next(p for p, s in trans if s == "win")
        assert abs(p_win - 18.0 / 37.0) < 1e-10

    def test_roulette_terminal_states(self):
        assert not RouletteSingleNumber.is_terminal("initial")
        assert RouletteSingleNumber.is_terminal("win")
        assert RouletteSingleNumber.is_terminal("lose")

    def test_roulette_desire(self):
        assert RouletteSingleNumber.compute_intrinsic_desire("win") == 1.0
        assert RouletteSingleNumber.compute_intrinsic_desire("lose") == 0.0


class TestSlotMachine:
    def test_slot_probabilities_sum_to_1(self):
        trans = SlotMachine.get_transitions("spinning")
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_slot_terminal_states(self):
        assert not SlotMachine.is_terminal("spinning")
        assert SlotMachine.is_terminal("jackpot")
        assert SlotMachine.is_terminal("loss")

    def test_slot_desire_ordering(self):
        d_jackpot = SlotMachine.compute_intrinsic_desire("jackpot")
        d_big = SlotMachine.compute_intrinsic_desire("big_win")
        d_small = SlotMachine.compute_intrinsic_desire("small_win")
        d_miss = SlotMachine.compute_intrinsic_desire("near_miss")
        d_loss = SlotMachine.compute_intrinsic_desire("loss")
        assert d_jackpot > d_big > d_small > d_miss
        assert d_miss == d_loss == 0.0


class TestBlackjack:
    def test_initial_state(self):
        state = Blackjack.initial_state()
        assert state == (12, 7, True)
        assert not Blackjack.is_terminal(state)

    def test_bust_is_terminal(self):
        assert Blackjack.is_terminal((22, 7, False))
        assert Blackjack.compute_intrinsic_desire((22, 7, False)) == 0.0

    def test_standing_is_terminal(self):
        assert Blackjack.is_terminal((18, 7, False))

    def test_transitions_include_hit_and_stand(self):
        trans = Blackjack.get_transitions((12, 7, True))
        assert len(trans) > 0
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-6

    def test_high_total_better_desire(self):
        d20 = Blackjack.compute_intrinsic_desire((20, 7, False))
        d14 = Blackjack.compute_intrinsic_desire((14, 7, False))
        assert d20 > d14


class TestMultiSpinSlots:
    def test_initial_state(self):
        state = multispin_initial_state(5)
        assert state == (5, 5)

    def test_terminal_when_broke(self):
        assert multispin_is_terminal((0, 3), 5)

    def test_terminal_when_done(self):
        assert multispin_is_terminal((5, 0), 5)

    def test_not_terminal_mid_session(self):
        assert not multispin_is_terminal((5, 3), 5)

    def test_transitions_valid(self):
        trans = multispin_get_transitions((5, 3))
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-6
        for p, s in trans:
            assert p > 0
            balance, remaining = s
            assert remaining == 2  # one less
            assert balance >= 0


class TestRouletteSession:
    def test_initial_state(self):
        state = roulette_session_initial(5)
        assert state == (10, 5)

    def test_terminal_when_done(self):
        assert roulette_session_is_terminal((5, 0), 5)

    def test_terminal_when_broke(self):
        assert roulette_session_is_terminal((0, 3), 5)

    def test_desire_profit(self):
        # Profit state
        d = roulette_session_desire((12, 0))
        assert d > 0

    def test_desire_broke(self):
        d = roulette_session_desire((0, 0))
        assert d == 0.0


# ============================================================
# Key Findings Validation
# ============================================================

class TestKeyFindings:
    """Validate the core research findings with assertions."""

    def _analyze_game(self, game_cls, nest=5):
        return analyze(
            initial_state=game_cls.initial_state(),
            is_terminal=game_cls.is_terminal,
            get_transitions=game_cls.get_transitions,
            compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
            nest_level=nest,
        )

    def test_fair_coin_has_highest_single_turn_gds(self):
        """Fair 50/50 game maximizes single-turn GDS."""
        from toa.games.coin_toss import CoinToss
        coin = self._analyze_game(CoinToss)
        redblack = self._analyze_game(RouletteRedBlack)
        single = self._analyze_game(RouletteSingleNumber)
        slot = self._analyze_game(SlotMachine)

        assert coin.game_design_score >= redblack.game_design_score
        assert coin.game_design_score > single.game_design_score
        assert coin.game_design_score > slot.game_design_score

    def test_asymmetric_payouts_reduce_gds(self):
        """More skewed probability distributions → lower GDS."""
        redblack = self._analyze_game(RouletteRedBlack)
        single = self._analyze_game(RouletteSingleNumber)
        # Red/Black (near 50/50) has much higher GDS than single number (2.7%)
        assert redblack.game_design_score > single.game_design_score * 2

    def test_slot_lower_than_coin(self):
        """Slot machine with tiered payouts has lower GDS than simple coin flip."""
        from toa.games.coin_toss import CoinToss
        coin = self._analyze_game(CoinToss)
        slot = self._analyze_game(SlotMachine)
        assert coin.game_design_score > slot.game_design_score * 3

    def test_house_edge_reduces_engagement(self):
        """Any deviation from 50/50 reduces GDS for single-turn binary games."""
        # Fair game
        fair = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.5, "w"), (0.5, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        )
        # House edge game (45%)
        house = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.45, "w"), (0.55, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        )
        assert fair.game_design_score > house.game_design_score

    def test_house_edge_effect_is_small(self):
        """House edge of 5% reduces GDS by less than 1%."""
        fair = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.5, "w"), (0.5, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        )
        edge = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.45, "w"), (0.55, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        )
        reduction = (fair.game_design_score - edge.game_design_score) / fair.game_design_score
        assert reduction < 0.01  # less than 1% reduction

    def test_blackjack_highest_gambling_gds(self):
        """Blackjack (with decisions) has higher GDS than pure-chance gambling."""
        bj = self._analyze_game(Blackjack, nest=10)

        # Compare with 5-round roulette
        rou = analyze(
            initial_state=roulette_session_initial(5),
            is_terminal=lambda s: roulette_session_is_terminal(s, 5),
            get_transitions=roulette_session_transitions,
            compute_intrinsic_desire=roulette_session_desire,
            nest_level=10,
        )
        assert bj.game_design_score > rou.game_design_score

    def test_hpgame_beats_all_gambling(self):
        """Well-designed game (HpGame) has higher GDS than any gambling model."""
        from toa.games.hpgame import HpGame
        hp = analyze(
            initial_state=HpGame.initial_state(),
            is_terminal=HpGame.is_terminal,
            get_transitions=HpGame.get_transitions,
            compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
            nest_level=10,
        )

        # Slots (5 spins)
        slots = analyze(
            initial_state=multispin_initial_state(5),
            is_terminal=lambda s: multispin_is_terminal(s, 5),
            get_transitions=multispin_get_transitions,
            compute_intrinsic_desire=multispin_compute_desire,
            nest_level=10,
        )

        # Roulette (5 rounds)
        rou = analyze(
            initial_state=roulette_session_initial(5),
            is_terminal=lambda s: roulette_session_is_terminal(s, 5),
            get_transitions=roulette_session_transitions,
            compute_intrinsic_desire=roulette_session_desire,
            nest_level=10,
        )

        assert hp.game_design_score > slots.game_design_score
        assert hp.game_design_score > rou.game_design_score

    def test_multi_turn_gambling_grows_with_depth(self):
        """Multi-turn gambling sessions still grow with depth (like games)."""
        gds_values = []
        for n in [3, 5, 7, 10]:
            analysis = analyze(
                initial_state=roulette_session_initial(n),
                is_terminal=lambda s, nr=n: roulette_session_is_terminal(s, nr),
                get_transitions=roulette_session_transitions,
                compute_intrinsic_desire=roulette_session_desire,
                nest_level=10,
            )
            gds_values.append(analysis.game_design_score)

        # GDS should increase with more rounds
        for i in range(len(gds_values) - 1):
            assert gds_values[i + 1] > gds_values[i]

    def test_gds_symmetric_around_50pct(self):
        """GDS is symmetric: P(win)=40% has same GDS as P(win)=60%."""
        gds_40 = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.4, "w"), (0.6, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        ).game_design_score

        gds_60 = analyze(
            initial_state="s", is_terminal=lambda s: s != "s",
            get_transitions=lambda s, _: [(0.6, "w"), (0.4, "l")] if s == "s" else [],
            compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
            nest_level=5,
        ).game_design_score

        assert abs(gds_40 - gds_60) < 1e-10

    def test_single_turn_games_have_zero_higher_components(self):
        """Single-turn games have A₂=A₃=...=0 (no depth to analyze)."""
        for game_cls in [RouletteSingleNumber, RouletteRedBlack, SlotMachine]:
            analysis = self._analyze_game(game_cls)
            for i in range(1, 5):
                assert abs(analysis.gds_components[i]) < 1e-10


# ============================================================
# GDS Formula Verification
# ============================================================

class TestGDSFormulas:
    def test_binary_game_a1_formula(self):
        """For binary game with P(win)=p: A₁ = sqrt(p*(1-p))."""
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            analysis = analyze(
                initial_state="s", is_terminal=lambda s: s != "s",
                get_transitions=lambda s, _, pw=p: [(pw, "w"), (1 - pw, "l")] if s == "s" else [],
                compute_intrinsic_desire=lambda s: 1.0 if s == "w" else 0.0,
                nest_level=5,
            )
            expected_a1 = math.sqrt(p * (1 - p))
            actual_a1 = analysis.gds_components[0]
            assert abs(actual_a1 - expected_a1) < 1e-10, f"p={p}: {actual_a1} != {expected_a1}"

    def test_roulette_single_a1_exact(self):
        """Roulette single number: A₁ = sqrt(1/37 * 36/37)."""
        analysis = analyze(
            initial_state=RouletteSingleNumber.initial_state(),
            is_terminal=RouletteSingleNumber.is_terminal,
            get_transitions=RouletteSingleNumber.get_transitions,
            compute_intrinsic_desire=RouletteSingleNumber.compute_intrinsic_desire,
            nest_level=5,
        )
        expected = math.sqrt(1.0 / 37.0 * 36.0 / 37.0)
        assert abs(analysis.gds_components[0] - expected) < 1e-10
