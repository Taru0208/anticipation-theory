"""Tests for the core analysis engine, verified against C++ reference implementation."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.coin_toss import CoinToss
from toa.games.rps import RPS
from toa.games.hpgame import HpGame
from toa.games.goldgame import GoldGame


def approx(a, b, tolerance=0.01):
    """Check approximate equality."""
    return abs(a - b) < tolerance


# ── Coin Toss ──────────────────────────────────────────────


def test_coin_toss_a1():
    """Coin toss should achieve A₁ = 0.5, the theoretical maximum for binary games."""
    result = analyze(
        initial_state=CoinToss.initial_state(),
        is_terminal=CoinToss.is_terminal,
        get_transitions=CoinToss.get_transitions,
        compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
    )
    initial = result.state_nodes["initial"]
    assert approx(initial.a[0], 0.5), f"A₁ = {initial.a[0]}, expected 0.5"


def test_coin_toss_terminal_states():
    """Terminal states should have zero anticipation."""
    result = analyze(
        initial_state=CoinToss.initial_state(),
        is_terminal=CoinToss.is_terminal,
        get_transitions=CoinToss.get_transitions,
        compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
    )
    assert result.state_nodes["win"].a[0] == 0.0
    assert result.state_nodes["loss"].a[0] == 0.0


def test_coin_toss_d_global():
    """D_global for initial state should be 0.5 (expected value of winning)."""
    result = analyze(
        initial_state=CoinToss.initial_state(),
        is_terminal=CoinToss.is_terminal,
        get_transitions=CoinToss.get_transitions,
        compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
    )
    assert approx(result.state_nodes["initial"].d_global, 0.5)


def test_coin_toss_gds():
    """GDS should equal A₁ for a single-turn game."""
    result = analyze(
        initial_state=CoinToss.initial_state(),
        is_terminal=CoinToss.is_terminal,
        get_transitions=CoinToss.get_transitions,
        compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
    )
    assert approx(result.game_design_score, 0.5)


# ── Rock Paper Scissors ────────────────────────────────────


def test_rps_a1():
    """RPS should achieve A₁ ≈ 0.471 (94.2% of theoretical max 0.5)."""
    result = analyze(
        initial_state=RPS.initial_state(),
        is_terminal=RPS.is_terminal,
        get_transitions=RPS.get_transitions,
        compute_intrinsic_desire=RPS.compute_intrinsic_desire,
    )
    initial = result.state_nodes["initial"]
    assert approx(initial.a[0], 0.471, tolerance=0.005), f"A₁ = {initial.a[0]}"


def test_rps_higher_components_zero():
    """Single-turn game should have zero A₂+ components."""
    result = analyze(
        initial_state=RPS.initial_state(),
        is_terminal=RPS.is_terminal,
        get_transitions=RPS.get_transitions,
        compute_intrinsic_desire=RPS.compute_intrinsic_desire,
        nest_level=5,
    )
    initial = result.state_nodes["initial"]
    for i in range(1, 5):
        assert initial.a[i] == 0.0, f"A_{i+1} = {initial.a[i]}, expected 0"


def test_rps_gds():
    """RPS GDS should match A₁ for single-turn game."""
    result = analyze(
        initial_state=RPS.initial_state(),
        is_terminal=RPS.is_terminal,
        get_transitions=RPS.get_transitions,
        compute_intrinsic_desire=RPS.compute_intrinsic_desire,
    )
    assert approx(result.game_design_score, 0.471, tolerance=0.005)


# ── HpGame ─────────────────────────────────────────────────


def test_hpgame_gds():
    """HpGame GDS ≈ 0.430 with 5 anticipation components (from C++ reference)."""
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    assert approx(result.game_design_score, 0.430, tolerance=0.02), \
        f"GDS = {result.game_design_score}, expected ~0.430"


def test_hpgame_initial_d_global():
    """Initial state D_global should reflect overall win probability."""
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    initial = result.state_nodes[(5, 5)]
    # Symmetric game: D_global should be close to the win probability
    assert 0.0 < initial.d_global < 1.0, f"D_global = {initial.d_global}"


def test_hpgame_most_engaging_low_hp():
    """States with both low HP should be more engaging than high HP states.

    From C++ reference: HP(1,1) has A₁ ≈ 0.47, close to coin toss maximum.
    """
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    a_1_1 = result.state_nodes[(1, 1)].sum_a()
    a_5_5 = result.state_nodes[(5, 5)].sum_a()
    assert a_1_1 > a_5_5, f"(1,1) sum_A={a_1_1} should be > (5,5) sum_A={a_5_5}"


def test_hpgame_win_state_zero_anticipation():
    """Win/loss states should have zero anticipation (terminal)."""
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    # P1 wins (P2 has 0 HP)
    for hp1 in range(1, 6):
        node = result.state_nodes.get((hp1, 0))
        if node:
            assert node.sum_a() == 0.0, f"Terminal state ({hp1},0) has non-zero A"


def test_hpgame_state_count():
    """HpGame should enumerate all reachable states correctly."""
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=1,
    )
    # States: all (i,j) where 0<=i<=5, 0<=j<=5, reachable from (5,5)
    # Minus (0,0) which is reachable via both-attack chain
    assert len(result.states) > 20, f"Only {len(result.states)} states found"


def test_hpgame_higher_components_exist():
    """Multi-turn game should have non-zero A₂ components."""
    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    assert result.gds_components[1] > 0, "A₂ GDS component should be > 0"


# ── GoldGame ───────────────────────────────────────────────


def test_goldgame_basic():
    """GoldGame should produce a valid analysis with positive GDS."""
    config = GoldGame.Config(max_turns=5)

    def is_terminal(state):
        return state[2] >= config.max_turns

    def get_transitions(state, cfg=None):
        return GoldGame.get_transitions(state, config)

    def compute_desire(state):
        if state[2] < config.max_turns:
            return 0.0
        return 1.0 if state[0] > state[1] else 0.0

    result = analyze(
        initial_state=(1000, 1000, 0),
        is_terminal=is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=compute_desire,
        nest_level=5,
    )
    assert result.game_design_score > 0, f"GDS = {result.game_design_score}"


def test_goldgame_gds_increases_with_turns():
    """GDS should increase with more turns — supports Unbound Conjecture."""
    scores = []
    for max_turns in [3, 5, 7]:
        config = GoldGame.Config(max_turns=max_turns)

        def is_terminal(state, mt=max_turns):
            return state[2] >= mt

        def get_transitions(state, cfg=None, c=config):
            return GoldGame.get_transitions(state, c)

        def compute_desire(state, mt=max_turns):
            if state[2] < mt:
                return 0.0
            return 1.0 if state[0] > state[1] else 0.0

        result = analyze(
            initial_state=(1000, 1000, 0),
            is_terminal=is_terminal,
            get_transitions=get_transitions,
            compute_intrinsic_desire=compute_desire,
            nest_level=5,
        )
        scores.append(result.game_design_score)

    assert scores[1] > scores[0], f"GDS should increase: {scores}"
    assert scores[2] > scores[1], f"GDS should increase: {scores}"


# ── Edge cases ─────────────────────────────────────────────


def test_probability_conservation():
    """All analyses should conserve probability (total reach = 1.0)."""
    # This is checked internally by the engine, but verify no exception is raised
    for game_cls in [CoinToss, RPS, HpGame]:
        analyze(
            initial_state=game_cls.initial_state(),
            is_terminal=game_cls.is_terminal,
            get_transitions=game_cls.get_transitions,
            compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
            nest_level=5,
        )


def test_nest_level_limit():
    """Should raise ValueError for excessive nest level."""
    try:
        analyze(
            initial_state=CoinToss.initial_state(),
            is_terminal=CoinToss.is_terminal,
            get_transitions=CoinToss.get_transitions,
            compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
            nest_level=25,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
