"""Tests for the core analysis engine, verified against C++ reference implementation.

48 tests covering:
- 8 reference game models (CoinToss, RPS, HpGame, HpGameRage, GoldGame, etc.)
- Monte Carlo simulation verification
- Player Choice Paradox (Nash vs random play)
- Unbound Conjecture (GDS growth with depth)
- Education model (non-game application)
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.coin_toss import CoinToss
from toa.games.rps import RPS
from toa.games.hpgame import HpGame
from toa.games.goldgame import GoldGame
from toa.games.hpgame_rage import HpGameRage
from toa.games.lanegame import LaneGame
from toa.games.two_turn_game import TwoTurnGame
from toa.games.goldgame_critical import GoldGameCritical
from toa.simulate import simulate_gds


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


# ── HpGame_Rage ────────────────────────────────────────────


def test_hpgame_rage_gds():
    """HpGame_Rage GDS ≈ 0.544 with 10% critical (from C++ reference / paper)."""
    config = HpGameRage.Config(critical_chance=0.10)
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    assert approx(result.game_design_score, 0.544, tolerance=0.005), \
        f"GDS = {result.game_design_score}, expected ~0.544"


def test_hpgame_rage_improvement_over_baseline():
    """HpGame_Rage should have ~26.5% higher GDS than baseline HpGame."""
    baseline = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    config = HpGameRage.Config(critical_chance=0.10)
    rage = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    improvement = (rage.game_design_score - baseline.game_design_score) / baseline.game_design_score
    assert 0.20 < improvement < 0.35, f"Improvement = {improvement*100:.1f}%, expected ~26.5%"


def test_hpgame_rage_optimal_crit_13():
    """Optimal critical chance should be near 13% (paper finding)."""
    best_crit = 0
    best_gds = 0
    for crit_pct in [10, 11, 12, 13, 14, 15, 16]:
        config = HpGameRage.Config(critical_chance=crit_pct / 100.0)
        result = analyze(
            initial_state=HpGameRage.initial_state(),
            is_terminal=HpGameRage.is_terminal,
            get_transitions=HpGameRage.get_transitions,
            compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
            config=config,
            nest_level=5,
        )
        if result.game_design_score > best_gds:
            best_gds = result.game_design_score
            best_crit = crit_pct
    assert best_crit == 13, f"Optimal crit = {best_crit}%, expected 13%"


def test_hpgame_rage_state_count():
    """HpGame_Rage with rage should have more states than baseline."""
    config = HpGameRage.Config(critical_chance=0.10)
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=1,
    )
    assert len(result.states) > 100, f"Only {len(result.states)} states (expected >100 due to rage dimension)"


def test_hpgame_rage_asymmetric_engagement():
    """Slightly asymmetric states should generally be more engaging (paper insight)."""
    config = HpGameRage.Config(critical_chance=0.10)
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    # The paper shows (4,3) states are more engaging than (3,3) or (4,4) states
    # Find top states and verify that asymmetric (hp difference of 1) dominates
    state_engagement = []
    for s in result.states:
        node = result.state_nodes[s]
        total_a = sum(node.a[:5])
        if total_a > 0.5 and not HpGameRage.is_terminal(s):
            state_engagement.append((s, total_a))
    state_engagement.sort(key=lambda x: x[1], reverse=True)
    # Top state should have HP difference of 1
    top_state = state_engagement[0][0]
    hp_diff = abs(top_state[0] - top_state[1])
    assert hp_diff == 1, f"Top state {top_state} has HP diff {hp_diff}, expected 1"


def test_hpgame_rage_higher_components():
    """Rage mechanics should produce significant higher-order components."""
    config = HpGameRage.Config(critical_chance=0.10)
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    # A₃ and A₄ components should be significant
    assert result.gds_components[2] > 0.05, f"A₃ = {result.gds_components[2]}, expected > 0.05"
    assert result.gds_components[3] > 0.02, f"A₄ = {result.gds_components[3]}, expected > 0.02"


def test_hpgame_rage_zero_crit_equals_baseline():
    """With 0% critical chance, HpGame_Rage should approximate baseline HpGame."""
    config = HpGameRage.Config(critical_chance=0.0)
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    baseline = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    assert approx(result.game_design_score, baseline.game_design_score, tolerance=0.005), \
        f"0% crit GDS={result.game_design_score}, baseline={baseline.game_design_score}"


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


# ── LaneGame ──────────────────────────────────────────────


def test_lanegame_gds():
    """LaneGame should produce positive GDS — similar structure to HpGame."""
    result = analyze(
        initial_state=LaneGame.initial_state(),
        is_terminal=LaneGame.is_terminal,
        get_transitions=LaneGame.get_transitions,
        compute_intrinsic_desire=LaneGame.compute_intrinsic_desire,
        nest_level=5,
    )
    assert result.game_design_score > 0.3, f"GDS = {result.game_design_score}"


def test_lanegame_symmetric():
    """LaneGame should be symmetric (D₀ close to 0.5)."""
    result = analyze(
        initial_state=LaneGame.initial_state(),
        is_terminal=LaneGame.is_terminal,
        get_transitions=LaneGame.get_transitions,
        compute_intrinsic_desire=LaneGame.compute_intrinsic_desire,
        nest_level=1,
    )
    d0 = result.state_nodes[LaneGame.initial_state()].d_global
    assert 0.3 < d0 < 0.7, f"D₀ = {d0}, expected near 0.5"


# ── TwoTurnGame ──────────────────────────────────────────


def test_two_turn_default():
    """Default TwoTurnGame (all p=0.5) should have D₀ = 0.5."""
    result = analyze(
        initial_state=TwoTurnGame.initial_state(),
        is_terminal=TwoTurnGame.is_terminal,
        get_transitions=TwoTurnGame.get_transitions,
        compute_intrinsic_desire=TwoTurnGame.compute_intrinsic_desire,
        nest_level=5,
    )
    d0 = result.state_nodes[(0, 0)].d_global
    assert approx(d0, 0.5), f"D₀ = {d0}"


def test_two_turn_intermediate_a1():
    """TwoTurnGame intermediate nodes should have A₁ = 0.5 (coin toss equivalent)."""
    result = analyze(
        initial_state=TwoTurnGame.initial_state(),
        is_terminal=TwoTurnGame.is_terminal,
        get_transitions=TwoTurnGame.get_transitions,
        compute_intrinsic_desire=TwoTurnGame.compute_intrinsic_desire,
        nest_level=5,
    )
    # With all p=0.5, initial A₁=0 (children have equal d_global),
    # but intermediate nodes have A₁=0.5 (coin toss)
    a1_10 = result.state_nodes[(1, 0)].a[0]
    assert approx(a1_10, 0.5), f"A₁ at (1,0) = {a1_10}, expected 0.5"


def test_two_turn_config():
    """TwoTurnGame with asymmetric probabilities should change GDS."""
    config_sym = TwoTurnGame.Config(0.5, 0.5, 0.5)
    config_asym = TwoTurnGame.Config(0.3, 0.7, 0.2)

    r1 = analyze(
        initial_state=TwoTurnGame.initial_state(),
        is_terminal=TwoTurnGame.is_terminal,
        get_transitions=TwoTurnGame.get_transitions,
        compute_intrinsic_desire=TwoTurnGame.compute_intrinsic_desire,
        config=config_sym,
        nest_level=5,
    )
    r2 = analyze(
        initial_state=TwoTurnGame.initial_state(),
        is_terminal=TwoTurnGame.is_terminal,
        get_transitions=TwoTurnGame.get_transitions,
        compute_intrinsic_desire=TwoTurnGame.compute_intrinsic_desire,
        config=config_asym,
        nest_level=5,
    )
    assert r1.game_design_score != r2.game_design_score, "Config should change GDS"


# ── GoldGameCritical ─────────────────────────────────────


def test_goldgame_critical_basic():
    """GoldGameCritical with no crits should work."""
    config = GoldGameCritical.Config(critical_chance=0.0)
    result = analyze(
        initial_state=GoldGameCritical.initial_state(),
        is_terminal=GoldGameCritical.is_terminal,
        get_transitions=GoldGameCritical.get_transitions,
        compute_intrinsic_desire=GoldGameCritical.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    assert result.game_design_score > 0, f"GDS = {result.game_design_score}"


def test_goldgame_critical_with_steal():
    """Adding steal should change GDS."""
    no_steal = GoldGameCritical.Config(critical_chance=0.1, steal_percentage=0.0)
    with_steal = GoldGameCritical.Config(critical_chance=0.1, steal_percentage=0.2)

    r1 = analyze(
        initial_state=GoldGameCritical.initial_state(),
        is_terminal=GoldGameCritical.is_terminal,
        get_transitions=GoldGameCritical.get_transitions,
        compute_intrinsic_desire=GoldGameCritical.compute_intrinsic_desire,
        config=no_steal,
        nest_level=5,
    )
    r2 = analyze(
        initial_state=GoldGameCritical.initial_state(),
        is_terminal=GoldGameCritical.is_terminal,
        get_transitions=GoldGameCritical.get_transitions,
        compute_intrinsic_desire=GoldGameCritical.compute_intrinsic_desire,
        config=with_steal,
        nest_level=5,
    )
    assert r1.game_design_score != r2.game_design_score, "Steal should change GDS"


def test_goldgame_critical_geometric():
    """Geometric reward type should also work."""
    config = GoldGameCritical.Config(
        reward_type="geometric",
        critical_chance=0.1,
        steal_percentage=0.1,
    )
    result = analyze(
        initial_state=GoldGameCritical.initial_state(),
        is_terminal=GoldGameCritical.is_terminal,
        get_transitions=GoldGameCritical.get_transitions,
        compute_intrinsic_desire=GoldGameCritical.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    assert result.game_design_score > 0, f"GDS = {result.game_design_score}"


# ── Monte Carlo Simulation ────────────────────────────────


def test_simulate_coin_toss():
    """Monte Carlo GDS for coin toss should approximate exact GDS."""
    result = simulate_gds(
        initial_state=CoinToss.initial_state(),
        is_terminal=CoinToss.is_terminal,
        get_transitions=CoinToss.get_transitions,
        compute_intrinsic_desire=CoinToss.compute_intrinsic_desire,
        nest_level=5,
        num_simulations=10000,
        seed=42,
    )
    assert approx(result["gds_sim"], result["gds_exact"], tolerance=0.05), \
        f"Sim={result['gds_sim']:.4f}, Exact={result['gds_exact']:.4f}"


def test_simulate_hpgame():
    """Monte Carlo GDS for HpGame should approximate exact GDS."""
    result = simulate_gds(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
        num_simulations=10000,
        seed=42,
    )
    assert result["relative_error"] < 0.1, \
        f"Relative error {result['relative_error']:.3f} too high"


def test_simulate_hpgame_rage():
    """Monte Carlo GDS for HpGame_Rage should approximate exact GDS."""
    config = HpGameRage.Config(critical_chance=0.10)
    result = simulate_gds(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
        num_simulations=10000,
        seed=42,
    )
    assert result["relative_error"] < 0.15, \
        f"Relative error {result['relative_error']:.3f} too high"


# ── Edge cases ─────────────────────────────────────────────


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


# ── Player Choice Paradox ─────────────────────────────────


def test_choice_random_play_gds():
    """Tactical combat with random play should produce GDS > 0.4."""
    from experiments.player_choice import build_choice_game, random_play
    init, is_term, trans, desire = build_choice_game(5, random_play, random_play)
    result = analyze(
        initial_state=init(), is_terminal=is_term,
        get_transitions=trans, compute_intrinsic_desire=desire, nest_level=5,
    )
    assert approx(result.game_design_score, 0.527, tolerance=0.02), \
        f"GDS = {result.game_design_score}, expected ~0.527"


def test_choice_nash_reduces_gds():
    """Nash equilibrium should produce LOWER GDS than random play."""
    from experiments.player_choice import build_choice_game, random_play, compute_nash_equilibrium
    nash_model, _ = compute_nash_equilibrium(5)

    init_r, _, trans_r, des_r = build_choice_game(5, random_play, random_play)
    init_n, _, trans_n, des_n = build_choice_game(5, nash_model, nash_model)

    r_random = analyze(initial_state=init_r(), is_terminal=lambda s: s[0]<=0 or s[1]<=0,
                       get_transitions=trans_r, compute_intrinsic_desire=des_r, nest_level=5)
    r_nash = analyze(initial_state=init_n(), is_terminal=lambda s: s[0]<=0 or s[1]<=0,
                     get_transitions=trans_n, compute_intrinsic_desire=des_n, nest_level=5)

    assert r_nash.game_design_score < r_random.game_design_score, \
        f"Nash GDS {r_nash.game_design_score} should be < Random GDS {r_random.game_design_score}"


def test_choice_pure_strategy_zero_gds():
    """Pure strategy (always same action) should produce GDS = 0."""
    from experiments.player_choice import build_choice_game, aggressive_play
    init, is_term, trans, desire = build_choice_game(5, aggressive_play, aggressive_play)
    result = analyze(
        initial_state=init(), is_terminal=is_term,
        get_transitions=trans, compute_intrinsic_desire=desire, nest_level=5,
    )
    assert result.game_design_score < 0.01, \
        f"Pure strategy GDS = {result.game_design_score}, expected ~0"


# ── Unbound Conjecture ───────────────────────────────────


def test_unbound_bestofn_grows():
    """Best-of-N GDS should grow with N."""
    from experiments.unbound_conjecture_v2 import BestOfN, measure_gds
    gds_3 = measure_gds(BestOfN(n_rounds=5), nest_level=5)["gds"]
    gds_9 = measure_gds(BestOfN(n_rounds=17), nest_level=5)["gds"]
    assert gds_9 > gds_3, f"GDS should grow: depth 3={gds_3}, depth 9={gds_9}"


def test_unbound_bestofn_exceeds_one():
    """Best-of-N should exceed GDS 1.0 at sufficient depth."""
    from experiments.unbound_conjecture_v2 import BestOfN, measure_gds
    r = measure_gds(BestOfN(n_rounds=33), nest_level=10)
    assert r["gds"] > 1.0, f"GDS = {r['gds']}, expected > 1.0 at depth 17"


def test_unbound_combat_grows():
    """Simple combat GDS should grow with HP."""
    from experiments.unbound_conjecture_v2 import ScalableCombat, measure_gds
    gds_3 = measure_gds(ScalableCombat(max_hp=3), nest_level=5)["gds"]
    gds_10 = measure_gds(ScalableCombat(max_hp=10), nest_level=5)["gds"]
    assert gds_10 > gds_3, f"GDS should grow: hp3={gds_3}, hp10={gds_10}"


def test_unbound_goldgame_grows():
    """GoldGame GDS should grow with turns."""
    from experiments.unbound_conjecture_v2 import GoldGameMultiplicative, measure_gds
    gds_3 = measure_gds(GoldGameMultiplicative(max_turns=3), nest_level=5)["gds"]
    gds_10 = measure_gds(GoldGameMultiplicative(max_turns=10), nest_level=5)["gds"]
    assert gds_10 > gds_3, f"GDS should grow: t3={gds_3}, t10={gds_10}"


# ── Education Model ──────────────────────────────────────


def test_education_easy_beats_hard():
    """Easy quizzes should have higher GDS than hard quizzes."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_flat_curve
    easy = QuizConfig(num_questions=8, difficulties=make_flat_curve(8, 1.0), difficulty_mode="curve")
    hard = QuizConfig(num_questions=8, difficulties=make_flat_curve(8, 5.0), difficulty_mode="curve")
    r_easy = analyze_quiz(easy)
    r_hard = analyze_quiz(hard)
    assert r_easy.game_design_score > r_hard.game_design_score * 10, \
        f"Easy GDS={r_easy.game_design_score}, Hard GDS={r_hard.game_design_score}"


def test_education_binary_beats_graduated():
    """Binary pass/fail should produce higher GDS than graduated scoring."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_flat_curve
    binary = QuizConfig(num_questions=8, difficulties=make_flat_curve(8, 1.0),
                       difficulty_mode="curve", graduated_desire=False)
    graduated = QuizConfig(num_questions=8, difficulties=make_flat_curve(8, 1.0),
                          difficulty_mode="curve", graduated_desire=True)
    r_bin = analyze_quiz(binary)
    r_grad = analyze_quiz(graduated)
    assert r_bin.game_design_score > r_grad.game_design_score, \
        f"Binary GDS={r_bin.game_design_score}, Graduated GDS={r_grad.game_design_score}"


def test_education_forgetting_reduces_gds():
    """Forgetting mechanics should reduce quiz GDS (unlike games)."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_ascending_curve
    no_forget = QuizConfig(num_questions=8, difficulties=make_ascending_curve(8, 1.0, 5.0),
                          difficulty_mode="curve", incorrect_loss=0)
    forget = QuizConfig(num_questions=8, difficulties=make_ascending_curve(8, 1.0, 5.0),
                       difficulty_mode="curve", incorrect_loss=1)
    r_no = analyze_quiz(no_forget)
    r_forget = analyze_quiz(forget)
    assert r_no.game_design_score > r_forget.game_design_score, \
        f"No-forget GDS={r_no.game_design_score}, Forget GDS={r_forget.game_design_score}"


def test_education_anti_unbound():
    """Quiz GDS should decrease (not increase) with length — opposite of games."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_ascending_curve
    short = QuizConfig(num_questions=3, max_knowledge=6, mastery_threshold=2,
                      difficulties=make_ascending_curve(3, 1.0, 2.0), difficulty_mode="curve")
    long = QuizConfig(num_questions=15, max_knowledge=15, mastery_threshold=7,
                     difficulties=make_ascending_curve(15, 1.0, 9.0), difficulty_mode="curve")
    r_short = analyze_quiz(short)
    r_long = analyze_quiz(long)
    assert r_short.game_design_score > r_long.game_design_score, \
        f"Short GDS={r_short.game_design_score}, Long GDS={r_long.game_design_score}"


def test_education_goldilocks_zone():
    """Peak GDS should be at moderate difficulty, not easiest or hardest."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_flat_curve
    # Low mastery threshold so all difficulties are "winnable"
    results = []
    for d in [0, 1, 2, 3, 5]:
        config = QuizConfig(num_questions=8, difficulties=make_flat_curve(8, float(d)),
                           difficulty_mode="curve", mastery_threshold=1)
        r = analyze_quiz(config)
        results.append((d, r.game_design_score))
    # d=1 should beat d=0 (too easy) and d=5 (too hard)
    gds_0 = next(g for d, g in results if d == 0)
    gds_1 = next(g for d, g in results if d == 1)
    gds_5 = next(g for d, g in results if d == 5)
    assert gds_1 > gds_0, f"d=1 GDS={gds_1} should beat d=0 GDS={gds_0}"
    assert gds_1 > gds_5, f"d=1 GDS={gds_1} should beat d=5 GDS={gds_5}"


def test_education_excitement_at_threshold():
    """States near mastery threshold should have highest A₁."""
    from experiments.education_model import QuizConfig, analyze_quiz, make_ascending_curve
    config = QuizConfig(num_questions=8, difficulties=make_ascending_curve(8, 0.5, 3.0),
                       difficulty_mode="curve", mastery_threshold=4)
    analysis = analyze_quiz(config)
    # State (3, 7) — one below mastery, last question — should have A₁ = 0.5
    state_near = (3, 7)  # K=3, Q=7
    if state_near in analysis.state_nodes:
        a1 = analysis.state_nodes[state_near].a[0]
        assert approx(a1, 0.5, tolerance=0.01), f"A₁ at threshold-1, last Q = {a1}, expected 0.5"


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
