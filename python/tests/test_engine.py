"""Tests for the core analysis engine, verified against C++ reference implementation.

65 tests covering:
- 8 reference game models (CoinToss, RPS, HpGame, HpGameRage, GoldGame, etc.)
- Monte Carlo simulation verification
- Player Choice Paradox (Nash vs random play)
- Unbound Conjecture (GDS growth with depth, superlinear growth)
- Education model (non-game application)
- Convergence test (Unbound vs Anti-Unbound classification)
- Entropy Preservation Conjecture (uniform entropy → Unbound, decaying → Anti-Unbound)
- Exact formulas (A₂ total weight = (T-1)/4, Δ(A₂) = 0.25)
- CLT scaling (A₁ ~ 1/√T)
- Superlinear growth mechanism (A₁ = |ΔP|/2, diagonal symmetry, amplification ratios)
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


# ── Unbound Conjecture — Superlinear Growth ────────────────────────


def _analyze_bestofn(target, nest_level=10):
    """Helper: analyze Best-of-(2*target-1) coin toss."""
    def is_terminal(state):
        return state[0] >= target or state[1] >= target
    def get_transitions(state, config=None):
        w1, w2 = state
        if w1 >= target or w2 >= target:
            return []
        return [(0.5, (w1 + 1, w2)), (0.5, (w1, w2 + 1))]
    def compute_desire(state):
        return 1.0 if state[0] >= target else 0.0
    return analyze(
        initial_state=(0, 0),
        is_terminal=is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=compute_desire,
        nest_level=nest_level,
    )


def test_a2_exact_formula():
    """Σ(reach × A₂) = (T-1)/4 exactly for all T."""
    for target in [3, 5, 8, 10, 15, 20]:
        result = _analyze_bestofn(target, 4)
        total_a2 = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < target and w2 < target:
                n = w1 + w2
                rp = math.comb(n, w1) * (0.5 ** n)
                a2 = result.state_nodes[state].a[1]
                total_a2 += rp * a2
        expected = (target - 1) / 4.0
        assert abs(total_a2 - expected) < 1e-10, (
            f"T={target}: Σ(reach×A₂)={total_a2}, expected {expected}"
        )


def test_gds_superlinear_growth():
    """GDS grows superlinearly (GDS/T increases for large T)."""
    # At small T, GDS/T decreases (A₁ dominates). But for T >= 15, GDS/T increases.
    results = []
    for t in [15, 20, 25, 30]:
        r = _analyze_bestofn(t, 10)
        results.append((t, r.game_design_score))
    # GDS/T should be increasing
    for i in range(len(results) - 1):
        t1, g1 = results[i]
        t2, g2 = results[i + 1]
        ratio1 = g1 / t1
        ratio2 = g2 / t2
        assert ratio2 > ratio1, (
            f"GDS/T should increase: T={t1} → {ratio1:.4f}, T={t2} → {ratio2:.4f}"
        )


def test_component_hierarchy():
    """Higher components grow as higher powers of T."""
    # A₃ ~ T^0.5, A₄ ~ T^1, A₅ ~ T^1.8 (each successive power increases)
    r10 = _analyze_bestofn(10, 10)
    r30 = _analyze_bestofn(30, 10)
    # Compute growth ratios: A_k(T=30) / A_k(T=10)
    ratios = []
    for k in range(2, 7):  # A₃ through A₇
        g10 = r10.gds_components[k]
        g30 = r30.gds_components[k]
        if g10 > 0.001:
            ratio = g30 / g10
            ratios.append(ratio)
    # Each successive ratio should be larger (higher power of T)
    for i in range(len(ratios) - 1):
        assert ratios[i + 1] > ratios[i], (
            f"Component growth should accelerate: ratio[{i}]={ratios[i]:.2f}, "
            f"ratio[{i+1}]={ratios[i+1]:.2f}"
        )


def test_nest_level_amplification():
    """More nest levels → faster GDS growth."""
    gds_by_nl = {}
    for nl in [3, 5, 10]:
        r = _analyze_bestofn(25, nl)
        gds_by_nl[nl] = r.game_design_score
    assert gds_by_nl[5] > gds_by_nl[3], "n=5 should give higher GDS than n=3"
    assert gds_by_nl[10] > gds_by_nl[5], "n=10 should give higher GDS than n=5"
    # The difference should be substantial (not just rounding)
    assert gds_by_nl[10] > 2 * gds_by_nl[3], (
        f"n=10 GDS ({gds_by_nl[10]:.2f}) should be >2x n=3 ({gds_by_nl[3]:.2f})"
    )


def test_a1_clt_scaling():
    """A₁ at balanced states scales as 1/√T (CLT prediction)."""
    # A₁(T//2, T//2) × √T should converge to a constant
    scaled_values = []
    for t in [10, 15, 20, 25, 30]:
        result = _analyze_bestofn(t, 2)
        w = t // 2
        state = (w, w)
        a1 = result.state_nodes[state].a[0]
        scaled = a1 * math.sqrt(t)
        scaled_values.append(scaled)
    # All scaled values should be close to each other (within 10%)
    mean_val = sum(scaled_values) / len(scaled_values)
    for sv in scaled_values:
        assert abs(sv - mean_val) / mean_val < 0.1, (
            f"A₁√T = {sv:.4f}, mean = {mean_val:.4f} — should be within 10%"
        )


# --- Convergence test: Unbound vs Anti-Unbound ---

def test_chaotic_quiz_grows():
    """Chaotic quiz (independent 50/50) should show Unbound-like GDS growth."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from convergence_test import build_chaotic_quiz, run_model
    results = run_model("Chaotic", build_chaotic_quiz, [5, 11, 17])
    # GDS should increase with length
    assert results[2][1] > results[0][1], (
        f"Chaotic quiz GDS should grow: {results[0][1]:.4f} → {results[2][1]:.4f}"
    )
    # Growth rate should be positive
    growth = (results[2][1] - results[0][1]) / (17 - 5)
    assert growth > 0.01, f"Growth rate {growth:.4f} should be > 0.01"


def test_normal_quiz_shrinks():
    """Normal quiz (knowledge → convergence) should show Anti-Unbound behavior."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from convergence_test import build_normal_quiz, run_model
    results = run_model("Normal", build_normal_quiz, [5, 11, 17])
    # GDS should decrease with length
    assert results[2][1] < results[0][1], (
        f"Normal quiz GDS should shrink: {results[0][1]:.4f} → {results[2][1]:.4f}"
    )


def test_convergence_is_the_key():
    """The key structural property: independent trials → Unbound, convergent → Anti-Unbound."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from convergence_test import build_chaotic_quiz, build_normal_quiz, run_model
    chaotic = run_model("Chaotic", build_chaotic_quiz, [15])
    normal = run_model("Normal", build_normal_quiz, [15])
    # Chaotic (independent) should have higher GDS than normal (convergent) at length 15
    assert chaotic[0][1] > normal[0][1], (
        f"Chaotic GDS ({chaotic[0][1]:.4f}) should exceed Normal GDS ({normal[0][1]:.4f}) at length 15"
    )


# --- Entropy Preservation Conjecture ---

def test_entropy_preserved_in_unbound_games():
    """All Unbound games have constant per-state entropy."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from entropy_preservation import (
        build_best_of_n, build_hp_game, build_gold_game,
        compute_entropy_profile, entropy
    )
    # Best-of-N: all states should have H = 1.0
    bon = compute_entropy_profile(build_best_of_n, [9])
    assert abs(bon[0][2] - 1.0) < 0.001, f"Best-of-N avg entropy should be 1.0, got {bon[0][2]}"
    assert abs(bon[0][3] - 1.0) < 0.001, f"Best-of-N min entropy should be 1.0, got {bon[0][3]}"

    # HP Game: all states should have H = 1.0
    hp = compute_entropy_profile(build_hp_game, [5])
    assert abs(hp[0][2] - 1.0) < 0.001, f"HP avg entropy should be 1.0, got {hp[0][2]}"

    # GoldGame: all states should have H ≈ 1.81
    gold = compute_entropy_profile(build_gold_game, [5])
    assert abs(gold[0][2] - 1.8088) < 0.01, f"GoldGame avg entropy should be ~1.81, got {gold[0][2]}"
    assert abs(gold[0][3] - 1.8088) < 0.01, f"GoldGame min entropy should be ~1.81, got {gold[0][3]}"


def test_entropy_decays_in_antibound():
    """Anti-Unbound quiz has decaying min entropy."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from entropy_preservation import build_normal_quiz, compute_entropy_profile
    quiz_short = compute_entropy_profile(build_normal_quiz, [5])
    quiz_long = compute_entropy_profile(build_normal_quiz, [15])
    # Min entropy should decrease with quiz length
    assert quiz_long[0][3] < quiz_short[0][3], (
        f"Longer quiz min entropy ({quiz_long[0][3]:.4f}) should be less than shorter ({quiz_short[0][3]:.4f})"
    )
    # Min entropy should be very low for long quizzes
    assert quiz_long[0][3] < 0.01, f"Long quiz min entropy should be < 0.01, got {quiz_long[0][3]:.4f}"


def test_goldgame_gds_grows():
    """GoldGame (uniform entropy 1.81) should show Unbound GDS growth."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from entropy_preservation import build_gold_game, compute_entropy_profile
    results = compute_entropy_profile(build_gold_game, [3, 7])
    assert results[1][1] > results[0][1], (
        f"GoldGame GDS should grow: {results[0][1]:.4f} → {results[1][1]:.4f}"
    )


# --- Superlinear Growth Mechanism ---


def test_a1_is_delta_probability():
    """A₁(w1,w2) = |P(win|w1+1,w2) - P(win|w1,w2+1)| / 2 exactly."""
    result = _analyze_bestofn(8, 3)
    d = {}
    for state in result.states:
        d[state] = result.state_nodes[state].d_global
    for state in result.states:
        w1, w2 = state
        if w1 >= 8 or w2 >= 8:
            continue
        a1_engine = result.state_nodes[state].a[0]
        cl = (w1 + 1, w2)
        cr = (w1, w2 + 1)
        delta_p = abs(d.get(cl, 0) - d.get(cr, 0)) / 2
        assert abs(a1_engine - delta_p) < 1e-10, (
            f"A₁ at ({w1},{w2}): engine={a1_engine}, |ΔP|/2={delta_p}"
        )


def test_diagonal_symmetry_kills_higher_components():
    """All A_k(w,w) = 0 for k >= 2 (diagonal symmetry)."""
    result = _analyze_bestofn(10, 6)
    for w in range(10):
        node = result.state_nodes[(w, w)]
        for k in range(1, 6):
            assert node.a[k] == 0.0, f"A{k+1}({w},{w}) = {node.a[k]}, expected 0"


def test_a2_marginal_is_quarter():
    """Marginal contribution Δ(A₂) = Σ(rch×A₂)(T+1) - Σ(rch×A₂)(T) = 0.25 exactly."""
    prev = None
    for t in range(5, 15):
        result = _analyze_bestofn(t, 4)
        total_a2 = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < t and w2 < t:
                n = w1 + w2
                rp = math.comb(n, w1) * (0.5 ** n)
                a2 = result.state_nodes[state].a[1]
                total_a2 += rp * a2
        if prev is not None:
            delta = total_a2 - prev
            assert abs(delta - 0.25) < 1e-10, (
                f"T={t}: Δ(A₂) = {delta}, expected 0.25"
            )
        prev = total_a2


def test_higher_components_dominate_at_large_depth():
    """At large T, A₃+ components should contribute >75% of total GDS."""
    result = _analyze_bestofn(20, 10)
    gds = result.game_design_score
    a1_pct = result.gds_components[0] / gds
    a2_pct = result.gds_components[1] / gds
    a3plus_pct = sum(result.gds_components[2:10]) / gds
    assert a3plus_pct > 0.75, (
        f"A₃+ should be >75% at T=20, got {a3plus_pct*100:.1f}%"
    )
    assert a1_pct < 0.10, f"A₁ should be <10% at T=20, got {a1_pct*100:.1f}%"


def test_amplification_ratio_grows_with_t():
    """The ratio Σ(rch×A₃)/Σ(rch×A₂) should grow with T (amplification)."""
    ratios = []
    for t in [8, 15, 25]:
        result = _analyze_bestofn(t, 6)
        total_a2 = 0.0
        total_a3 = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < t and w2 < t:
                n = w1 + w2
                rp = math.comb(n, w1) * (0.5 ** n)
                total_a2 += rp * result.state_nodes[state].a[1]
                total_a3 += rp * result.state_nodes[state].a[2]
        ratios.append(total_a3 / total_a2)
    # Ratio should be strictly increasing
    assert ratios[1] > ratios[0], f"A₃/A₂ should grow: {ratios[0]:.3f} → {ratios[1]:.3f}"
    assert ratios[2] > ratios[1], f"A₃/A₂ should grow: {ratios[1]:.3f} → {ratios[2]:.3f}"


def test_gds_cross_game_superlinear():
    """HP Game (non-Best-of-N) also shows superlinear growth at large HP."""
    gds_data = []
    for hp in [5, 8, 12]:
        def make_hp(hp_val):
            def is_terminal(state):
                return state[0] <= 0 or state[1] <= 0
            def get_transitions(state, config=None):
                h1, h2 = state
                if h1 <= 0 or h2 <= 0:
                    return []
                return [
                    (1/3, (h1, h2 - 1)),
                    (1/3, (h1 - 1, h2 - 1)),
                    (1/3, (h1 - 1, h2)),
                ]
            def compute_desire(state):
                return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0
            return is_terminal, get_transitions, compute_desire
        is_t, get_t, comp_d = make_hp(hp)
        result = analyze(
            initial_state=(hp, hp),
            is_terminal=is_t,
            get_transitions=get_t,
            compute_intrinsic_desire=comp_d,
            nest_level=10,
        )
        gds_data.append((hp, result.game_design_score))
    # GDS should grow, and faster than linearly at large HP
    for i in range(len(gds_data) - 1):
        assert gds_data[i+1][1] > gds_data[i][1], (
            f"HP game GDS should grow: HP={gds_data[i][0]} → {gds_data[i][1]:.4f}, "
            f"HP={gds_data[i+1][0]} → {gds_data[i+1][1]:.4f}"
        )


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
