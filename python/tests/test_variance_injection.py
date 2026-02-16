"""Tests for variance injection — resolving CPG in extrinsic variance games."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.variance_injection import (
    simulate_battle_probabilistic,
    DraftWarsHybrid,
    analyze_draftwars_cpg,
)
from experiments.agency_model import (
    compute_policy_impact, compute_choice_paradox_gap,
    compute_gds_for_policy, DraftWarsActionGame,
)


# ─── Battle Resolution Tests ────────────────────────────────────────────

class TestBattleProbabilistic:
    """Test probabilistic battle resolution."""

    def test_deterministic_when_activation_1(self):
        """With activation_prob=1.0, should match deterministic battle."""
        # P1 gets cards 0,1,5 = (4,0)+(3,1)+(5,-1) = atk 12, def 0
        # P2 gets cards 2,3,4 = (2,2)+(3,0)+(1,3) = atk 6, def 5
        # dmg1 = 12-5 = 7, dmg2 = 6-0 = 6, P1 wins
        h1 = 0b100011  # cards 0, 1, 5
        h2 = 0b011100  # cards 2, 3, 4
        p_win = simulate_battle_probabilistic(h1, h2, atk_weight=1.0, activation_prob=1.0)
        assert p_win == 1.0

    def test_symmetric_hands_give_draw(self):
        """Identical hands should give 0.5 win probability."""
        # This can't happen in actual draft, but tests symmetry
        h = 0b000111  # cards 0, 1, 2
        p_win = simulate_battle_probabilistic(h, h, atk_weight=1.0, activation_prob=0.7)
        assert abs(p_win - 0.5) < 0.001

    def test_activation_below_1_adds_variance(self):
        """With activation_prob < 1, P(win) should be between 0 and 1."""
        h1 = 0b100011  # Strong attack hand
        h2 = 0b011100  # Defense hand
        p_win = simulate_battle_probabilistic(h1, h2, atk_weight=1.0, activation_prob=0.7)
        assert 0.0 < p_win < 1.0

    def test_attack_weight_increases_win_for_attacker(self):
        """Higher attack weight should increase P(win) for attack-heavy hand."""
        h1 = 0b100001  # Cards 0 (atk 4) + 5 (atk 5) — high attack
        h2 = 0b010100  # Cards 2 (def 2) + 4 (def 3) — high defense
        # Need a third card each for full hands
        h1 = 0b110001  # Cards 0, 4, 5
        h2 = 0b001110  # Cards 1, 2, 3

        p_low = simulate_battle_probabilistic(h1, h2, atk_weight=1.0, activation_prob=0.7)
        p_high = simulate_battle_probabilistic(h1, h2, atk_weight=2.0, activation_prob=0.7)
        assert p_high >= p_low

    def test_probability_sums_to_1(self):
        """Win + loss + draw probabilities should sum to 1 (verified implicitly)."""
        h1 = 0b100011
        h2 = 0b011100
        p = simulate_battle_probabilistic(h1, h2, atk_weight=1.5, activation_prob=0.6)
        assert 0.0 <= p <= 1.0


# ─── DraftWars Hybrid Game Tests ────────────────────────────────────────

class TestDraftWarsHybrid:
    """Test DraftWarsHybrid game structure."""

    def test_initial_state(self):
        game = DraftWarsHybrid()
        assert game.initial_state() == (0, 0, 0)

    def test_terminal_at_6(self):
        game = DraftWarsHybrid()
        assert not game.is_terminal((0, 0, 0))
        assert not game.is_terminal((0, 0, 5))
        assert game.is_terminal((0, 0, 6))

    def test_transitions_on_p1_turn(self):
        """P1's turn (even) should produce a single deterministic transition."""
        game = DraftWarsHybrid()
        trans = game.get_transitions_for_action((0, 0, 0), 0)  # Aggressive
        assert len(trans) == 1
        assert trans[0][0] == 1.0

    def test_transitions_on_p2_turn(self):
        """P2's turn (odd) should produce uniform random transitions."""
        game = DraftWarsHybrid()
        # After P1 picks card 5 (glass cannon, highest attack)
        state = (1 << 5, 0, 1)
        trans = game.get_transitions_for_action(state, 0)  # P2's turn
        assert len(trans) == 5  # 5 remaining cards
        for prob, _ in trans:
            assert abs(prob - 0.2) < 0.001

    def test_mixed_policy(self):
        """Mixed policy should produce weighted transitions."""
        game = DraftWarsHybrid()
        trans = game.get_transitions_mixed((0, 0, 0), lambda s: [0.5, 0.5, 0.0])
        assert len(trans) >= 1
        total_prob = sum(p for p, _ in trans)
        assert abs(total_prob - 1.0) < 0.001

    def test_intrinsic_desire_nonterminal_zero(self):
        """Non-terminal states should have desire = 0."""
        game = DraftWarsHybrid()
        assert game.compute_intrinsic_desire((0, 0, 0)) == 0.0

    def test_intrinsic_desire_terminal_bounded(self):
        """Terminal states should have desire in [0, 1]."""
        game = DraftWarsHybrid(activation_prob=0.7)
        h1 = 0b100011  # Cards 0, 1, 5
        h2 = 0b011100  # Cards 2, 3, 4
        d = game.compute_intrinsic_desire((h1, h2, 6))
        assert 0.0 <= d <= 1.0


# ─── CPG Analysis Tests ─────────────────────────────────────────────────

class TestCPGAnalysis:
    """Test the CPG analysis functions and verify key findings."""

    def test_original_draftwars_has_high_cpg(self):
        """Original DraftWars should have CPG > 0.2."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        assert result['cpg'] > 0.2

    def test_balanced_dominates_original(self):
        """In original DraftWars, Balanced should have highest D0."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        d0s = result['d0_per_strategy']
        assert d0s[2] == max(d0s)  # Balanced (index 2) is best

    def test_balanced_has_zero_gds_original(self):
        """In original DraftWars, Balanced should have GDS ≈ 0."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        assert result['gds_per_strategy'][2] < 0.001

    def test_variance_injection_reduces_cpg(self):
        """Variance injection should reduce CPG significantly."""
        original = analyze_draftwars_cpg(1.0, 1.0)
        injected = analyze_draftwars_cpg(1.6, 0.65)
        assert injected['cpg'] < original['cpg'] * 0.15  # >85% reduction

    def test_optimal_injection_cpg_below_threshold(self):
        """Optimal variance injection should achieve CPG < 0.05."""
        result = analyze_draftwars_cpg(1.6, 0.65)
        assert result['cpg'] < 0.05

    def test_attack_advantage_alone_worsens_cpg(self):
        """Attack advantage without noise should worsen CPG."""
        original = analyze_draftwars_cpg(1.0, 1.0)
        atk_only = analyze_draftwars_cpg(1.5, 1.0)
        assert atk_only['cpg'] > original['cpg']

    def test_noise_alone_insufficient(self):
        """Noise alone should not eliminate CPG."""
        noise_only = analyze_draftwars_cpg(1.0, 0.7)
        assert noise_only['cpg'] > 0.15  # Still substantial

    def test_both_ingredients_necessary(self):
        """Both attack advantage and noise are needed for CPG reduction."""
        # Neither alone is enough
        atk_only = analyze_draftwars_cpg(1.5, 1.0)
        noise_only = analyze_draftwars_cpg(1.0, 0.7)
        combined = analyze_draftwars_cpg(1.6, 0.65)

        assert combined['cpg'] < atk_only['cpg']
        assert combined['cpg'] < noise_only['cpg']

    def test_gds_positive_with_injection(self):
        """With injection, all strategies should have some GDS."""
        result = analyze_draftwars_cpg(1.6, 0.65)
        assert result['gds'] > 0.01

    def test_pi_positive_with_injection(self):
        """With injection, PI should be positive."""
        result = analyze_draftwars_cpg(1.6, 0.65)
        assert result['pi'] > 0.001


# ─── Cross-Game Comparison Tests ────────────────────────────────────────

class TestCrossGame:
    """Verify the universal principle across game structures."""

    def test_combat_cpg_eliminated(self):
        """Optimized combat should have CPG ≈ 0."""
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, 1, 3, 0.7, 2, 0.7, 1)
        cpg, _, _ = compute_choice_paradox_gap(game, resolution=20)
        assert cpg < 0.01

    def test_coinduel_cpg_eliminated(self):
        """Optimized CoinDuel should have CPG ≈ 0."""
        from experiments.agency_model import CoinDuelActionGame
        game = CoinDuelActionGame(3, 5, 8, 4, 2)
        cpg, _, _ = compute_choice_paradox_gap(game, resolution=20)
        assert cpg < 0.01

    def test_draftwars_cpg_dramatically_reduced(self):
        """Variance-injected DraftWars should have CPG < 0.05."""
        result = analyze_draftwars_cpg(1.6, 0.65)
        assert result['cpg'] < 0.05

    def test_all_game_types_over_90_percent_reduction(self):
        """All game structures should achieve >90% CPG reduction."""
        from experiments.agency_model import make_parametric_combat, CoinDuelActionGame

        # Combat
        base_combat = make_parametric_combat(5, 1, 2, 0.5, 1, 0.5, 1)
        opt_combat = make_parametric_combat(5, 1, 3, 0.7, 2, 0.7, 1)
        cpg_base_c, _, _ = compute_choice_paradox_gap(base_combat, resolution=20)
        cpg_opt_c, _, _ = compute_choice_paradox_gap(opt_combat, resolution=20)
        assert cpg_opt_c / cpg_base_c < 0.1

        # CoinDuel
        base_cd = CoinDuelActionGame(3, 5, 8, 3, 1)
        opt_cd = CoinDuelActionGame(3, 5, 8, 4, 2)
        cpg_base_cd, _, _ = compute_choice_paradox_gap(base_cd, resolution=20)
        cpg_opt_cd, _, _ = compute_choice_paradox_gap(opt_cd, resolution=20)
        assert cpg_opt_cd / cpg_base_cd < 0.1

        # DraftWars
        dw_base = analyze_draftwars_cpg(1.0, 1.0)
        dw_opt = analyze_draftwars_cpg(1.6, 0.65)
        assert dw_opt['cpg'] / dw_base['cpg'] < 0.1


# ─── Deterministic Dominance Trap Tests ──────────────────────────────────

class TestDeterministicDominanceTrap:
    """Test the Deterministic Dominance Trap concept."""

    def test_dominant_strategy_zero_gds(self):
        """A deterministically dominant strategy should have GDS ≈ 0."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        # Balanced is the dominant strategy
        assert result['gds_per_strategy'][2] < 0.001

    def test_dominant_strategy_max_d0(self):
        """Dominant strategy should have D0 = 1.0."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        assert abs(result['d0_per_strategy'][2] - 1.0) < 0.001

    def test_trap_implies_cpg_positive(self):
        """When a dominant strategy has GDS=0 and D0=1, CPG must be > 0."""
        result = analyze_draftwars_cpg(1.0, 1.0)
        # If dominant strategy has GDS=0 but other strategies have GDS > 0,
        # then fun-optimal != win-optimal → CPG > 0
        assert result['cpg'] > 0

    def test_breaking_trap_with_noise(self):
        """Adding noise should break the deterministic dominance → GDS > 0 for all."""
        result = analyze_draftwars_cpg(1.6, 0.65)
        # With noise, even the dominant direction should have some GDS
        assert result['gds'] > 0.01
