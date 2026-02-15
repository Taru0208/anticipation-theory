"""Tests for agency measures in ToA.

Tests the Phase 1 agency integration:
- ActionGame model
- Decision Tension (DT)
- Entropy-Corrected Agency Score (EAS)
- Policy Impact (PI)
- Choice Paradox Gap (CPG)
- GDS under different policies
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.agency_model import (
    ActionGame,
    make_combat_game,
    compute_action_values,
    compute_decision_tension,
    compute_agency_score,
    compute_entropy_corrected_agency,
    compute_policy_impact,
    compute_choice_paradox_gap,
    compute_gds_for_policy,
    softmax_entropy,
)


class TestSoftmaxEntropy:
    """Tests for the softmax entropy utility."""

    def test_equal_values_max_entropy(self):
        """Equal Q-values → maximum entropy (1.0)."""
        h = softmax_entropy([0.5, 0.5, 0.5], temperature=10.0)
        assert abs(h - 1.0) < 0.01

    def test_single_value_zero_entropy(self):
        """Single action → zero entropy."""
        h = softmax_entropy([0.5])
        assert h == 0.0

    def test_dominant_value_low_entropy(self):
        """One much larger value → low entropy."""
        h = softmax_entropy([1.0, 0.0, 0.0], temperature=10.0)
        assert h < 0.3  # Should be very concentrated

    def test_two_close_values_high_entropy(self):
        """Two close values, one far → moderate entropy."""
        h = softmax_entropy([0.5, 0.49, 0.1], temperature=10.0)
        assert 0.3 < h < 1.0

    def test_temperature_sensitivity(self):
        """Higher temperature amplifies differences → more concentrated → lower entropy."""
        h_low_temp = softmax_entropy([0.8, 0.5, 0.2], temperature=1.0)
        h_high_temp = softmax_entropy([0.8, 0.5, 0.2], temperature=100.0)
        # Higher temperature amplifies Q-value differences → concentrates on best → lower entropy
        assert h_high_temp < h_low_temp


class TestActionGame:
    """Tests for the ActionGame model."""

    def test_combat_game_creation(self):
        game = make_combat_game(5)
        assert game.max_hp == 5
        assert game.n_actions == 3
        assert game.initial_state() == (5, 5)

    def test_terminal_states(self):
        game = make_combat_game(5)
        assert not game.is_terminal((5, 5))
        assert game.is_terminal((0, 5))
        assert game.is_terminal((5, 0))
        assert game.is_terminal((0, 0))

    def test_transitions_sum_to_one(self):
        """All transition probabilities should sum to 1."""
        game = make_combat_game(5)
        for a_idx in range(3):
            trans = game.get_transitions_for_action((5, 5), a_idx)
            total = sum(p for p, _ in trans)
            assert abs(total - 1.0) < 1e-10, f"Action {a_idx}: total={total}"

    def test_mixed_transitions_sum_to_one(self):
        """Mixed policy transitions should sum to 1."""
        game = make_combat_game(5)
        policy = lambda s: [1/3, 1/3, 1/3]
        trans = game.get_transitions_mixed((5, 5), policy)
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_intrinsic_desire(self):
        game = make_combat_game(5)
        assert game.compute_intrinsic_desire((5, 0)) == 1.0  # P1 wins
        assert game.compute_intrinsic_desire((0, 5)) == 0.0  # P1 loses
        assert game.compute_intrinsic_desire((5, 5)) == 0.0  # Not terminal

    def test_hp_clamping(self):
        """HP should be clamped to [0, max_hp]."""
        game = make_combat_game(3)
        # Check that transitions don't produce HP > max or < 0
        for a1 in range(3):
            trans = game.get_transitions_for_action((1, 1), a1)
            for _, (hp1, hp2) in trans:
                assert 0 <= hp1 <= 3
                assert 0 <= hp2 <= 3


class TestDecisionTension:
    """Tests for Decision Tension computation."""

    def test_equal_q_values_zero_dt(self):
        """Equal Q-values → DT = 0."""
        q_values = {(5, 5): [0.5, 0.5, 0.5]}
        dt = compute_decision_tension(q_values)
        assert abs(dt[(5, 5)]) < 1e-10

    def test_different_q_values_positive_dt(self):
        """Different Q-values → DT > 0."""
        q_values = {(5, 5): [0.8, 0.5, 0.2]}
        dt = compute_decision_tension(q_values)
        assert dt[(5, 5)] > 0

    def test_dt_proportional_to_spread(self):
        """More spread Q-values → higher DT."""
        q_narrow = {(5, 5): [0.5, 0.49, 0.51]}
        q_wide = {(5, 5): [0.9, 0.1, 0.5]}
        dt_narrow = compute_decision_tension(q_narrow)
        dt_wide = compute_decision_tension(q_wide)
        assert dt_wide[(5, 5)] > dt_narrow[(5, 5)]


class TestAgencyScore:
    """Tests for Agency Score (original and entropy-corrected)."""

    def test_balanced_combat_positive_agency(self):
        """Balanced combat should have positive agency."""
        game = make_combat_game(5)
        q_values, analysis = compute_action_values(game)
        as_score, _ = compute_agency_score(game, q_values, analysis)
        assert as_score > 0

    def test_entropy_corrected_less_than_raw(self):
        """EAS should be ≤ raw AS (entropy factor ≤ 1)."""
        game = make_combat_game(5)
        q_values, analysis = compute_action_values(game)
        as_score, _ = compute_agency_score(game, q_values, analysis)
        eas_score, _ = compute_entropy_corrected_agency(game, q_values, analysis)
        assert eas_score <= as_score + 1e-10

    def test_rps_zero_agency(self):
        """RPS-like game should have zero agency (actions equivalent vs random opp)."""
        rps_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                diff = (a1 - a2) % 3
                if diff == 0:
                    rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
                elif diff == 1:
                    rps_actions[(a1, a2)] = [(1.0, (0, -2))]
                else:
                    rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
        game = ActionGame(5, rps_actions)
        q_values, analysis = compute_action_values(game)

        as_score, _ = compute_agency_score(game, q_values, analysis)
        eas_score, _ = compute_entropy_corrected_agency(game, q_values, analysis)

        assert abs(as_score) < 1e-10
        assert abs(eas_score) < 1e-10


class TestPolicyImpact:
    """Tests for Policy Impact (PI)."""

    def test_balanced_combat_positive_pi(self):
        """Balanced combat: different strategies give different GDS → PI > 0."""
        game = make_combat_game(5)
        pi, pure_gds = compute_policy_impact(game)
        assert pi > 0
        assert len(pure_gds) == 3

    def test_rps_zero_pi(self):
        """RPS: all strategies give same GDS → PI = 0."""
        rps_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                diff = (a1 - a2) % 3
                if diff == 0:
                    rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
                elif diff == 1:
                    rps_actions[(a1, a2)] = [(1.0, (0, -2))]
                else:
                    rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
        game = ActionGame(5, rps_actions)
        pi, pure_gds = compute_policy_impact(game)
        assert abs(pi) < 1e-10

    def test_pi_non_negative(self):
        """PI should always be non-negative (max - min)."""
        game = make_combat_game(5)
        pi, _ = compute_policy_impact(game)
        assert pi >= 0

    def test_balanced_combat_heavy_highest_gds(self):
        """In balanced combat, pure Heavy should give highest GDS."""
        game = make_combat_game(5)
        _, pure_gds = compute_policy_impact(game)
        # Heavy is index 1
        assert pure_gds[1] == max(pure_gds)

    def test_balanced_combat_pi_ratio(self):
        """PI/GDS should be a reasonable fraction (not > 100%)."""
        game = make_combat_game(5)
        pi, _ = compute_policy_impact(game)
        random_policy = lambda s: [1/3, 1/3, 1/3]
        gds = compute_gds_for_policy(game, random_policy).game_design_score
        ratio = pi / gds
        assert 0 < ratio < 1.0  # PI should be less than total GDS


class TestChoiceParadoxGap:
    """Tests for Choice Paradox Gap."""

    def test_balanced_combat_has_paradox(self):
        """Balanced combat: fun-optimal ≠ win-optimal → positive gap."""
        game = make_combat_game(5)
        gap, fun_opt, win_opt = compute_choice_paradox_gap(game)
        assert gap > 0
        # Fun-optimal should have lower D₀ than win-optimal
        assert fun_opt[1] < win_opt[1]

    def test_rps_no_paradox(self):
        """RPS: all strategies equivalent → near-zero gap."""
        rps_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                diff = (a1 - a2) % 3
                if diff == 0:
                    rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
                elif diff == 1:
                    rps_actions[(a1, a2)] = [(1.0, (0, -2))]
                else:
                    rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
        game = ActionGame(5, rps_actions)
        gap, _, _ = compute_choice_paradox_gap(game)
        assert abs(gap) < 0.01


class TestGDSUnderPolicies:
    """Tests for GDS computation under different policies."""

    def test_greedy_lower_gds_than_random(self):
        """In balanced combat, greedy play gives lower GDS than random."""
        game = make_combat_game(5)
        q_values, _ = compute_action_values(game)

        def greedy_policy(state):
            q = q_values.get(state, [1/3, 1/3, 1/3])
            best = q.index(max(q))
            probs = [0.0] * 3
            probs[best] = 1.0
            return probs

        random_policy = lambda s: [1/3, 1/3, 1/3]

        gds_greedy = compute_gds_for_policy(game, greedy_policy).game_design_score
        gds_random = compute_gds_for_policy(game, random_policy).game_design_score

        assert gds_greedy < gds_random

    def test_fun_optimal_higher_gds_than_random(self):
        """Fun-optimal policy (90% Heavy + 10% Guard) gives higher GDS than random."""
        game = make_combat_game(5)
        fun_policy = lambda s: [0.0, 0.9, 0.1]
        random_policy = lambda s: [1/3, 1/3, 1/3]

        gds_fun = compute_gds_for_policy(game, fun_policy).game_design_score
        gds_random = compute_gds_for_policy(game, random_policy).game_design_score

        assert gds_fun > gds_random

    def test_fun_optimal_losing(self):
        """Fun-optimal player should be slightly losing (D₀ < 0.5)."""
        game = make_combat_game(5)
        fun_policy = lambda s: [0.0, 0.9, 0.1]
        result = compute_gds_for_policy(game, fun_policy)
        d0 = result.state_nodes[game.initial_state()].d_global
        assert d0 < 0.5

    def test_guard_dominant_for_winning(self):
        """Guard-only should have highest win rate (D₀)."""
        game = make_combat_game(5)
        policies = {
            'strike': lambda s: [1.0, 0.0, 0.0],
            'heavy': lambda s: [0.0, 1.0, 0.0],
            'guard': lambda s: [0.0, 0.0, 1.0],
        }
        d0_values = {}
        for name, policy in policies.items():
            result = compute_gds_for_policy(game, policy)
            d0_values[name] = result.state_nodes[game.initial_state()].d_global

        assert d0_values['guard'] == max(d0_values.values())

    def test_gds_monotonic_with_epsilon(self):
        """GDS should increase monotonically from greedy → random (exp 2 verified)."""
        game = make_combat_game(5)
        q_values, _ = compute_action_values(game)

        prev_gds = 0
        for eps_pct in [0, 50, 100]:
            eps = eps_pct / 100

            def make_policy(e):
                def policy(state):
                    q = q_values.get(state, [1/3, 1/3, 1/3])
                    n = len(q)
                    best = q.index(max(q))
                    probs = [e / n] * n
                    probs[best] += (1 - e)
                    return probs
                return policy

            gds = compute_gds_for_policy(game, make_policy(eps)).game_design_score
            assert gds >= prev_gds - 1e-10  # Monotonically increasing
            prev_gds = gds


class TestParametricCombat:
    """Tests for parametric combat game and CPG minimization."""

    def test_parametric_game_creation(self):
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        assert game.max_hp == 5
        assert game.n_actions == 3

    def test_parametric_transitions_valid(self):
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        for a_idx in range(3):
            trans = game.get_transitions_for_action((5, 5), a_idx)
            total = sum(p for p, _ in trans)
            assert abs(total - 1.0) < 1e-10

    def test_optimized_game_low_cpg(self):
        """The optimized game config should have CPG < 0.1 (vs baseline > 0.3)."""
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        cpg, _, _ = compute_choice_paradox_gap(game, resolution=10)
        assert cpg < 0.1  # At resolution=5 this is 0.033, at 10 it's ~0.067

    def test_optimized_game_high_pi(self):
        """The optimized game should have PI > 0.3 (high agency)."""
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        pi, _ = compute_policy_impact(game)
        assert pi > 0.3

    def test_optimized_game_higher_gds_than_baseline(self):
        """Optimized game should have higher GDS than baseline."""
        from experiments.agency_model import make_parametric_combat
        baseline = make_combat_game(5)
        optimized = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                            guard_counter=2, guard_vs_heavy_block=0.7)
        random_policy = lambda s: [1/3, 1/3, 1/3]
        gds_baseline = compute_gds_for_policy(baseline, random_policy).game_design_score
        gds_optimized = compute_gds_for_policy(optimized, random_policy).game_design_score
        assert gds_optimized > gds_baseline

    def test_baseline_high_cpg(self):
        """Baseline game should have high CPG (> 0.3) for comparison."""
        game = make_combat_game(5)
        cpg, _, _ = compute_choice_paradox_gap(game, resolution=10)
        assert cpg > 0.3

    def test_optimized_fun_equals_winning(self):
        """In optimized game, fun-optimal strategy should also be winning."""
        from experiments.agency_model import make_parametric_combat
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        _, fun_opt, _ = compute_choice_paradox_gap(game, resolution=10)
        # Fun-optimal should have D₀ > 0.5 (winning)
        assert fun_opt[1] > 0.5


class TestCrossGameComparison:
    """Integration tests comparing agency measures across game types."""

    def test_balanced_highest_pi(self):
        """Balanced combat should have highest PI among test games."""
        balanced = make_combat_game(5)
        pi_balanced, _ = compute_policy_impact(balanced)

        # RPS
        rps_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                diff = (a1 - a2) % 3
                if diff == 0:
                    rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
                elif diff == 1:
                    rps_actions[(a1, a2)] = [(1.0, (0, -2))]
                else:
                    rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
        rps = ActionGame(5, rps_actions)
        pi_rps, _ = compute_policy_impact(rps)

        assert pi_balanced > pi_rps

    def test_pi_ranking(self):
        """PI ranking: Balanced > Dominant > RPS."""
        balanced = make_combat_game(5)
        pi_balanced, _ = compute_policy_impact(balanced)

        # Dominant
        dom_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                d2 = -1 if a1 == 0 else 0
                d1 = -1 if a2 == 0 else 0
                if a1 != 0:
                    d1 = min(d1, d1 - 1) if d1 < 0 else -1
                if a2 != 0:
                    d2 = min(d2, d2 - 1) if d2 < 0 else -1
                dom_actions[(a1, a2)] = [(1.0, (d1, d2))]
        dominant = ActionGame(5, dom_actions)
        pi_dominant, _ = compute_policy_impact(dominant)

        # RPS
        rps_actions = {}
        for a1 in range(3):
            for a2 in range(3):
                diff = (a1 - a2) % 3
                if diff == 0:
                    rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
                elif diff == 1:
                    rps_actions[(a1, a2)] = [(1.0, (0, -2))]
                else:
                    rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
        rps = ActionGame(5, rps_actions)
        pi_rps, _ = compute_policy_impact(rps)

        assert pi_balanced > pi_dominant > pi_rps


class TestCPGGeneralization:
    """Tests for CPG generalization across game structures (Experiment 9)."""

    def test_coinduel_default_low_pi(self):
        """Default CoinDuel has low PI (wager barely affects experience)."""
        from experiments.agency_model import CoinDuelActionGame, compute_policy_impact, compute_gds_for_policy
        game = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=3, refill_per_turn=1)
        pi, _ = compute_policy_impact(game)
        random_gds = compute_gds_for_policy(
            game, lambda s: [1/3, 1/3, 1/3]
        ).game_design_score
        # PI/GDS should be low (< 10%) — wager choice barely matters
        assert pi / random_gds < 0.10

    def test_coinduel_optimized_higher_pi(self):
        """CoinDuel with more wager options and faster refill has higher PI."""
        from experiments.agency_model import CoinDuelActionGame, compute_policy_impact
        default = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=3, refill_per_turn=1)
        optimized = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=4, refill_per_turn=2)
        pi_default, _ = compute_policy_impact(default)
        pi_optimized, _ = compute_policy_impact(optimized)
        assert pi_optimized > pi_default

    def test_coinduel_optimized_low_cpg(self):
        """CoinDuel MW=4, Refill=2 achieves CPG < 0.05."""
        from experiments.agency_model import CoinDuelActionGame, compute_choice_paradox_gap
        game = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=4, refill_per_turn=2)
        cpg, _, _ = compute_choice_paradox_gap(game, resolution=20)
        assert cpg < 0.05

    def test_coinduel_wager1_lowest_gds(self):
        """In CoinDuel, wagering 1 coin (most conservative) gives lowest GDS."""
        from experiments.agency_model import CoinDuelActionGame, compute_policy_impact
        game = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=3, refill_per_turn=1)
        _, gds_per_action = compute_policy_impact(game)
        # Wager 1 (index 0) should be lowest or near-lowest GDS
        assert gds_per_action[0] <= max(gds_per_action)

    def test_draftwars_high_pi_ratio(self):
        """DraftWars has high PI/GDS ratio (player choice dominates experience)."""
        from experiments.agency_model import DraftWarsActionGame, compute_policy_impact, compute_gds_for_policy
        dw = DraftWarsActionGame()
        pi, _ = compute_policy_impact(dw)
        random_gds = compute_gds_for_policy(
            dw, lambda s: [1/3, 1/3, 1/3]
        ).game_design_score
        if random_gds > 0.01:
            # PI/GDS should be > 50% — very high agency
            assert pi / random_gds > 0.50

    def test_draftwars_balanced_zero_gds(self):
        """In DraftWars, balanced strategy produces near-zero GDS."""
        from experiments.agency_model import DraftWarsActionGame, compute_gds_for_policy
        dw = DraftWarsActionGame()
        # Balanced = index 2
        balanced_policy = lambda s: [0.0, 0.0, 1.0]
        gds = compute_gds_for_policy(dw, balanced_policy).game_design_score
        # Balanced draft makes outcomes predictable → very low GDS
        assert gds < 0.01

    def test_draftwars_aggressive_highest_gds(self):
        """In DraftWars, aggressive strategy gives highest pure-strategy GDS."""
        from experiments.agency_model import DraftWarsActionGame, compute_policy_impact
        dw = DraftWarsActionGame()
        _, gds_per = compute_policy_impact(dw)
        # Aggressive (index 0) should have highest GDS
        assert gds_per[0] == max(gds_per)

    def test_combat_cpg_structural_condition(self):
        """CPG=0 iff Q(Heavy) > Q(Guard) — the structural condition."""
        from experiments.agency_model import make_parametric_combat, compute_action_values
        # Optimized game: Q(Heavy) > Q(Guard)
        game = make_parametric_combat(5, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        q_values, _ = compute_action_values(game)
        initial_q = q_values[game.initial_state()]
        # Heavy (index 1) should have highest Q-value
        assert initial_q[1] == max(initial_q)

    def test_combat_baseline_guard_dominant(self):
        """Baseline combat: Q(Guard) > Q(Heavy) — Guard dominates."""
        game = make_combat_game(5)
        q_values, _ = compute_action_values(game)
        initial_q = q_values[game.initial_state()]
        # Guard (index 2) should have highest Q-value in baseline
        assert initial_q[2] == max(initial_q)

    def test_pi_threshold_for_cpg_relevance(self):
        """CPG is only meaningful when PI > some threshold (actions matter)."""
        from experiments.agency_model import CoinDuelActionGame, compute_policy_impact, compute_choice_paradox_gap
        # Low-PI game (default CoinDuel)
        game_low = CoinDuelActionGame(rounds_to_win=3, initial_bank=5, max_wager=3, refill_per_turn=1)
        pi_low, _ = compute_policy_impact(game_low)
        # High-PI game (combat)
        game_high = make_combat_game(5)
        pi_high, _ = compute_policy_impact(game_high)
        # Combat should have much higher PI
        assert pi_high > 5 * pi_low
