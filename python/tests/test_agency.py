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
