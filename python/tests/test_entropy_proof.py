"""Tests for the Entropy Preservation Conjecture proof.

Validates:
1. Closed-form A₁ formula for Best-of-N
2. Win probability formula
3. Reach-weighted A₁ sum growth
4. GDS cascade mechanism
5. GDS growth (unboundedness)
6. Converse: quiz model GDS is bounded
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.entropy_proof import (
    binom,
    win_probability,
    delta_exact,
    a1_exact,
    a1_stirling,
    reach_probability,
    total_reach_weighted_a1,
    build_best_of_n,
    analyze_gds_components,
    build_quiz,
    compute_entropy_profile,
)
from toa.engine import analyze


class TestBinomialCoefficients:
    """Test the binomial coefficient helper."""

    def test_base_cases(self):
        assert binom(0, 0) == 1
        assert binom(1, 0) == 1
        assert binom(1, 1) == 1

    def test_pascal_triangle(self):
        assert binom(4, 2) == 6
        assert binom(6, 3) == 20
        assert binom(10, 5) == 252

    def test_boundary(self):
        assert binom(5, -1) == 0
        assert binom(5, 6) == 0

    def test_pascal_identity(self):
        """C(n,k) = C(n-1,k-1) + C(n-1,k) — the key identity used in the proof."""
        for n in range(2, 10):
            for k in range(1, n):
                assert binom(n, k) == binom(n - 1, k - 1) + binom(n - 1, k)


class TestWinProbability:
    """Test the win probability formula D(a,b)."""

    def test_base_cases(self):
        assert win_probability(0, 1) == 1.0
        assert win_probability(1, 0) == 0.0

    def test_symmetry(self):
        """D(a,b) + D(b,a) = 1 for fair coin."""
        for a in range(1, 6):
            for b in range(1, 6):
                assert abs(win_probability(a, b) + win_probability(b, a) - 1.0) < 1e-12

    def test_fair_start(self):
        """D(k,k) = 0.5 for all k (symmetric start)."""
        for k in range(1, 8):
            assert abs(win_probability(k, k) - 0.5) < 1e-12

    def test_recursion(self):
        """D(a,b) = [D(a-1,b) + D(a,b-1)] / 2."""
        for a in range(1, 7):
            for b in range(1, 7):
                expected = (win_probability(a - 1, b) + win_probability(a, b - 1)) / 2
                assert abs(win_probability(a, b) - expected) < 1e-12

    def test_known_values(self):
        assert abs(win_probability(2, 2) - 0.5) < 1e-12
        assert abs(win_probability(1, 2) - 0.75) < 1e-12
        assert abs(win_probability(2, 1) - 0.25) < 1e-12


class TestDeltaFormula:
    """Test the closed-form Δ(a,b) = D(a-1,b) - D(a,b-1)."""

    def test_matches_direct_computation(self):
        """Verify Δ formula against direct win probability difference."""
        for a in range(1, 8):
            for b in range(1, 8):
                direct = win_probability(a - 1, b) - win_probability(a, b - 1)
                formula = delta_exact(a, b)
                assert abs(direct - formula) < 1e-12, f"Mismatch at ({a},{b})"

    def test_positivity(self):
        """Δ(a,b) > 0 for all a,b ≥ 1 (being closer to winning is always better)."""
        for a in range(1, 10):
            for b in range(1, 10):
                assert delta_exact(a, b) > 0

    def test_boundary_values(self):
        """Δ(1,b) = 1/2^{b-1}, Δ(a,1) = 1/2^{a-1}."""
        for b in range(1, 10):
            assert abs(delta_exact(1, b) - 1 / 2 ** (b - 1)) < 1e-12
        for a in range(1, 10):
            assert abs(delta_exact(a, 1) - 1 / 2 ** (a - 1)) < 1e-12

    def test_recursion(self):
        """Δ(a,b) = [Δ(a-1,b) + Δ(a,b-1)] / 2."""
        for a in range(2, 8):
            for b in range(2, 8):
                expected = (delta_exact(a - 1, b) + delta_exact(a, b - 1)) / 2
                assert abs(delta_exact(a, b) - expected) < 1e-12


class TestA1Formula:
    """Test the closed-form A₁ formula and its properties."""

    def test_formula(self):
        """A₁(a,b) = C(a+b-2,a-1) / 2^{a+b-1}."""
        for a in range(1, 8):
            for b in range(1, 8):
                expected = binom(a + b - 2, a - 1) / (2 ** (a + b - 1))
                assert abs(a1_exact(a, b) - expected) < 1e-15

    def test_equals_half_delta(self):
        """A₁ = Δ/2."""
        for a in range(1, 8):
            for b in range(1, 8):
                assert abs(a1_exact(a, b) - delta_exact(a, b) / 2) < 1e-15

    def test_matches_engine(self):
        """Verify A₁ formula matches the ToA engine output."""
        for k in [3, 5, 7]:
            init, term, trans, des = build_best_of_n(k)
            result = analyze(
                initial_state=init(), is_terminal=term, get_transitions=trans,
                compute_intrinsic_desire=des, config=None, nest_level=5,
            )
            for state, node in result.state_nodes.items():
                if term(state):
                    continue
                w, l = state
                a_rem, b_rem = k - w, k - l
                assert abs(node.a[0] - a1_exact(a_rem, b_rem)) < 1e-10

    def test_positivity(self):
        """A₁ > 0 at all non-terminal states."""
        for a in range(1, 15):
            for b in range(1, 15):
                assert a1_exact(a, b) > 0

    def test_minimum_value(self):
        """Minimum A₁ at (1,k) or (k,1) equals 1/2^k.

        A₁(1,k) = C(k-1, 0) / 2^k = 1/2^k.
        """
        for k in range(2, 10):
            min_a1 = min(a1_exact(a, b) for a in range(1, k + 1) for b in range(1, k + 1))
            expected_min = 1 / 2 ** k
            assert abs(min_a1 - expected_min) < 1e-15

    def test_maximum_at_near_terminal(self):
        """Maximum A₁ = 0.5 at state (1,1) for all k."""
        for k in range(2, 10):
            max_a1 = max(a1_exact(a, b) for a in range(1, k + 1) for b in range(1, k + 1))
            assert abs(max_a1 - 0.5) < 1e-15

    def test_symmetry(self):
        """A₁(a,b) = A₁(b,a)."""
        for a in range(1, 8):
            for b in range(1, 8):
                assert abs(a1_exact(a, b) - a1_exact(b, a)) < 1e-15

    def test_stirling_approximation(self):
        """Stirling approximation is accurate for large a+b."""
        for a in [10, 15, 20]:
            for b in [10, 15, 20]:
                exact = a1_exact(a, b)
                approx = a1_stirling(a, b)
                rel_error = abs(exact - approx) / exact
                assert rel_error < 0.05, f"Stirling error {rel_error:.4f} at ({a},{b})"


class TestReachWeightedSum:
    """Test the reach-weighted A₁ sum S₁(k)."""

    def test_growth(self):
        """S₁(k) is monotonically increasing."""
        prev = 0
        for k in range(2, 12):
            s1 = total_reach_weighted_a1(k)
            assert s1 > prev, f"S₁({k}) = {s1} not greater than S₁({k-1}) = {prev}"
            prev = s1

    def test_sqrt_k_scaling(self):
        """S₁(k) / √k converges to 1/√π ≈ 0.5642."""
        target = 1 / math.sqrt(math.pi)
        for k in [10, 15, 20]:
            ratio = total_reach_weighted_a1(k) / math.sqrt(k)
            assert abs(ratio - target) < 0.01, f"Ratio {ratio} far from {target} at k={k}"


class TestGDSCascade:
    """Test the GDS cascade mechanism."""

    def test_component_positivity(self):
        """All GDS components up to nest_level are non-negative."""
        for k in [5, 10, 15]:
            components, _ = analyze_gds_components(k, nest_level=10)
            for m in range(10):
                assert components[m] >= -1e-10, f"GDS_{m+1} negative at k={k}"

    def test_gds1_decreasing(self):
        """GDS₁ decreases with k (A₁ spread thins out over longer paths)."""
        prev = float("inf")
        for k in range(3, 15):
            components, _ = analyze_gds_components(k, nest_level=10)
            assert components[0] < prev + 1e-6
            prev = components[0]

    def test_gds2_converges(self):
        """GDS₂ converges to approximately 0.137."""
        components_10, _ = analyze_gds_components(10, nest_level=10)
        components_15, _ = analyze_gds_components(15, nest_level=10)
        # Both should be close to 0.137
        assert abs(components_10[1] - 0.137) < 0.01
        assert abs(components_15[1] - 0.137) < 0.01

    def test_higher_components_grow(self):
        """GDS₃, GDS₄, GDS₅ are larger at k=15 than at k=5."""
        comp_5, _ = analyze_gds_components(5, nest_level=10)
        comp_15, _ = analyze_gds_components(15, nest_level=10)
        for m in [2, 3, 4]:  # GDS₃, GDS₄, GDS₅
            assert comp_15[m] > comp_5[m], f"GDS_{m+1} not growing"


class TestGDSGrowth:
    """Test that GDS is unbounded (grows with k)."""

    def test_gds_increasing(self):
        """GDS is monotonically increasing for k ≥ 4."""
        prev = 0
        for k in range(4, 16):
            _, gds = analyze_gds_components(k, nest_level=10)
            assert gds > prev, f"GDS({k}) = {gds} not greater than GDS({k-1}) = {prev}"
            prev = gds

    def test_gds_exceeds_one(self):
        """GDS exceeds 1.0 at some finite k (proving unboundedness)."""
        _, gds = analyze_gds_components(16, nest_level=20)
        assert gds > 1.0, f"GDS(16) = {gds} should exceed 1.0"

    def test_superlinear_growth(self):
        """GDS grows faster than linear in k."""
        _, gds_5 = analyze_gds_components(5, nest_level=20)
        _, gds_10 = analyze_gds_components(10, nest_level=20)
        _, gds_20 = analyze_gds_components(20, nest_level=20)
        # If linear: gds_20/gds_10 ≈ gds_10/gds_5 ≈ 2
        # If superlinear: gds_20/gds_10 > gds_10/gds_5
        ratio_1 = gds_10 / gds_5
        ratio_2 = gds_20 / gds_10
        assert ratio_2 > ratio_1, "Growth is not superlinear"

    def test_nest_level_matters(self):
        """Higher nest levels capture more GDS at large k."""
        comp_10, gds_10 = analyze_gds_components(20, nest_level=10)
        comp_20, gds_20 = analyze_gds_components(20, nest_level=20)
        assert gds_20 > gds_10 * 1.05, "NL=20 should capture >5% more GDS than NL=10 at k=20"


class TestConverse:
    """Test the converse: entropy decay → bounded GDS."""

    def test_quiz_gds_bounded(self):
        """Quiz model GDS remains bounded as quiz length increases."""
        gds_values = []
        for num_q in [5, 10, 15]:
            init, term, trans, des = build_quiz(num_q)
            result = analyze(
                initial_state=init(), is_terminal=term, get_transitions=trans,
                compute_intrinsic_desire=des, config=None, nest_level=10,
            )
            gds_values.append(result.game_design_score)
        # GDS should not grow significantly
        assert gds_values[-1] < gds_values[0] * 2, "Quiz GDS growing too fast"
        assert gds_values[-1] < 0.5, "Quiz GDS should be bounded below 0.5"

    def test_quiz_entropy_decays(self):
        """Quiz model has decreasing average entropy."""
        prof_5 = compute_entropy_profile(5, "quiz")
        prof_15 = compute_entropy_profile(15, "quiz")
        assert prof_15["avg_entropy"] < prof_5["avg_entropy"]
        assert prof_15["min_entropy"] < 0.01  # Some states nearly deterministic

    def test_best_of_n_entropy_preserved(self):
        """Best-of-N has constant entropy = 1.0 at all states."""
        for k in [5, 10]:
            prof = compute_entropy_profile(k, "best_of_n")
            assert abs(prof["avg_entropy"] - 1.0) < 1e-10
            assert abs(prof["min_entropy"] - 1.0) < 1e-10

    def test_entropy_gds_correlation(self):
        """Games with preserved entropy have higher GDS than those without."""
        bon_gds = compute_entropy_profile(10, "best_of_n")["gds"]
        quiz_gds = compute_entropy_profile(10, "quiz")["gds"]
        assert bon_gds > quiz_gds * 2, "Best-of-N should have much higher GDS"


class TestEntropyConnection:
    """Test the connection between entropy and A₁."""

    def test_a1_proportional_to_sqrt_pq(self):
        """For binary transitions with probability p: A₁ = √(p(1-p)) × |ΔD|."""
        # In Best-of-N (p=0.5): √(p(1-p)) = 0.5
        # A₁ = 0.5 × |ΔD| = Δ/2
        for a in range(1, 6):
            for b in range(1, 6):
                delta_d = abs(win_probability(a - 1, b) - win_probability(a, b - 1))
                expected = 0.5 * delta_d
                assert abs(a1_exact(a, b) - expected) < 1e-12

    def test_zero_entropy_kills_a1(self):
        """When p→0 or p→1 (H→0), A₁→0 regardless of |ΔD|."""
        # Build a game where some states have p=1 (deterministic)
        k = 5

        def init():
            return (0, 0)

        def term(s):
            return s[0] >= k or s[1] >= k

        def trans(s, _):
            if term(s):
                return []
            w, l = s
            if w >= k - 1:
                return [(1.0, (w + 1, l))]  # Deterministic
            return [(0.5, (w + 1, l)), (0.5, (w, l + 1))]

        def des(s):
            if not term(s):
                return 0.0
            return 1.0 if s[0] >= k else 0.0

        result = analyze(
            initial_state=init(), is_terminal=term, get_transitions=trans,
            compute_intrinsic_desire=des, config=None, nest_level=5,
        )

        # States with w=k-1 should have A₁=0 (deterministic transition)
        for state, node in result.state_nodes.items():
            if term(state):
                continue
            w, l = state
            if w >= k - 1:
                assert abs(node.a[0]) < 1e-10, f"A₁ should be 0 at deterministic state {state}"
