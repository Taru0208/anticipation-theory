"""Tests for Phase 3: Composite Fun Score (CFS).

Validates the CFS formula and its key properties:
1. CFS correctly rewards tension, agency, AND fun-winning alignment
2. Optimized game always scores higher than baseline
3. CPG is the dominant discriminator
4. CFS-optimal configuration is the CPG=0 game
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.composite_fun_score import compute_composite_fun_score
from experiments.agency_model import (
    make_combat_game,
    make_parametric_combat,
    compute_gds_for_policy,
    compute_policy_impact,
    compute_choice_paradox_gap,
)


# ─── Helpers ──────────────────────────────────────────────────────────

def baseline_game():
    game = make_combat_game(max_hp=5)
    game.action_names = ["Strike", "Heavy", "Guard"]
    return game

def optimized_game():
    game = make_parametric_combat(
        max_hp=5, heavy_dmg=3, heavy_hit_prob=0.7,
        guard_counter=2, guard_vs_heavy_block=0.7,
    )
    game.action_names = ["Strike", "Heavy", "Guard"]
    return game


# ─── Core Properties ─────────────────────────────────────────────────

def test_cfs_is_positive():
    """CFS should be positive for any playable game."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5)
    assert r["cfs"] > 0

def test_cfs_components_present():
    """All expected components should be in the result."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    required = ["cfs", "wgds", "gds", "pi", "pi_ratio", "cpg", "cpg_norm",
                 "enl", "profile", "agency_amp", "paradox_penalty", "components",
                 "fun_optimal", "win_optimal"]
    for key in required:
        assert key in r, f"Missing component: {key}"

def test_cfs_formula():
    """CFS = wGDS × agency_amp × paradox_penalty."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    expected = r["wgds"] * r["agency_amp"] * r["paradox_penalty"]
    assert abs(r["cfs"] - expected) < 1e-10, f"CFS={r['cfs']} != wGDS*amp*pen={expected}"

def test_agency_amp_formula():
    """Agency amplifier = 1 + beta * PI/GDS."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    expected = 1.0 + r["pi_ratio"]
    assert abs(r["agency_amp"] - expected) < 1e-10

def test_paradox_penalty_formula():
    """Paradox penalty = 1 - CPG_norm."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    expected = 1.0 - r["cpg_norm"]
    assert abs(r["paradox_penalty"] - expected) < 1e-10


# ─── Headline Result: Optimized > Baseline ──────────────────────────

def test_optimized_beats_baseline_cfs():
    """The CPG-optimized game should have higher CFS than baseline."""
    rb = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    ro = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert ro["cfs"] > rb["cfs"], f"Optimized CFS={ro['cfs']:.4f} <= Baseline CFS={rb['cfs']:.4f}"

def test_optimized_beats_baseline_all_alphas():
    """Optimized > Baseline at all perception levels."""
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        rb = compute_composite_fun_score(baseline_game(), alpha=alpha, nest_level=5, cpg_resolution=10)
        ro = compute_composite_fun_score(optimized_game(), alpha=alpha, nest_level=5, cpg_resolution=10)
        assert ro["cfs"] > rb["cfs"], f"α={alpha}: Opt CFS={ro['cfs']:.4f} <= Base CFS={rb['cfs']:.4f}"

def test_optimized_improvement_is_large():
    """CFS improvement should be substantial (>50%)."""
    rb = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    ro = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    improvement = (ro["cfs"] - rb["cfs"]) / rb["cfs"]
    assert improvement > 0.5, f"Only {improvement*100:.0f}% improvement, expected >50%"


# ─── CPG is the Dominant Factor ──────────────────────────────────────

def test_cpg_zero_in_optimized():
    """Optimized game should have CPG=0 (or very close)."""
    r = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=20)
    assert r["cpg"] < 0.01, f"CPG={r['cpg']:.4f}, expected ~0"

def test_paradox_penalty_one_when_cpg_zero():
    """When CPG=0, paradox penalty should be 1.0 (no penalty)."""
    r = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=20)
    assert r["paradox_penalty"] > 0.99, f"Penalty={r['paradox_penalty']:.4f}, expected ~1.0"

def test_baseline_has_significant_cpg():
    """Baseline game should have non-trivial CPG (>0.2)."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert r["cpg"] > 0.2, f"CPG={r['cpg']:.4f}, expected >0.2"


# ─── Agency Properties ──────────────────────────────────────────────

def test_optimized_has_higher_pi():
    """Optimized game should have higher Policy Impact."""
    rb = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    ro = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert ro["pi"] > rb["pi"], f"Opt PI={ro['pi']:.4f} <= Base PI={rb['pi']:.4f}"

def test_optimized_has_higher_agency_fraction():
    """Optimized game should have higher PI/GDS ratio."""
    rb = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    ro = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert ro["pi_ratio"] > rb["pi_ratio"]


# ─── Fun = Winning Alignment ────────────────────────────────────────

def test_optimized_fun_wins():
    """In the optimized game, the fun-optimal strategy should also win."""
    r = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=20)
    fun_d0 = r["fun_optimal"][1]
    assert fun_d0 > 0.5, f"Fun-optimal D₀={fun_d0:.3f}, but it should be winning (>0.5)"

def test_baseline_fun_loses():
    """In the baseline game, the fun-optimal strategy should lose or barely break even."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    fun_d0 = r["fun_optimal"][1]
    # Fun-optimal should be close to 0.5 or below
    assert fun_d0 < 0.55, f"Fun-optimal D₀={fun_d0:.3f}, expected <0.55 (Choice Paradox)"

def test_optimized_fun_equals_win():
    """In the optimized game, fun-optimal and win-optimal strategies should converge."""
    r = compute_composite_fun_score(optimized_game(), alpha=0.5, nest_level=5, cpg_resolution=20)
    fun_strat = r["fun_optimal"][2]
    win_strat = r["win_optimal"][2]
    # They should be the same or very close
    # Both should be Heavy-dominant
    fun_heavy = fun_strat[1] if len(fun_strat) > 1 else 0
    win_heavy = win_strat[1] if len(win_strat) > 1 else 0
    assert fun_heavy > 0.5, f"Fun-optimal Heavy%={fun_heavy:.0%}, expected >50%"
    assert win_heavy > 0.5, f"Win-optimal Heavy%={win_heavy:.0%}, expected >50%"


# ─── Perceptual Weighting ───────────────────────────────────────────

def test_wgds_less_than_gds():
    """Weighted GDS (α<1) should be less than raw GDS."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert r["wgds"] < r["gds"], f"wGDS={r['wgds']:.4f} >= GDS={r['gds']:.4f}"

def test_wgds_equals_gds_at_alpha_1():
    """At α=1.0, wGDS should equal GDS."""
    r = compute_composite_fun_score(baseline_game(), alpha=1.0, nest_level=5, cpg_resolution=10)
    assert abs(r["wgds"] - r["gds"]) < 1e-6, f"wGDS={r['wgds']:.4f} != GDS={r['gds']:.4f} at α=1"

def test_enl_is_reasonable():
    """Effective nest level should be between 1 and nest_level."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=10, cpg_resolution=10)
    assert 1 <= r["enl"] <= 10, f"ENL={r['enl']}, out of range"

def test_profile_classification():
    """Game should be classified into a valid profile."""
    r = compute_composite_fun_score(baseline_game(), alpha=0.5, nest_level=5, cpg_resolution=10)
    assert r["profile"] in ["SNOWBALL", "BALANCED", "DECAYING", "SHALLOW"]


# ─── Edge Cases ──────────────────────────────────────────────────────

def test_cfs_monotonic_with_beta():
    """Higher beta (agency weight) should increase CFS when PI > 0."""
    game = optimized_game()
    r1 = compute_composite_fun_score(game, alpha=0.5, beta=0.5, nest_level=5, cpg_resolution=10)
    r2 = compute_composite_fun_score(game, alpha=0.5, beta=2.0, nest_level=5, cpg_resolution=10)
    assert r2["cfs"] > r1["cfs"], "Higher beta should increase CFS"

def test_cfs_with_different_hp():
    """CFS should work for different HP values."""
    for hp in [3, 4, 5, 6]:
        game = make_parametric_combat(max_hp=hp, heavy_dmg=3, heavy_hit_prob=0.7,
                                       guard_counter=2, guard_vs_heavy_block=0.7)
        game.action_names = ["Strike", "Heavy", "Guard"]
        r = compute_composite_fun_score(game, alpha=0.5, nest_level=5, cpg_resolution=10)
        assert r["cfs"] > 0, f"HP={hp}: CFS={r['cfs']:.4f} <= 0"
        assert r["gds"] > 0, f"HP={hp}: GDS={r['gds']:.4f} <= 0"


# ─── CFS as Optimization Target ─────────────────────────────────────

def test_cfs_optimal_is_cpg_minimal():
    """The CFS-optimal game should have very low CPG.

    This validates that CFS correctly identifies fun-winning alignment
    as the most important factor.
    """
    results = []
    for hd in [2, 3, 4]:
        for hh in [0.4, 0.6, 0.7, 0.8]:
            for gc in [1, 2]:
                for gb in [0.5, 0.7]:
                    game = make_parametric_combat(
                        max_hp=5, heavy_dmg=hd, heavy_hit_prob=hh,
                        guard_counter=gc, guard_vs_heavy_block=gb,
                    )
                    game.action_names = ["Strike", "Heavy", "Guard"]
                    r = compute_composite_fun_score(game, alpha=0.5, nest_level=5, cpg_resolution=10)
                    results.append(r)

    results.sort(key=lambda x: -x["cfs"])
    best = results[0]
    # The best CFS config should have low CPG
    assert best["cpg"] < 0.1, f"Best CFS config has CPG={best['cpg']:.3f}, expected <0.1"
