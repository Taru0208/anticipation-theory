"""Phase 3: Composite Fun Score — Combining all ToA measures into one metric.

Brings together:
- Phase 0: GDS (structural tension)
- Phase 1: PI (agency), CPG (fun-winning alignment)
- Phase 2: wGDS (perceptually weighted tension), tension profile classification

The Composite Fun Score (CFS) answers: "How engaging is this game design,
accounting for tension, agency, and fun-winning alignment?"

CFS = wGDS(α) × (1 + β × PI/GDS) × (1 - CPG_normalized)

Where:
- wGDS(α): perceptually weighted tension (base engagement)
- PI/GDS: agency fraction (amplifier — player control matters)
- CPG_normalized: normalized paradox gap (penalty — fun should align with winning)
- α: perceptual decay (default 0.5 for general audience)
- β: agency amplification weight (default 1.0)

The formula captures:
1. Higher tension → higher score (wGDS)
2. More player control → higher score (PI/GDS amplifier)
3. Fun ≈ winning → higher score (CPG penalty)
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from experiments.agency_model import (
    ActionGame, make_combat_game, make_parametric_combat,
    compute_gds_for_policy, compute_policy_impact,
    compute_choice_paradox_gap,
)
from experiments.perceptual_weighting import weighted_gds, effective_nest_level


def compute_composite_fun_score(game, alpha=0.5, beta=1.0, nest_level=10, cpg_resolution=10):
    """Compute the Composite Fun Score for a game with explicit actions.

    Returns dict with all components:
    - cfs: the composite score
    - wgds: weighted GDS
    - gds: raw GDS
    - pi: policy impact
    - pi_ratio: PI / GDS
    - cpg: choice paradox gap
    - cpg_norm: normalized CPG (0-1)
    - enl: effective nest level
    - profile: tension profile classification
    """
    # Step 1: Compute GDS under random policy
    random_policy = lambda s: [1/game.n_actions] * game.n_actions
    result = compute_gds_for_policy(game, random_policy, nest_level)
    comp = result.gds_components[:nest_level]
    gds = sum(comp)
    wgds = weighted_gds(comp, alpha=alpha)
    enl = effective_nest_level(comp)

    # Growth rate classification
    ratios = []
    for k in range(1, min(5, len(comp))):
        if comp[k-1] > 1e-10:
            ratios.append(comp[k] / comp[k-1])
    avg_r = sum(ratios) / len(ratios) if ratios else 0
    if avg_r > 1.2:
        profile = "SNOWBALL"
    elif avg_r > 0.8:
        profile = "BALANCED"
    elif avg_r > 0.3:
        profile = "DECAYING"
    else:
        profile = "SHALLOW"

    # Step 2: Policy Impact
    pi, gds_values = compute_policy_impact(game, nest_level=nest_level)
    pi_ratio = pi / gds if gds > 1e-10 else 0.0

    # Step 3: Choice Paradox Gap
    cpg, fun_opt, win_opt = compute_choice_paradox_gap(game, nest_level=nest_level, resolution=cpg_resolution)

    # Normalize CPG to [0, 1] range
    # CPG is |D₀(fun) - D₀(win)|, max possible is 1.0 (opposite outcomes)
    cpg_norm = min(cpg, 1.0)

    # Step 4: Composite
    agency_amp = 1.0 + beta * pi_ratio
    paradox_penalty = 1.0 - cpg_norm
    cfs = wgds * agency_amp * paradox_penalty

    return {
        "cfs": cfs,
        "wgds": wgds,
        "gds": gds,
        "pi": pi,
        "pi_ratio": pi_ratio,
        "cpg": cpg,
        "cpg_norm": cpg_norm,
        "enl": enl,
        "profile": profile,
        "agency_amp": agency_amp,
        "paradox_penalty": paradox_penalty,
        "components": comp,
        "fun_optimal": fun_opt,
        "win_optimal": win_opt,
    }


# ═══════════════════════════════════════════════
# EXPERIMENT 1: Baseline vs Optimized Combat
# ═══════════════════════════════════════════════

def experiment_1_baseline_vs_optimized():
    """Compare CFS for baseline and CPG-optimized combat games.

    This is the headline test: does the optimized game score definitively
    higher across all metrics?
    """
    print("=" * 70)
    print("EXPERIMENT 1: Baseline vs CPG-Optimized Combat")
    print("=" * 70)
    print()

    # Baseline combat (HP=5, standard params)
    baseline = make_combat_game(max_hp=5)
    baseline.action_names = ["Strike", "Heavy", "Guard"]

    # CPG-optimized combat (HP=5, tuned params)
    optimized = make_parametric_combat(
        max_hp=5,
        heavy_dmg=3,
        heavy_hit_prob=0.7,
        guard_counter=2,
        guard_vs_heavy_block=0.7,
    )
    optimized.action_names = ["Strike", "Heavy", "Guard"]

    games = {"Baseline": baseline, "Optimized": optimized}

    for alpha in [0.5, 0.7, 1.0]:
        print(f"  α = {alpha}:")
        print(f"  {'Metric':<20} {'Baseline':>10} {'Optimized':>10} {'Δ':>10}")
        print("  " + "-" * 55)

        for name, game in games.items():
            result = compute_composite_fun_score(game, alpha=alpha, nest_level=10, cpg_resolution=20)
            if name == "Baseline":
                baseline_r = result
            else:
                optimized_r = result

        metrics = [
            ("GDS", "gds"),
            ("wGDS(α)", "wgds"),
            ("PI", "pi"),
            ("PI/GDS", "pi_ratio"),
            ("CPG", "cpg"),
            ("Agency Amp", "agency_amp"),
            ("Paradox Penalty", "paradox_penalty"),
            ("CFS", "cfs"),
        ]

        for label, key in metrics:
            b = baseline_r[key]
            o = optimized_r[key]
            delta = o - b
            pct = (delta / b * 100) if abs(b) > 1e-10 else float('inf')
            print(f"  {label:<20} {b:>10.4f} {o:>10.4f} {delta:>+10.4f} ({pct:>+.0f}%)")

        print()

    # Fun-optimal strategies
    print("  Fun-optimal strategies:")
    print(f"    Baseline: {baseline_r['fun_optimal'][2]} → D₀={baseline_r['fun_optimal'][1]:.3f}")
    print(f"    Optimized: {optimized_r['fun_optimal'][2]} → D₀={optimized_r['fun_optimal'][1]:.3f}")
    print()
    print("  Win-optimal strategies:")
    print(f"    Baseline: {baseline_r['win_optimal'][2]} → D₀={baseline_r['win_optimal'][1]:.3f}")
    print(f"    Optimized: {optimized_r['win_optimal'][2]} → D₀={optimized_r['win_optimal'][1]:.3f}")
    print()

    return baseline_r, optimized_r


# ═══════════════════════════════════════════════
# EXPERIMENT 2: Parametric CFS Landscape
# ═══════════════════════════════════════════════

def experiment_2_parametric_landscape():
    """Search the combat parameter space for highest CFS.

    Vary heavy_dmg, heavy_hit_prob, guard_counter to find the
    CFS-optimal game configuration.
    """
    print("=" * 70)
    print("EXPERIMENT 2: CFS Landscape — Parameter Search")
    print("=" * 70)
    print()

    results = []
    alpha = 0.5

    for heavy_dmg in [2, 3, 4]:
        for heavy_hit in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for guard_counter in [1, 2, 3]:
                game = make_parametric_combat(
                    max_hp=5,
                    heavy_dmg=heavy_dmg,
                    heavy_hit_prob=heavy_hit,
                    guard_counter=guard_counter,
                    guard_vs_heavy_block=0.5,
                )
                game.action_names = ["Strike", "Heavy", "Guard"]
                r = compute_composite_fun_score(game, alpha=alpha, nest_level=5, cpg_resolution=10)
                results.append({
                    "heavy_dmg": heavy_dmg,
                    "heavy_hit": heavy_hit,
                    "guard_counter": guard_counter,
                    **r,
                })

    # Sort by CFS
    results.sort(key=lambda x: -x["cfs"])

    print(f"  Top 10 configurations by CFS (α={alpha}):")
    print(f"  {'HD':>3} {'HH':>4} {'GC':>3} {'GDS':>6} {'wGDS':>6} {'PI':>6} {'CPG':>6} {'CFS':>8}")
    print("  " + "-" * 50)

    for r in results[:10]:
        print(f"  {r['heavy_dmg']:>3} {r['heavy_hit']:>4.1f} {r['guard_counter']:>3} "
              f"{r['gds']:>6.3f} {r['wgds']:>6.3f} {r['pi']:>6.3f} {r['cpg']:>6.3f} {r['cfs']:>8.4f}")

    print()
    print(f"  Worst 5 configurations:")
    print(f"  {'HD':>3} {'HH':>4} {'GC':>3} {'GDS':>6} {'wGDS':>6} {'PI':>6} {'CPG':>6} {'CFS':>8}")
    print("  " + "-" * 50)
    for r in results[-5:]:
        print(f"  {r['heavy_dmg']:>3} {r['heavy_hit']:>4.1f} {r['guard_counter']:>3} "
              f"{r['gds']:>6.3f} {r['wgds']:>6.3f} {r['pi']:>6.3f} {r['cpg']:>6.3f} {r['cfs']:>8.4f}")

    best = results[0]
    print()
    print(f"  CFS-optimal: HD={best['heavy_dmg']} HH={best['heavy_hit']:.1f} GC={best['guard_counter']}")
    print(f"    CFS = {best['cfs']:.4f}")
    print(f"    GDS = {best['gds']:.4f}, wGDS = {best['wgds']:.4f}")
    print(f"    PI = {best['pi']:.4f}, PI/GDS = {best['pi_ratio']:.3f}")
    print(f"    CPG = {best['cpg']:.4f}")
    print()

    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 3: CFS Decomposition
# ═══════════════════════════════════════════════

def experiment_3_decomposition():
    """Decompose CFS into its three factors to understand relative importance.

    Which factor contributes most to CFS variance:
    - wGDS (base tension)?
    - Agency amplifier?
    - Paradox penalty?
    """
    print("=" * 70)
    print("EXPERIMENT 3: CFS Factor Decomposition")
    print("=" * 70)
    print()

    # Generate a range of game configs
    configs = []
    for heavy_dmg in [2, 3, 4]:
        for heavy_hit in [0.4, 0.6, 0.8]:
            for guard_counter in [1, 2, 3]:
                configs.append((heavy_dmg, heavy_hit, guard_counter))

    results = []
    alpha = 0.5

    for hd, hh, gc in configs:
        game = make_parametric_combat(
            max_hp=5, heavy_dmg=hd, heavy_hit_prob=hh,
            guard_counter=gc, guard_vs_heavy_block=0.5,
        )
        game.action_names = ["Strike", "Heavy", "Guard"]
        r = compute_composite_fun_score(game, alpha=alpha, nest_level=5, cpg_resolution=10)
        results.append(r)

    # Compute variance contribution of each factor
    cfss = [r["cfs"] for r in results]
    wgdss = [r["wgds"] for r in results]
    amps = [r["agency_amp"] for r in results]
    pens = [r["paradox_penalty"] for r in results]

    def variance(xs):
        mean = sum(xs) / len(xs)
        return sum((x - mean) ** 2 for x in xs) / len(xs)

    def coeff_var(xs):
        mean = sum(xs) / len(xs)
        if mean < 1e-10:
            return 0
        return math.sqrt(variance(xs)) / mean

    print(f"  Factor statistics across {len(results)} configurations:")
    print(f"  {'Factor':<20} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 55)

    for name, vals in [("CFS", cfss), ("wGDS(0.5)", wgdss), ("Agency Amp", amps), ("Paradox Penalty", pens)]:
        mean = sum(vals) / len(vals)
        std = math.sqrt(variance(vals))
        cv = coeff_var(vals)
        print(f"  {name:<20} {mean:>8.4f} {std:>8.4f} {cv:>8.3f} {min(vals):>8.4f} {max(vals):>8.4f}")

    print()
    print("  CV (coefficient of variation) shows relative variability.")
    print("  The factor with highest CV has the most discriminating power.")
    print()

    # Correlation analysis: which factor correlates most with CFS?
    def correlation(xs, ys):
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        if dx < 1e-10 or dy < 1e-10:
            return 0
        return num / (dx * dy)

    print("  Correlation with CFS:")
    for name, vals in [("wGDS(0.5)", wgdss), ("Agency Amp", amps), ("Paradox Penalty", pens)]:
        r = correlation(vals, cfss)
        print(f"    {name:<20} r = {r:.4f}")

    print()


# ═══════════════════════════════════════════════
# SYNTHESIS
# ═══════════════════════════════════════════════

def synthesis():
    """Phase 3 conclusions."""
    print("=" * 70)
    print("PHASE 3 SYNTHESIS: Composite Fun Score")
    print("=" * 70)
    print()
    print("The Composite Fun Score integrates all ToA research into one metric:")
    print()
    print("  CFS = wGDS(α) × (1 + PI/GDS) × (1 - CPG)")
    print()
    print("Where each component captures a distinct aspect of game quality:")
    print("  wGDS(α) — How much tension does the game structure create?")
    print("  PI/GDS  — How much control does the player have?")
    print("  CPG     — Does the fun strategy also win?")
    print()
    print("This is the first metric that simultaneously rewards:")
    print("  1. Rich uncertainty structure (high wGDS)")
    print("  2. Meaningful player agency (high PI)")
    print("  3. Fun-winning alignment (low CPG)")
    print()
    print("Previous research milestones that made this possible:")
    print("  Phase 0: GDS measures structural tension (baseline)")
    print("  Phase 1: PI measures agency; CPG measures fun-winning gap")
    print("  Phase 2: wGDS accounts for human perception limits")
    print("  Phase 3: CFS combines all three into a design optimization target")
    print()


if __name__ == "__main__":
    experiment_1_baseline_vs_optimized()
    experiment_2_parametric_landscape()
    experiment_3_decomposition()
    synthesis()
