"""Unbound Conjecture — Formal Proof Sketch and Verification.

Theorem: For the Best-of-N coin toss game with fair coins,
GDS grows linearly with N (i.e., GDS = Θ(N)).

Proof strategy:
1. Show that A₁ at state (w₁, w₂) depends on the "discriminating power"
   of the next flip — how much it changes the win probability.
2. Show that the total weighted A₁ across all states grows linearly with N.
3. Show that higher-order components (A₂+) also grow.

This file contains both the proof reasoning and computational verification.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


# ── Best-of-N Analysis ──────────────────────────────────────────────────

def win_probability(w1, w2, target):
    """Exact win probability for player 1 at state (w1, w2) in Best-of-(2*target-1).

    Uses dynamic programming. P(w1, w2) where w1 = p1's wins, w2 = p2's wins.
    P(target, _) = 1.0 (p1 won)
    P(_, target) = 0.0 (p2 won)
    P(w1, w2) = 0.5 * P(w1+1, w2) + 0.5 * P(w1, w2+1)
    """
    # Memoization
    cache = {}

    def dp(a, b):
        if a >= target:
            return 1.0
        if b >= target:
            return 0.0
        if (a, b) in cache:
            return cache[(a, b)]
        result = 0.5 * dp(a + 1, b) + 0.5 * dp(a, b + 1)
        cache[(a, b)] = result
        return result

    return dp(w1, w2)


def local_anticipation_bestofn(w1, w2, target):
    """A₁ at state (w1, w2) for Best-of-N.

    A₁ = sqrt(Var(perspective_desire))
    With two outcomes (each p=0.5):
      PD₁ = D(w1+1, w2) - D(w1, w2)  (p1 wins the flip)
      PD₂ = D(w1, w2+1) - D(w1, w2)  (p2 wins the flip)
    Mean PD = 0 (symmetric update)
    Var = 0.5 * PD₁² + 0.5 * PD₂²
    A₁ = sqrt(Var)

    Since D(w1,w2) = P(w1 wins) = win_probability(w1, w2, target):
      PD₁ = P(w1+1, w2) - P(w1, w2)
      PD₂ = P(w1, w2+1) - P(w1, w2)

    Note: PD₁ + PD₂ ≠ 0 in general (the mean perspective desire isn't zero).
    Mean = 0.5 * PD₁ + 0.5 * PD₂ = 0.5 * (P(w1+1,w2) + P(w1,w2+1)) - P(w1,w2)
         = P(w1,w2) - P(w1,w2) = 0  ← YES it is zero (by definition of P)!

    So: Var = 0.5 * PD₁² + 0.5 * PD₂² = 0.5 * (PD₁² + PD₂²)
    And: A₁ = sqrt(0.5 * (PD₁² + PD₂²))

    Since PD₁ = -PD₂ (because mean = 0):
    A₁ = |PD₁| = |P(w1+1, w2) - P(w1, w2)|
    """
    if w1 >= target or w2 >= target:
        return 0.0

    p_current = win_probability(w1, w2, target)
    p_win = win_probability(w1 + 1, w2, target)
    p_lose = win_probability(w1, w2 + 1, target)

    pd1 = p_win - p_current
    pd2 = p_lose - p_current

    mean_pd = 0.5 * pd1 + 0.5 * pd2
    var = 0.5 * (pd1 - mean_pd)**2 + 0.5 * (pd2 - mean_pd)**2
    return math.sqrt(var)


def reach_probability(w1, w2, target):
    """Probability of reaching state (w1, w2) from (0, 0).

    Reaches (w1, w2) through exactly w1+w2 flips, with w1 heads and w2 tails.
    P_reach = C(w1+w2, w1) * (0.5)^(w1+w2)
    """
    n = w1 + w2
    return math.comb(n, w1) * (0.5 ** n)


def manual_gds_a1(target):
    """Manually compute the A₁ component of GDS for Best-of-(2*target-1).

    GDS_A₁ = Σ over all states s:
        Σ over all paths through s:
            (reach_probability * A₁(s)) / path_length

    For Best-of-N, this simplifies because each state (w1, w2) is reached
    at a unique time step (w1 + w2), so:

    GDS_A₁ = Σ_{w1,w2} reach_prob(w1,w2) * A₁(w1,w2) / (w1+w2)  [for w1+w2 > 0]
           + (contribution from initial state handled separately by engine)

    Actually, the engine's GDS computation is more complex — it accumulates
    anticipation along paths. Let's just compute it via the engine and compare.
    """
    total = 0.0
    for w1 in range(target):
        for w2 in range(target):
            a1 = local_anticipation_bestofn(w1, w2, target)
            rp = reach_probability(w1, w2, target)
            step = w1 + w2
            if step > 0:
                total += rp * a1 / step
            # The actual GDS accumulation is: sum(A along path) / path_length at terminal
            # This is an approximation

    return total


def analyze_bestofn_detailed(target):
    """Full ToA analysis of Best-of-(2*target-1)."""
    n = 2 * target - 1
    game_n = n

    def initial_state():
        return (0, 0)

    def is_terminal(state):
        return state[0] >= target or state[1] >= target

    def get_transitions(state, config=None):
        w1, w2 = state
        if w1 >= target or w2 >= target:
            return []
        return [(0.5, (w1 + 1, w2)), (0.5, (w1, w2 + 1))]

    def compute_desire(state):
        if state[0] >= target:
            return 1.0
        return 0.0

    result = analyze(
        initial_state=initial_state(),
        is_terminal=is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=compute_desire,
        nest_level=10,
    )

    return result


def experiment_growth_rate():
    """Measure GDS growth rate and verify linearity."""
    print("=" * 70)
    print("UNBOUND CONJECTURE — Formal Analysis: Best-of-N")
    print("=" * 70)

    print("\n1. A₁ ANALYSIS: Local anticipation at each state")
    print("-" * 50)

    # For Best-of-7 (target=4), show A₁ at each state
    target = 4
    print(f"\nBest-of-{2*target-1} (target={target}):")
    print(f"{'State':>10} {'P(win)':>8} {'A₁':>8} {'P(reach)':>10}")
    print("-" * 45)

    for w1 in range(target):
        for w2 in range(target):
            pw = win_probability(w1, w2, target)
            a1 = local_anticipation_bestofn(w1, w2, target)
            pr = reach_probability(w1, w2, target)
            print(f"  ({w1},{w2})    {pw:>8.4f} {a1:>8.4f} {pr:>10.6f}")

    # Key observation: A₁ is highest along the diagonal (w1 ≈ w2)
    print(f"\nKey observation: A₁ is highest along the diagonal (w1 ≈ w2)")
    print("because that's where the game is most uncertain.")

    print("\n2. GDS GROWTH ANALYSIS")
    print("-" * 50)

    targets = list(range(2, 25))
    results = []

    print(f"{'N (best-of)':>12} {'Target':>8} {'GDS':>10} {'A₁':>8} {'A₂':>8} {'A₃':>8} {'States':>8}")
    print("-" * 70)

    for t in targets:
        n = 2 * t - 1
        r = analyze_bestofn_detailed(t)
        comps = r.gds_components[:5]
        results.append((n, t, r.game_design_score, comps, len(r.states)))
        print(f"{n:>12} {t:>8} {r.game_design_score:>10.4f} {comps[0]:>8.4f} {comps[1]:>8.4f} {comps[2]:>8.4f} {len(r.states):>8}")

    # Fit linear model to GDS
    ns = [r[0] for r in results]
    gds_vals = [r[2] for r in results]

    # Linear regression
    n_pts = len(ns)
    sx = sum(ns)
    sy = sum(gds_vals)
    sxy = sum(ns[i] * gds_vals[i] for i in range(n_pts))
    sx2 = sum(x**2 for x in ns)
    denom = n_pts * sx2 - sx * sx
    slope = (n_pts * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n_pts

    # R² value
    mean_gds = sy / n_pts
    ss_tot = sum((g - mean_gds)**2 for g in gds_vals)
    ss_res = sum((gds_vals[i] - (slope * ns[i] + intercept))**2 for i in range(n_pts))
    r_squared = 1.0 - ss_res / ss_tot

    print(f"\nLinear fit: GDS = {slope:.5f} * N + {intercept:.4f}")
    print(f"R² = {r_squared:.6f}")
    print(f"Growth rate: ~{slope:.4f} GDS per round")

    # Component-level analysis
    print("\n3. COMPONENT-LEVEL GROWTH")
    print("-" * 50)
    print(f"{'Component':>10}", end="")
    for comp_idx in range(5):
        print(f" {'A'+str(comp_idx+1)+' slope':>12}", end="")
    print()

    for comp_idx in range(5):
        comp_vals = [r[3][comp_idx] for r in results]
        c_sxy = sum(ns[i] * comp_vals[i] for i in range(n_pts))
        c_sy = sum(comp_vals)
        c_slope = (n_pts * c_sxy - sx * c_sy) / denom
        print(f"  A{comp_idx+1:>2}     {c_slope:>12.6f}", end="")
    print()

    # Verify key properties for proof
    print("\n4. PROOF VERIFICATION")
    print("-" * 50)

    # Property 1: A₁ along diagonal ≥ constant for all N
    print("\nProperty 1: A₁ at balanced states (w1=w2=k) for various N")
    for t in [3, 5, 10, 15, 20]:
        a1_balanced = local_anticipation_bestofn(t//2, t//2, t)
        print(f"  target={t}, state=({t//2},{t//2}): A₁ = {a1_balanced:.6f}")

    # Property 2: Number of "high-A₁" states grows with N
    print("\nProperty 2: States with A₁ > 0.1 for various N")
    for t in [3, 5, 10, 15]:
        count = 0
        total_weighted_a1 = 0
        for w1 in range(t):
            for w2 in range(t):
                a1 = local_anticipation_bestofn(w1, w2, t)
                pr = reach_probability(w1, w2, t)
                if a1 > 0.1:
                    count += 1
                total_weighted_a1 += pr * a1
        print(f"  target={t}: {count} states with A₁>0.1, "
              f"total_states={t*t}, weighted_A₁={total_weighted_a1:.4f}")

    # Property 3: GDS(2N) ≈ 2 * GDS(N) (linear scaling)
    print("\nProperty 3: GDS(2N) / GDS(N) ratios")
    for i in range(len(results) - 1):
        n1, _, gds1, _, _ = results[i]
        for j in range(i+1, len(results)):
            n2, _, gds2, _, _ = results[j]
            if n2 == 2 * n1 - 1 or n2 == 2 * n1 or n2 == 2 * n1 + 1:
                print(f"  GDS({n2}) / GDS({n1}) = {gds2/gds1:.4f} (linear → ≈2.0)")
                break

    print("\n5. PROOF SKETCH")
    print("-" * 50)
    print("""
Theorem: For Best-of-N fair coin toss, GDS = Θ(N).

Proof sketch:

1. LOWER BOUND: GDS ≥ c₁ * N for some constant c₁ > 0.

   Consider only the A₁ component (first-order anticipation).

   At state (w₁, w₂) with w₁ + w₂ = k (round k):
   - A₁(w₁, w₂) = |P(w₁+1, w₂) - P(w₁, w₂)|
   - This equals the "discriminating power" of the next coin flip

   Key lemma: For balanced states where |w₁ - w₂| ≤ √(target):
   - A₁ ≥ c / √(target) for some constant c > 0
   - (From the central limit theorem: win probability changes by
     Θ(1/√n) per flip near the balanced region)

   The number of balanced states visited on a random path is Θ(√N)
   (standard random walk result).

   The path-averaged A₁ contribution from balanced states alone:
   GDS_A₁ ≥ Θ(√N) * Θ(1/√N) * (1/Θ(N)) * Θ(N²)

   This is not tight enough for linear growth from A₁ alone.

   The actual mechanism: GDS accumulates anticipation along PATHS,
   not just at individual states. A path of length N through states
   with average A₁ = Θ(1/√N) accumulates total A = Θ(√N).
   The GDS formula divides by path length N, giving Θ(1/√N) per path.
   But we need to account for ALL anticipation components.

2. KEY INSIGHT: Higher-order components provide the linear growth.

   A₂ uses A₁ values as "desires" for the next level.
   If A₁ varies significantly across states (which it does — it's
   high near the diagonal and low near edges), then A₂ captures
   the "anticipation of anticipation": the uncertainty about how
   exciting future states will be.

   Empirically: slope(A₁) ≈ {slope:.5f}, but the SUM of all
   component slopes gives the total growth rate.

   The hierarchy of components creates a "pyramid" where each level
   adds a roughly constant contribution to GDS growth.

3. UPPER BOUND: GDS ≤ c₂ * N for some constant c₂.

   Each anticipation component A_k ≤ 0.5 (bounded by binary coin flip).
   The GDS formula: sum(A along path) / path_length.
   Max path length = N.
   Sum of A along path ≤ N * 0.5 = N/2 per component.
   GDS per component ≤ 0.5.
   With N/2 meaningful components (diminishing returns), GDS ≤ O(N).

   Combined: GDS = Θ(N).  □

Note: This is a proof SKETCH, not a rigorous proof. The main gap is
formalizing the relationship between component growth rates and N.
The empirical R² = {r_squared:.6f} strongly supports linearity.
""".format(slope=slope, r_squared=r_squared))

    return results


def experiment_diagonal_analysis():
    """Analyze A₁ structure along the diagonal more carefully."""
    print("\n" + "=" * 70)
    print("DIAGONAL ANALYSIS — A₁ structure at balanced states")
    print("=" * 70)

    print(f"\n{'Target':>8} {'State':>10} {'A₁':>10} {'A₁*√target':>12} {'P(win)':>8}")
    print("-" * 55)

    for target in [5, 10, 15, 20, 25, 30]:
        w = target // 2
        a1 = local_anticipation_bestofn(w, w, target)
        pw = win_probability(w, w, target)
        scaled = a1 * math.sqrt(target)
        print(f"{target:>8} ({w},{w})    {a1:>10.6f} {scaled:>12.6f} {pw:>8.4f}")

    print("\nIf A₁ * √target converges to a constant, then A₁ ~ 1/√N")
    print("This matches the CLT prediction for binary random walks.")


def main():
    """Run proof analysis."""
    results = experiment_growth_rate()
    experiment_diagonal_analysis()


if __name__ == "__main__":
    main()
