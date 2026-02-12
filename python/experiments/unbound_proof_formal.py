"""Unbound Conjecture — Formal Proof via Component Structure Analysis.

Goal: Prove that GDS = Θ(N) for Best-of-N coin toss by showing:
1. A₁ contributes O(1) to GDS (decreasing per-state, bounded total)
2. A₂ contributes O(1) to GDS (nearly constant)
3. A_k for k ≥ 3 each contribute Θ(N) to GDS, with growth rate increasing by ~0.002 per level
4. The sum of infinitely many linear terms gives linear total growth

Key mathematical tools:
- Central Limit Theorem (for A₁ behavior)
- Self-similarity of the anticipation landscape
- Recursive structure of nested anticipation
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


# ── Utility functions ───────────────────────────────────────────────

def win_probability(w1, w2, target):
    """Exact win probability for player 1 at state (w1, w2)."""
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


def reach_probability(w1, w2):
    """Probability of reaching state (w1, w2) from (0, 0)."""
    n = w1 + w2
    return math.comb(n, w1) * (0.5 ** n)


def analyze_bestofn(target, nest_level=10):
    """Full ToA analysis of Best-of-(2*target-1)."""
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
        return 1.0 if state[0] >= target else 0.0

    return analyze(
        initial_state=initial_state(),
        is_terminal=is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=compute_desire,
        nest_level=nest_level,
    )


# ── Analysis 1: Component landscape structure ───────────────────────

def analyze_landscape(target, nest_level=8):
    """Extract the full A_k landscape for Best-of-(2*target-1).

    Returns dict: state -> [A₁, A₂, ..., A_nest_level]
    """
    result = analyze_bestofn(target, nest_level)
    landscape = {}
    for state in result.states:
        w1, w2 = state
        if w1 < target and w2 < target:
            landscape[state] = [result.state_nodes[state].a[k] for k in range(nest_level)]
    return landscape, result


def experiment_landscape_scaling():
    """How does each component's landscape scale with N?

    Key question: As N doubles, does the spatial pattern of A_k values
    simply scale, or does it change qualitatively?
    """
    print("=" * 70)
    print("EXPERIMENT 1: Component Landscape Scaling")
    print("=" * 70)

    targets = [4, 8, 12, 16, 20]

    for comp_idx in range(6):
        print(f"\n--- A{comp_idx+1} ---")
        print(f"{'Target':>8} {'Max':>10} {'Min>0':>10} {'Mean':>10} {'Std':>10} {'MaxState':>12}")

        for t in targets:
            landscape, _ = analyze_landscape(t, 8)
            vals = [landscape[s][comp_idx] for s in landscape]
            nonzero = [v for v in vals if v > 1e-10]

            if nonzero:
                max_val = max(nonzero)
                min_val = min(nonzero)
                mean_val = sum(nonzero) / len(nonzero)
                std_val = math.sqrt(sum((v - mean_val)**2 for v in nonzero) / len(nonzero))
                max_state = max(landscape.keys(), key=lambda s: landscape[s][comp_idx])
                print(f"{t:>8} {max_val:>10.6f} {min_val:>10.6f} {mean_val:>10.6f} {std_val:>10.6f} {str(max_state):>12}")
            else:
                print(f"{t:>8}  (all zero)")


# ── Analysis 2: Why A₂ is constant ─────────────────────────────────

def experiment_a2_mechanism():
    """Understand WHY A₂ ≈ 0.136 regardless of N.

    A₂ at state s = √Var(A₁(s') - A₁(s) | s→s')

    For Best-of-N at state (w₁, w₂):
    A₂(w₁,w₂) = |A₁(w₁+1,w₂) - A₁(w₁,w₂+1)| / 2^(1/2)

    No wait — it uses the GLOBAL desire propagation of A₁, not A₁ directly.
    Let me trace through the exact computation.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: A₂ Mechanism — Why Is It Constant?")
    print("=" * 70)

    for target in [5, 10, 15, 20]:
        print(f"\n--- Target = {target} (Best-of-{2*target-1}) ---")
        landscape, result = analyze_landscape(target, 8)

        # For A₂, the "desire" is A₁.
        # D_global for A₂ at state s = A₁(s) + Σ P(s→s') * D_global(s')
        # This propagates A₁ values backwards from terminals.

        # Show the diagonal states: A₁ values and how they change
        print(f"{'State':>10} {'A₁':>10} {'A₂':>10} {'A₃':>10} {'Reach':>10}")
        for k in range(min(target, 8)):
            s = (k, k)
            if s in landscape:
                rp = reach_probability(k, k)
                print(f"  ({k},{k})   {landscape[s][0]:>10.6f} {landscape[s][1]:>10.6f} {landscape[s][2]:>10.6f} {rp:>10.6f}")

        # Key: A₂ GDS component
        print(f"\n  GDS components: A₁={result.gds_components[0]:.4f}, A₂={result.gds_components[1]:.4f}, A₃={result.gds_components[2]:.4f}")


# ── Analysis 3: Recursive growth mechanism ──────────────────────────

def experiment_recursive_growth():
    """Show that each A_k+1 grows because A_k has MORE spatial variation with N.

    Hypothesis: The "landscape complexity" (spatial variation of A_k values)
    increases with N for each k, which drives growth of A_k+1.

    Measure: coefficient of variation (std/mean) of A_k values across states.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Recursive Growth — Landscape Complexity")
    print("=" * 70)

    targets = list(range(3, 22))

    print(f"\n{'Target':>8}", end="")
    for k in range(6):
        print(f" {'CV(A'+str(k+1)+')':>10}", end="")
    print()
    print("-" * 70)

    cv_data = {k: [] for k in range(6)}

    for t in targets:
        landscape, _ = analyze_landscape(t, 8)
        print(f"{t:>8}", end="")

        for comp_idx in range(6):
            vals = [landscape[s][comp_idx] for s in landscape if landscape[s][comp_idx] > 1e-10]
            if len(vals) > 1:
                mean_v = sum(vals) / len(vals)
                std_v = math.sqrt(sum((v - mean_v)**2 for v in vals) / len(vals))
                cv = std_v / mean_v if mean_v > 0 else 0
                cv_data[comp_idx].append((2*t-1, cv))
                print(f" {cv:>10.4f}", end="")
            else:
                cv_data[comp_idx].append((2*t-1, 0))
                print(f" {'---':>10}", end="")
        print()

    # Fit linear trends to CV
    print("\nLinear fit of CV vs N:")
    for comp_idx in range(6):
        pts = [(n, cv) for n, cv in cv_data[comp_idx] if cv > 0]
        if len(pts) >= 3:
            ns = [p[0] for p in pts]
            cvs = [p[1] for p in pts]
            n_pts = len(pts)
            sx = sum(ns)
            sy = sum(cvs)
            sxy = sum(ns[i] * cvs[i] for i in range(n_pts))
            sx2 = sum(x**2 for x in ns)
            denom = n_pts * sx2 - sx * sx
            if abs(denom) > 1e-10:
                slope = (n_pts * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n_pts
                print(f"  CV(A{comp_idx+1}) = {slope:.6f} * N + {intercept:.4f}")


# ── Analysis 4: The telescoping argument ────────────────────────────

def experiment_telescoping():
    """Attempt a telescoping proof structure.

    Define: L_k(N) = landscape complexity of A_k at game size N

    Claim: L_k(N) = f(L_{k-1}(N)), where f is a function that maps
    "less complex" to "more complex" landscapes, up to saturation.

    If L_k grows linearly with N for k ≥ some k₀, and the GDS contribution
    of A_k depends on L_k, then each component beyond k₀ adds Θ(N) to GDS.

    The total is bounded by the nest level, giving GDS = Θ(N * n_levels_active).
    If the number of active levels also grows with N... we need to check.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Active Components vs N")
    print("=" * 70)

    targets = list(range(3, 25))

    print(f"\n{'N':>6} {'GDS':>10}", end="")
    for k in range(8):
        print(f" {'A'+str(k+1):>8}", end="")
    print(f" {'Active':>8}")
    print("-" * 90)

    for t in targets:
        result = analyze_bestofn(t, 10)
        n = 2 * t - 1
        comps = result.gds_components[:8]
        active = sum(1 for c in comps if c > 0.001)
        print(f"{n:>6} {result.game_design_score:>10.4f}", end="")
        for c in comps:
            print(f" {c:>8.4f}", end="")
        print(f" {active:>8}")

    print("\nIf 'Active' grows with N, then GDS = Θ(N * log(N)) or similar.")
    print("If 'Active' is bounded, then GDS = Θ(N).")


# ── Analysis 5: Exact component formulas ────────────────────────────

def experiment_exact_formulas():
    """Try to find exact closed-form expressions for GDS components.

    For Best-of-N with fair coins:
    - States: (w₁, w₂) with 0 ≤ w₁, w₂ < target
    - P(w₁,w₂) = C(w₁+w₂, w₁) * 2^-(w₁+w₂) * I(w₁<target, w₂<target)
                  where I is the regularized incomplete beta function... complex.

    Simpler approach: analyze the GDS formula directly.

    GDS_k = Σ (over terminal states t) Σ (over paths p to t)
            [Π(prob along p) * Σ(A_k at each state on p) / |p|]

    For binary coin toss, each path has prob = 2^(-|p|), and there are
    C(|p|, w₁) paths to terminal (w₁, target) or (target, w₂).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: GDS Component Structure")
    print("=" * 70)

    # For small targets, enumerate the exact contribution of each state to GDS
    for target in [4, 8]:
        print(f"\n--- Target = {target} ---")
        landscape, result = analyze_landscape(target, 6)

        # Each state contributes to GDS through all paths that pass through it.
        # The exact contribution = Σ_paths (reach_prob * a_k * exit_prob / path_length)
        # For our engine, this is computed via forward propagation.

        # Let's compute the "GDS contribution" of each state for each component
        # GDS_contribution(s, k) = Σ_{paths through s} prob(path) * A_k(s) / len(path)

        # Approximation: contribution ≈ reach_prob(s) * A_k(s) * expected_remaining_weight
        # where expected_remaining_weight accounts for how the A_k value at s
        # gets divided by the total path length at each terminal.

        print(f"{'State':>10} {'Reach':>10} {'A₁':>8} {'A₂':>8} {'A₃':>8} {'A₁*Rch':>10} {'A₂*Rch':>10}")
        total_weighted = [0.0] * 6
        for w1 in range(target):
            for w2 in range(target):
                s = (w1, w2)
                rp = reach_probability(w1, w2)
                a_vals = landscape[s]
                if w1 + w2 < 4:  # Only show first few rounds
                    print(f"  ({w1},{w2})   {rp:>10.6f} {a_vals[0]:>8.4f} {a_vals[1]:>8.4f} {a_vals[2]:>8.4f} {rp*a_vals[0]:>10.6f} {rp*a_vals[1]:>10.6f}")
                for k in range(6):
                    total_weighted[k] += rp * a_vals[k]

        print(f"\n  Total weighted A_k (Σ reach * A_k):")
        for k in range(6):
            print(f"    A{k+1}: {total_weighted[k]:.6f} (GDS component: {result.gds_components[k]:.6f})")


# ── Analysis 6: The key mathematical relationship ──────────────────

def experiment_key_relationship():
    """Find the relationship: GDS_k ≈ f(total_weighted_A_k, N).

    If GDS_k ≈ c * total_weighted_A_k for some constant c,
    then proving total_weighted_A_k grows linearly suffices.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: GDS_k vs Total Weighted A_k")
    print("=" * 70)

    targets = list(range(3, 22))

    for comp_idx in range(5):
        print(f"\n--- Component A{comp_idx+1} ---")
        print(f"{'Target':>8} {'GDS_k':>10} {'Σ(rch*A_k)':>12} {'Ratio':>10} {'N':>6}")

        for t in targets:
            landscape, result = analyze_landscape(t, 8)
            total_w = sum(reach_probability(s[0], s[1]) * landscape[s][comp_idx]
                         for s in landscape)
            gds_k = result.gds_components[comp_idx]
            ratio = gds_k / total_w if total_w > 1e-10 else float('inf')
            n = 2 * t - 1
            print(f"{t:>8} {gds_k:>10.6f} {total_w:>12.6f} {ratio:>10.4f} {n:>6}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("UNBOUND CONJECTURE — FORMAL PROOF ANALYSIS")
    print("=" * 70)
    print("Goal: Prove GDS = Θ(N) for Best-of-N fair coin toss\n")

    experiment_landscape_scaling()
    experiment_a2_mechanism()
    experiment_recursive_growth()
    experiment_telescoping()
    experiment_exact_formulas()
    experiment_key_relationship()

    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print("""
Summary of findings:

1. A₁ landscape: max A₁ ~ 1/√N (CLT), spatial pattern self-similar.
   GDS contribution of A₁ decreases or stays bounded.

2. A₂ landscape: nearly constant GDS contribution (~0.136).
   A₂ at each state measures variation of A₁ among children.
   Since A₁'s spatial structure is self-similar, A₂ is scale-invariant.

3. A₃+ landscapes: GDS contribution grows linearly with N.
   Each component's landscape has increasing spatial variation as N grows.
   This increasing variation is captured by the next component.

4. The number of "active" components (contributing > 0.001 to GDS)
   appears to grow with N, suggesting each new component eventually
   starts contributing.

The formal proof requires showing:
- For k ≥ 3: Σ_{states s} reach(s) * A_k(s) = Θ(N)
- The GDS formula converts this to Θ(1) per component (division by path length ~N)
  BUT the accumulation along paths adds a factor... need to trace carefully.
""")


if __name__ == "__main__":
    main()
