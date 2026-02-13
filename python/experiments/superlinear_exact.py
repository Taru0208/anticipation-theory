"""Exact Formula Search for Σ(reach × A_k).

Known: Σ(reach × A₂) = (T-1)/4 exactly.
Goal: find exact or near-exact formulas for A₃, A₄.

Approach: compute high-precision values for T=3..50 and look for
patterns in terms of T, H_T (harmonic numbers), combinatorial expressions.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


def analyze_bestofn(target, nest_level=10):
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


def reach_probability(w1, w2):
    n = w1 + w2
    return math.comb(n, w1) * (0.5 ** n)


def harmonic(n):
    """H_n = 1 + 1/2 + 1/3 + ... + 1/n"""
    return sum(1.0 / k for k in range(1, n + 1))


def experiment_a1_exact():
    """A₁: The exact formula.

    A₁(w,w) = C(2w, w) * 2^(-2w-1) for diagonal states.
    This is Catalan-like.

    Σ(reach × A₁) = Σ_{w1+w2<2T-1} reach(w1,w2) × A₁(w1,w2)

    For diagonal states (w,w): reach = C(2w,w) / 2^{2w}, A₁ = C(2w,w) / 2^{2w+1}
    So reach × A₁ = [C(2w,w)]² / 2^{4w+1}

    But off-diagonal states also contribute...

    Let's try: is Σ(reach × A₁) = Σ_{w=0}^{T-1} something?
    """
    print("=" * 70)
    print("A₁ EXACT FORMULA")
    print("=" * 70)

    targets = list(range(3, 40))
    vals = []

    for t in targets:
        result = analyze_bestofn(t, 3)
        total = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < t and w2 < t:
                rp = reach_probability(w1, w2)
                a1 = result.state_nodes[state].a[0]
                total += rp * a1
        vals.append((t, total))

    # Try: Σ(rch×A₁) = c * √T * something
    # From data: ≈ 0.56 * √T
    # More precisely, the ratio Σ/√T seems to converge
    print(f"\n{'T':>4} {'Σ(rch×A₁)':>12} {'Σ/√T':>10} {'Σ/√(T-0.5)':>12} {'Σ²/T':>10}")
    for t, v in vals:
        print(f"{t:>4} {v:>12.8f} {v/math.sqrt(t):>10.6f} {v/math.sqrt(t-0.5):>12.6f} {v*v/t:>10.6f}")

    # Is A₁ related to the central binomial coefficient?
    # C(2n,n) / 4^n ~ 1/√(πn)
    # Σ_{w=0}^{T-1} [C(2w,w)]²/4^{2w} ... hmm
    print(f"\n--- Catalan-like sum test ---")
    for t, v in vals[:15]:
        cat_sum = sum(math.comb(2*w, w)**2 / (4**(2*w)) for w in range(t))
        print(f"  T={t}: Σ(rch×A₁) = {v:.8f}, catalan_sum = {cat_sum:.8f}, ratio = {v/cat_sum:.6f}")


def experiment_a3_decomposition():
    """Decompose A₃ to find its structure.

    A₃ uses A₂ as desire seed.
    A₂ = 0 on diagonal (by symmetry).
    A₂ is large off-diagonal.

    So A₃ at state (w1,w2) depends on the variation of accumulated A₂ across children.
    Since A₂ = 0 on diagonal and nonzero off-diagonal, A₃ captures the
    "diagonal crossing" effect.

    Let's decompose Σ(reach × A₃) by diagonal offset.
    """
    print("\n" + "=" * 70)
    print("A₃ DECOMPOSITION BY DIAGONAL OFFSET")
    print("=" * 70)

    for target in [8, 12, 16, 20]:
        result = analyze_bestofn(target, 6)

        # Group by |w1 - w2|
        by_offset = {}
        for state in result.states:
            w1, w2 = state
            if w1 >= target or w2 >= target:
                continue
            offset = abs(w1 - w2)
            rp = reach_probability(w1, w2)
            a3 = result.state_nodes[state].a[2]  # A₃
            if offset not in by_offset:
                by_offset[offset] = 0.0
            by_offset[offset] += rp * a3

        total = sum(by_offset.values())
        print(f"\nTarget = {target}, total Σ(rch×A₃) = {total:.6f}")
        print(f"  {'|δ|':>4} {'Σ(rch×A₃)':>12} {'%total':>8}")
        for d in sorted(by_offset.keys()):
            pct = 100 * by_offset[d] / total if total > 0 else 0
            print(f"  {d:>4} {by_offset[d]:>12.6f} {pct:>7.1f}%")


def experiment_gds_component_ratios():
    """Study the ratios GDS_{k+1} / GDS_k as T grows.

    If GDS_k ~ T^{α_k}, then GDS_{k+1}/GDS_k ~ T^{α_{k+1} - α_k}.

    This gives us the "amplification factor" per nesting level.
    """
    print("\n" + "=" * 70)
    print("GDS COMPONENT RATIOS")
    print("=" * 70)

    targets = list(range(5, 30))

    print(f"\n{'T':>4}", end="")
    for k in range(7):
        print(f" {'GDS_'+str(k+1):>10}", end="")
    print()
    print("-" * 80)

    prev = None
    for t in targets:
        result = analyze_bestofn(t, 10)
        comps = result.gds_components

        print(f"{t:>4}", end="")
        for k in range(7):
            print(f" {comps[k]:>10.6f}", end="")
        print()

    # Now show ratios
    print(f"\nRatios GDS_{{k+1}} / GDS_k:")
    print(f"\n{'T':>4}", end="")
    for k in range(6):
        label = f"r{k+2}/{k+1}"
        print(f" {label:>10}", end="")
    print()
    print("-" * 70)

    for t in targets:
        result = analyze_bestofn(t, 10)
        comps = result.gds_components

        print(f"{t:>4}", end="")
        for k in range(6):
            if comps[k] > 1e-8:
                ratio = comps[k + 1] / comps[k]
                print(f" {ratio:>10.4f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()


def experiment_recursive_formula():
    """Test if there's a recursive relationship between Σ(rch×A_k) terms.

    Hypothesis: Σ(rch×A_{k+1}) = f(T) × Σ(rch×A_k)
    where f(T) is some function of T (possibly linear or √T).
    """
    print("\n" + "=" * 70)
    print("RECURSIVE RELATIONSHIP TEST")
    print("=" * 70)

    targets = list(range(5, 30))
    all_data = {k: [] for k in range(8)}

    for t in targets:
        result = analyze_bestofn(t, 10)
        for k in range(8):
            total = 0.0
            for state in result.states:
                w1, w2 = state
                if w1 < t and w2 < t:
                    rp = reach_probability(w1, w2)
                    a_val = result.state_nodes[state].a[k]
                    total += rp * a_val
            all_data[k].append((t, total))

    # Compute ratios Σ(rch×A_{k+1}) / Σ(rch×A_k)
    print(f"\nRatios Σ(rch×A_{{k+1}}) / Σ(rch×A_k):")
    print(f"\n{'T':>4}", end="")
    for k in range(6):
        label = f"A{k+2}/A{k+1}"
        print(f" {label:>10}", end="")
    print()
    print("-" * 70)

    for i, t in enumerate(targets):
        print(f"{t:>4}", end="")
        for k in range(6):
            a_k = all_data[k][i][1]
            a_k1 = all_data[k+1][i][1]
            if a_k > 1e-8:
                ratio = a_k1 / a_k
                print(f" {ratio:>10.4f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # Now normalize: ratio / T
    print(f"\nNormalized ratios (ratio / T):")
    print(f"\n{'T':>4}", end="")
    for k in range(6):
        label = f"r/T"
        print(f" {label:>10}", end="")
    print()
    print("-" * 70)

    for i, t in enumerate(targets):
        print(f"{t:>4}", end="")
        for k in range(6):
            a_k = all_data[k][i][1]
            a_k1 = all_data[k+1][i][1]
            if a_k > 1e-8:
                ratio = a_k1 / (a_k * t)
                print(f" {ratio:>10.6f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()


def experiment_incremental_contribution():
    """For each new state added when T increases by 1, what's its contribution?

    When T → T+1, new states appear: (T, w) for w=0..T and (w, T) for w=0..T-1.
    (The T→T+1 boundary moves.)

    Actually, the structure changes more fundamentally — old states get new
    transition probabilities because the "ceiling" moves.

    Instead: track Σ(rch×A_k)(T+1) - Σ(rch×A_k)(T) = marginal contribution.
    """
    print("\n" + "=" * 70)
    print("MARGINAL CONTRIBUTION: Δ(T) = Σ(rch×A_k)(T+1) - Σ(rch×A_k)(T)")
    print("=" * 70)

    targets = list(range(3, 30))
    all_data = {k: [] for k in range(6)}

    for t in targets:
        result = analyze_bestofn(t, 8)
        for k in range(6):
            total = 0.0
            for state in result.states:
                w1, w2 = state
                if w1 < t and w2 < t:
                    rp = reach_probability(w1, w2)
                    a_val = result.state_nodes[state].a[k]
                    total += rp * a_val
            all_data[k].append((t, total))

    print(f"\n{'T':>4}", end="")
    for k in range(6):
        label = f"Δ(A{k+1})"
        print(f" {label:>10}", end="")
    print()
    print("-" * 70)

    for i in range(1, len(targets)):
        t = targets[i]
        print(f"{t:>4}", end="")
        for k in range(6):
            delta = all_data[k][i][1] - all_data[k][i-1][1]
            print(f" {delta:>10.6f}", end="")
        print()

    # Check if Δ(A₂) = 0.25 (constant, since A₂ total = (T-1)/4)
    print(f"\nΔ(A₂) should be exactly 0.25 (since Σ(rch×A₂) = (T-1)/4):")
    for i in range(1, min(6, len(targets))):
        delta = all_data[1][i][1] - all_data[1][i-1][1]
        print(f"  T={targets[i]}: Δ(A₂) = {delta:.10f}")


def main():
    experiment_a1_exact()
    experiment_a3_decomposition()
    experiment_gds_component_ratios()
    experiment_recursive_formula()
    experiment_incremental_contribution()


if __name__ == "__main__":
    main()
