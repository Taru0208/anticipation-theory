"""Unbound Conjecture — Scaling Laws for Component Weights.

Key discovery from unbound_proof_formal.py:
- Σ(reach × A₂) grows EXACTLY as N/4 (= 0.25 * target)
- This suggests exact formulas exist for each component's total weight.

This experiment derives and verifies these scaling laws.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


def win_probability(w1, w2, target):
    cache = {}
    def dp(a, b):
        if a >= target: return 1.0
        if b >= target: return 0.0
        if (a, b) in cache: return cache[(a, b)]
        result = 0.5 * dp(a + 1, b) + 0.5 * dp(a, b + 1)
        cache[(a, b)] = result
        return result
    return dp(w1, w2)


def reach_probability(w1, w2):
    n = w1 + w2
    return math.comb(n, w1) * (0.5 ** n)


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


def experiment_total_weight_scaling():
    """Compute Σ(reach × A_k) for each k and fit power laws."""
    print("=" * 70)
    print("TOTAL WEIGHT SCALING: Σ(reach × A_k) vs target")
    print("=" * 70)

    targets = list(range(3, 25))
    weight_data = {k: [] for k in range(8)}

    print(f"\n{'T':>4} {'N':>4}", end="")
    for k in range(8):
        print(f" {'ΣrA'+str(k+1):>10}", end="")
    print()
    print("-" * 95)

    for t in targets:
        result = analyze_bestofn(t, 10)
        n = 2 * t - 1
        print(f"{t:>4} {n:>4}", end="")

        for comp_idx in range(8):
            total_w = 0.0
            for state in result.states:
                w1, w2 = state
                if w1 < t and w2 < t:
                    rp = reach_probability(w1, w2)
                    a_val = result.state_nodes[state].a[comp_idx]
                    total_w += rp * a_val
            weight_data[comp_idx].append((t, total_w))
            print(f" {total_w:>10.4f}", end="")
        print()

    # Fit: Σ(reach × A_k) = a * T^b
    print("\n\nPower law fits: Σ(reach × A_k) ≈ a * T^b")
    print("-" * 50)

    for comp_idx in range(8):
        pts = [(t, w) for t, w in weight_data[comp_idx] if w > 1e-6]
        if len(pts) < 3:
            print(f"  A{comp_idx+1}: insufficient data")
            continue

        # Log-log regression
        log_t = [math.log(p[0]) for p in pts]
        log_w = [math.log(p[1]) for p in pts]
        n_pts = len(pts)
        sx = sum(log_t)
        sy = sum(log_w)
        sxy = sum(log_t[i] * log_w[i] for i in range(n_pts))
        sx2 = sum(x**2 for x in log_t)
        denom = n_pts * sx2 - sx * sx
        b = (n_pts * sxy - sx * sy) / denom
        log_a = (sy - b * sx) / n_pts
        a = math.exp(log_a)

        # R² in log space
        mean_ly = sy / n_pts
        ss_tot = sum((ly - mean_ly)**2 for ly in log_w)
        ss_res = sum((log_w[i] - (b * log_t[i] + log_a))**2 for i in range(n_pts))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Also try linear fit: Σ(reach × A_k) = m * T + c
        ts = [p[0] for p in pts]
        ws = [p[1] for p in pts]
        sx_l = sum(ts)
        sy_l = sum(ws)
        sxy_l = sum(ts[i] * ws[i] for i in range(n_pts))
        sx2_l = sum(x**2 for x in ts)
        denom_l = n_pts * sx2_l - sx_l * sx_l
        m = (n_pts * sxy_l - sx_l * sy_l) / denom_l
        c = (sy_l - m * sx_l) / n_pts
        mean_w = sy_l / n_pts
        ss_tot_l = sum((w - mean_w)**2 for w in ws)
        ss_res_l = sum((ws[i] - (m * ts[i] + c))**2 for i in range(n_pts))
        r_sq_l = 1 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 0

        print(f"  A{comp_idx+1}: power law T^{b:.3f} (R²={r_sq:.6f}), linear {m:.4f}*T + {c:.4f} (R²={r_sq_l:.6f})")


def experiment_gds_path_averaging():
    """Understand the GDS path-averaging effect.

    GDS_k = Σ_{terminal t} Σ_{paths p to t} prob(p) * [Σ_{s in p} A_k(s)] / |p|

    The key: A_k values are ACCUMULATED along paths and then divided by path length.
    This is different from just Σ(reach × A_k).

    For Best-of-N, all paths have length between target and 2*target-1.
    The average path length is known: for Best-of-(2T-1), E[length] = T * H_T
    where H_T is a harmonic-like sum (approximately T for large T, giving ~T²... no).

    Actually, E[length] for Best-of-(2T-1) ≈ T² / something... let me compute directly.
    """
    print("\n" + "=" * 70)
    print("PATH AVERAGING EFFECT")
    print("=" * 70)

    targets = list(range(3, 25))

    print(f"\n{'T':>4} {'N':>4} {'E[len]':>8} {'GDS':>10} {'ΣrA_tot':>10} {'GDS/ΣrA':>10} {'E[len]*GDS/ΣrA':>15}")
    print("-" * 70)

    for t in targets:
        result = analyze_bestofn(t, 10)
        n = 2 * t - 1

        # Compute expected path length
        # At each non-terminal state, the game continues by 1 step
        # E[length] = Σ reach(s) for non-terminal s (each state adds 1 step)
        e_len = 0.0
        total_weight_all = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < t and w2 < t:
                rp = reach_probability(w1, w2)
                e_len += rp
                for comp_idx in range(10):
                    total_weight_all += rp * result.state_nodes[state].a[comp_idx]

        gds = result.game_design_score
        ratio = gds / total_weight_all if total_weight_all > 0 else 0
        print(f"{t:>4} {n:>4} {e_len:>8.2f} {gds:>10.4f} {total_weight_all:>10.4f} {ratio:>10.6f} {e_len * ratio:>15.6f}")


def experiment_a2_exact():
    """Verify that Σ(reach × A₂) = T/4 exactly.

    A₂ at state (w₁,w₂) uses A₁ values as "desires."
    The global desire of A₁ propagates A₁ values backwards.

    For A₂:
    - desire_local(s) = A₁(s)
    - desire_global(s) = A₁(s) + Σ P(s→s') * desire_global(s')
    - A₂(s) = sqrt(Var_transitions(perspective_desire_global))

    On the diagonal (w,w): A₂ = 0 because the two children (w+1,w) and (w,w+1)
    are symmetric, so perspective desires are equal → variance = 0.
    """
    print("\n" + "=" * 70)
    print("A₂ EXACT FORMULA VERIFICATION")
    print("=" * 70)

    print(f"\n{'T':>4} {'Σ(rch×A₂)':>12} {'T/4':>10} {'Diff':>12}")
    print("-" * 45)

    for t in range(3, 30):
        result = analyze_bestofn(t, 4)
        total_w = 0.0
        for state in result.states:
            w1, w2 = state
            if w1 < t and w2 < t:
                rp = reach_probability(w1, w2)
                a2 = result.state_nodes[state].a[1]
                total_w += rp * a2

        expected = t / 4.0
        diff = total_w - expected
        print(f"{t:>4} {total_w:>12.8f} {expected:>10.4f} {diff:>12.2e}")


def experiment_component_gds_formula():
    """Find the exact relationship: GDS_k = f(T) for each k.

    From the data:
    - GDS(A₁) ~ 1/T (decreasing)
    - GDS(A₂) ~ constant ≈ 0.136
    - GDS(A₃) ~ c₃ * T + d₃ (linear, slope ~0.003)
    - GDS(A₄) ~ c₄ * T + d₄ (linear, slope ~0.005)
    - GDS(A₅) ~ c₅ * T + d₅ (linear, slope ~0.007)

    Pattern: slope(A_k) ≈ 0.001 + 0.002*(k-3) for k ≥ 3?
    Let's verify.
    """
    print("\n" + "=" * 70)
    print("GDS COMPONENT SLOPES")
    print("=" * 70)

    targets = list(range(3, 25))
    gds_data = {k: [] for k in range(10)}

    for t in targets:
        result = analyze_bestofn(t, 10)
        for k in range(10):
            gds_data[k].append((t, result.gds_components[k]))

    print(f"\n{'Comp':>6} {'Slope':>12} {'Intercept':>12} {'R²':>10} {'Slope pattern':>15}")
    print("-" * 60)

    slopes = []
    for k in range(10):
        pts = gds_data[k]
        ts = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        n_pts = len(pts)
        sx = sum(ts)
        sy = sum(vs)
        sxy = sum(ts[i] * vs[i] for i in range(n_pts))
        sx2 = sum(x**2 for x in ts)
        denom = n_pts * sx2 - sx * sx
        slope = (n_pts * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n_pts
        mean_v = sy / n_pts
        ss_tot = sum((v - mean_v)**2 for v in vs)
        ss_res = sum((vs[i] - (slope * ts[i] + intercept))**2 for i in range(n_pts))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        slopes.append(slope)

        pattern = ""
        if k >= 3 and slopes[k-1] > 0:
            delta = slope - slopes[k-1]
            pattern = f"Δ={delta:.6f}"

        print(f"  A{k+1:>2}   {slope:>12.6f} {intercept:>12.6f} {r_sq:>10.6f} {pattern:>15}")

    print("\nSlope differences (A_k+1 slope - A_k slope):")
    for k in range(2, 9):
        if slopes[k] > 0:
            diff = slopes[k+1] - slopes[k] if k+1 < 10 else 0
            print(f"  Δ(A{k+1}→A{k+2}) = {diff:.6f}")

    # Total GDS slope
    total_slope = sum(slopes[:10])
    print(f"\nTotal GDS slope (sum of component slopes): {total_slope:.6f}")


def main():
    experiment_total_weight_scaling()
    experiment_gds_path_averaging()
    experiment_a2_exact()
    experiment_component_gds_formula()


if __name__ == "__main__":
    main()
