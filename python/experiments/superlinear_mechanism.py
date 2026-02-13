"""Superlinear Growth Mechanism — Why A_k ~ T^(k-1).

This experiment dissects the recursive amplification mechanism that causes
higher anticipation components to grow faster with game depth.

Key insight to prove:
- A₁ is the variance of desire across children → scales as ~T^0 (bounded)
- A₂ uses A₁ as desire seed. A₁ varies across states → A₂ captures this variation
- Each nesting level captures the "variation of the variation of the variation..."
- For a binomial lattice: the k-th order variation scales as T^((k-1)/2)

The goal: derive exact or near-exact formulas for Σ(reach × A_k) for each k.
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


def experiment_a_profile_by_diagonal():
    """Study how A_k values change along diagonals and anti-diagonals.

    In Best-of-T, state (w1, w2) lies on anti-diagonal d = w1 + w2 (depth)
    and diagonal δ = w1 - w2 (advantage).

    Hypothesis: A_k at depth d has a characteristic shape that amplifies
    with each nesting level.
    """
    print("=" * 70)
    print("A_k PROFILES ALONG DIAGONALS")
    print("=" * 70)

    target = 8  # Small enough to visualize fully
    result = analyze_bestofn(target, 10)

    # Group by anti-diagonal (depth)
    max_depth = 2 * target - 2
    print(f"\nTarget = {target}, max game length = {max_depth + 1}")

    for comp in range(5):  # A₁ through A₅
        print(f"\n--- A{comp + 1} values by state ---")
        print(f"{'(w1,w2)':>10} {'depth':>6} {'δ':>4} {'reach':>10} {'A_k':>10} {'rch×A_k':>10}")
        print("-" * 55)

        depth_totals = {}
        for state in result.states:
            w1, w2 = state
            if w1 >= target or w2 >= target:
                continue
            depth = w1 + w2
            delta = w1 - w2
            rp = reach_probability(w1, w2)
            a_val = result.state_nodes[state].a[comp]

            if depth not in depth_totals:
                depth_totals[depth] = {'sum_ra': 0.0, 'max_a': 0.0, 'count': 0}
            depth_totals[depth]['sum_ra'] += rp * a_val
            depth_totals[depth]['max_a'] = max(depth_totals[depth]['max_a'], a_val)
            depth_totals[depth]['count'] += 1

            if target <= 8:  # Only print individual states for small games
                print(f"  ({w1},{w2}){' ':>4} {depth:>6} {delta:>+4} {rp:>10.6f} {a_val:>10.6f} {rp * a_val:>10.6f}")

        print(f"\n  Depth summary for A{comp + 1}:")
        print(f"  {'depth':>6} {'Σ(rch×A)':>10} {'max(A)':>10} {'states':>7}")
        for d in sorted(depth_totals.keys()):
            info = depth_totals[d]
            print(f"  {d:>6} {info['sum_ra']:>10.6f} {info['max_a']:>10.6f} {info['count']:>7}")


def experiment_recursive_amplification():
    """Trace exactly how A₁ → A₂ → A₃ amplification works.

    For a specific state, show:
    - The A₁ values of its children
    - How d_global propagation of A₁ creates the "A₂ desire"
    - The variance of A₂ desire across children → A₂ value
    - And so on for A₃

    This makes the amplification mechanism concrete.
    """
    print("\n" + "=" * 70)
    print("RECURSIVE AMPLIFICATION TRACE")
    print("=" * 70)

    target = 5
    result = analyze_bestofn(target, 6)

    # Focus on the initial state (0,0)
    state = (0, 0)
    w1, w2 = state

    print(f"\nTarget = {target}")
    print(f"\nState (0,0) — children: (1,0) and (0,1)")
    print(f"By symmetry, these have identical A_k values.\n")

    print(f"{'State':>10}", end="")
    for k in range(6):
        print(f" {'A'+str(k+1):>10}", end="")
    print(f" {'d_global':>10}")
    print("-" * 80)

    # Show all states
    for s in result.states:
        sw1, sw2 = s
        if sw1 >= target or sw2 >= target:
            continue
        node = result.state_nodes[s]
        print(f"  ({sw1},{sw2}){' ':>4}", end="")
        for k in range(6):
            print(f" {node.a[k]:>10.6f}", end="")
        print(f" {node.d_global:>10.6f}")

    # Now trace the amplification for a specific asymmetric state
    print(f"\n--- Amplification at state (2,1) ---")
    s = (2, 1)
    if s in result.state_nodes:
        node = result.state_nodes[s]
        child_left = (3, 1)
        child_right = (2, 2)

        print(f"\nChildren: (3,1) and (2,2), P = 0.5 each")
        nl = result.state_nodes[child_left]
        nr = result.state_nodes[child_right]

        for k in range(5):
            print(f"\n  Component A{k + 1}:")
            print(f"    A{k + 1}(3,1) = {nl.a[k]:.6f}")
            print(f"    A{k + 1}(2,2) = {nr.a[k]:.6f}")
            print(f"    Ratio (3,1)/(2,2) = {nl.a[k] / nr.a[k]:.3f}" if nr.a[k] > 1e-10 else "    Ratio: N/A")
            print(f"    A{k + 1}(2,1) = {node.a[k]:.6f}")

            # Show the amplification: how much does the ratio grow?
            if k >= 1 and nl.a[k-1] > 1e-10 and nr.a[k-1] > 1e-10:
                prev_ratio = nl.a[k-1] / nr.a[k-1]
                curr_ratio = nl.a[k] / nr.a[k] if nr.a[k] > 1e-10 else float('inf')
                print(f"    Ratio amplification: {prev_ratio:.3f} → {curr_ratio:.3f}")


def experiment_exact_formulas():
    """Derive exact formulas for Σ(reach × A_k) for small k.

    Known: Σ(reach × A₂) = (T-1)/4

    Attempt: find exact formula for Σ(reach × A₃) and Σ(reach × A₄).
    """
    print("\n" + "=" * 70)
    print("EXACT FORMULA SEARCH")
    print("=" * 70)

    targets = list(range(3, 30))
    data = {k: [] for k in range(8)}

    for t in targets:
        result = analyze_bestofn(t, 10)
        for comp in range(8):
            total = 0.0
            for state in result.states:
                w1, w2 = state
                if w1 < t and w2 < t:
                    rp = reach_probability(w1, w2)
                    a_val = result.state_nodes[state].a[comp]
                    total += rp * a_val
            data[comp].append((t, total))

    # A₁: Σ(reach × A₁) — check if it's related to sqrt(T) or harmonic numbers
    print("\n--- A₁: Σ(reach × A₁) ---")
    print(f"{'T':>4} {'ΣrA₁':>12} {'√T/2':>10} {'ratio':>10} {'T^0.5':>10} {'ΣrA₁/T^0.5':>12}")
    for t, val in data[0]:
        sqrt_t = math.sqrt(t)
        print(f"{t:>4} {val:>12.6f} {sqrt_t/2:>10.6f} {val/(sqrt_t/2):>10.6f} {sqrt_t:>10.6f} {val/sqrt_t:>12.6f}")

    # A₂: already known to be (T-1)/4
    print("\n--- A₂: Σ(reach × A₂) = (T-1)/4 ✓ ---")
    for t, val in data[1][:5]:
        print(f"  T={t}: {val:.6f} vs (T-1)/4 = {(t-1)/4:.6f}, diff = {val - (t-1)/4:.2e}")

    # A₃: try various formulas
    print("\n--- A₃: Σ(reach × A₃) — pattern search ---")
    print(f"{'T':>4} {'ΣrA₃':>12} {'T²/16':>10} {'ratio':>10} {'(T-1)(T-2)/24':>15} {'ratio':>10}")
    for t, val in data[2]:
        t2_16 = t * t / 16
        tri = (t - 1) * (t - 2) / 24
        print(f"{t:>4} {val:>12.6f} {t2_16:>10.6f} {val/t2_16:>10.6f} {tri:>15.6f} {val/tri if tri > 0 else 0:>10.6f}")

    # A₄: try various formulas
    print("\n--- A₄: Σ(reach × A₄) — pattern search ---")
    print(f"{'T':>4} {'ΣrA₄':>12} {'T³/64':>10} {'ratio':>10} {'(T-1)³/64':>12} {'ratio':>10}")
    for t, val in data[3]:
        t3_64 = t**3 / 64
        t3_adj = (t - 1)**3 / 64
        print(f"{t:>4} {val:>12.6f} {t3_64:>10.6f} {val/t3_64:>10.6f} {t3_adj:>12.6f} {val/t3_adj if t3_adj > 0 else 0:>10.6f}")

    # General pattern: Σ(reach × A_k) ≈ c_k * T^(k-1)
    print("\n--- Power law coefficients ---")
    print(f"{'k':>4} {'exponent':>10} {'coefficient':>12}")
    for k in range(8):
        pts = [(t, v) for t, v in data[k] if v > 1e-6 and t >= 5]
        if len(pts) < 5:
            continue
        # Fit log(v) = b * log(t) + log(a)
        log_t = [math.log(p[0]) for p in pts]
        log_v = [math.log(p[1]) for p in pts]
        n = len(pts)
        sx = sum(log_t); sy = sum(log_v)
        sxy = sum(log_t[i] * log_v[i] for i in range(n))
        sx2 = sum(x**2 for x in log_t)
        denom = n * sx2 - sx * sx
        b = (n * sxy - sx * sy) / denom
        log_a = (sy - b * sx) / n
        a = math.exp(log_a)
        print(f"  A{k+1}: T^{b:.3f},  c = {a:.6f}")


def experiment_a1_structure():
    """Deep dive into A₁ structure on the binomial lattice.

    A₁(w1,w2) = |P(win|w1+1,w2) - P(win|w1,w2+1)| / 2
    where P(win|w1,w2) = probability of P1 winning from state (w1,w2).

    Actually: A₁(s) = sqrt(Var[D_perspective])
    For 2 transitions with P=0.5:
    A₁ = |D_global(child_L) - D_global(child_R)| / 2

    But D_global = P(win), so:
    A₁(w1,w2) = |P(w1+1,w2) - P(w1,w2+1)| / 2

    This is exactly half the "probability jump" caused by the outcome.
    """
    print("\n" + "=" * 70)
    print("A₁ STRUCTURE: THE PROBABILITY JUMP")
    print("=" * 70)

    for target in [5, 10, 15]:
        result = analyze_bestofn(target, 3)
        print(f"\n--- Target = {target} ---")

        # Compute A₁ analytically and compare
        # First, get P(win) = d_global for each state
        d = {}
        for state in result.states:
            d[state] = result.state_nodes[state].d_global

        print(f"{'(w1,w2)':>10} {'P(win)':>10} {'A₁(engine)':>12} {'|ΔP|/2':>10} {'match':>7}")
        for state in result.states:
            w1, w2 = state
            if w1 >= target or w2 >= target:
                continue
            a1_engine = result.state_nodes[state].a[0]
            # Compute analytically
            cl = (w1 + 1, w2)
            cr = (w1, w2 + 1)
            delta_p = abs(d.get(cl, 0) - d.get(cr, 0)) / 2
            match = abs(a1_engine - delta_p) < 1e-10
            if w1 + w2 <= 4 or (w1 == w2):
                print(f"  ({w1},{w2}){' ':>4} {d[state]:>10.6f} {a1_engine:>12.6f} {delta_p:>10.6f} {'✓' if match else '✗':>7}")


def experiment_variance_amplification():
    """The core mechanism: how variance amplifies through nesting.

    For Best-of-T with fair coin:
    - A₁(s) = |ΔP|/2 where ΔP = P(win|hit) - P(win|miss)
    - A₁ is maximized at diagonal states (w1=w2) where the outcome matters most
    - A₁ varies smoothly across the lattice

    For A₂:
    - Desire seed = A₁
    - D_global(A₂) = A₁(s) + 0.5*D_global(child_L) + 0.5*D_global(child_R)
    - A₂(s) = |D_global(A₂, child_L) - D_global(A₂, child_R)| / 2
    - This captures how much A₁ VARIES between the two possible futures

    The key: A₁ has a "hill" shape (peaked at diagonal).
    Going left vs right moves you to different slopes of this hill.
    The DIFFERENCE in slopes = A₂.
    A₂ is the "curvature" of A₁.

    Then A₃ is the curvature of A₂, and so on.
    Each level of curvature adds one power of the lattice scale.
    """
    print("\n" + "=" * 70)
    print("VARIANCE AMPLIFICATION: CURVATURE INTERPRETATION")
    print("=" * 70)

    target = 10
    result = analyze_bestofn(target, 6)

    # Show A_k values along the main diagonal (w, w)
    print(f"\nA_k along diagonal (w,w) for target={target}:")
    print(f"{'w':>4}", end="")
    for k in range(6):
        print(f" {'A'+str(k+1):>10}", end="")
    print()
    print("-" * 65)

    for w in range(target):
        s = (w, w)
        if s in result.state_nodes:
            node = result.state_nodes[s]
            print(f"{w:>4}", end="")
            for k in range(6):
                print(f" {node.a[k]:>10.6f}", end="")
            print()

    # Show A_k along δ=1 line (w+1, w)
    print(f"\nA_k along δ=+1 line (w+1,w):")
    print(f"{'w':>4}", end="")
    for k in range(6):
        print(f" {'A'+str(k+1):>10}", end="")
    print()
    print("-" * 65)

    for w in range(target - 1):
        s = (w + 1, w)
        if s in result.state_nodes:
            node = result.state_nodes[s]
            print(f"{w:>4}", end="")
            for k in range(6):
                print(f" {node.a[k]:>10.6f}", end="")
            print()

    # The amplification factor: A_{k+1}(0,0) / A_k(0,0)
    node_00 = result.state_nodes[(0, 0)]
    print(f"\nAmplification at (0,0): A_{{k+1}} / A_k")
    for k in range(5):
        if node_00.a[k] > 1e-10:
            ratio = node_00.a[k + 1] / node_00.a[k]
            print(f"  A{k+2}/A{k+1} = {ratio:.6f}")


def experiment_growth_decomposition():
    """Decompose GDS growth into exact contributions.

    GDS = Σ_k GDS_k where GDS_k = path-averaged sum of A_k.

    For each k, track:
    - How many non-zero A_k states exist
    - The magnitude of A_k values
    - The reach-weighted total
    """
    print("\n" + "=" * 70)
    print("GDS GROWTH DECOMPOSITION")
    print("=" * 70)

    targets = list(range(3, 25))

    print(f"\n{'T':>4} {'GDS':>10}", end="")
    for k in range(6):
        print(f" {'GDS_'+str(k+1):>10}", end="")
    print(f" {'%A₁':>6} {'%A₂':>6} {'%A₃+':>6}")
    print("-" * 100)

    for t in targets:
        result = analyze_bestofn(t, 10)
        gds = result.game_design_score
        comps = result.gds_components

        pct_a1 = 100 * comps[0] / gds if gds > 0 else 0
        pct_a2 = 100 * comps[1] / gds if gds > 0 else 0
        pct_a3plus = 100 * sum(comps[2:10]) / gds if gds > 0 else 0

        print(f"{t:>4} {gds:>10.4f}", end="")
        for k in range(6):
            print(f" {comps[k]:>10.4f}", end="")
        print(f" {pct_a1:>5.1f}% {pct_a2:>5.1f}% {pct_a3plus:>5.1f}%")


def experiment_cross_game_verification():
    """Verify A_k ~ T^(k-1) pattern in HP Game and GoldGame.

    If the pattern holds across different game structures,
    it's a universal property of the nesting mechanism, not specific to Best-of-N.
    """
    print("\n" + "=" * 70)
    print("CROSS-GAME VERIFICATION OF SUPERLINEAR GROWTH")
    print("=" * 70)

    # HP Game (equivalent to Best-of-N by isomorphism)
    print("\n--- HP Game (HP = 3..12) ---")
    print(f"{'HP':>4} {'GDS':>10}", end="")
    for k in range(5):
        print(f" {'GDS_'+str(k+1):>10}", end="")
    print()
    print("-" * 65)

    for hp in range(3, 13):
        def make_hp(hp_val):
            def is_terminal(state):
                return state[0] <= 0 or state[1] <= 0
            def get_transitions(state, config=None):
                h1, h2 = state
                if h1 <= 0 or h2 <= 0:
                    return []
                return [
                    (1/3, (h1, h2 - 1)),      # P1 wins round
                    (1/3, (h1 - 1, h2 - 1)),   # draw
                    (1/3, (h1 - 1, h2)),        # P2 wins round
                ]
            def compute_desire(state):
                return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0
            return is_terminal, get_transitions, compute_desire
        is_t, get_t, comp_d = make_hp(hp)
        result = analyze(
            initial_state=(hp, hp),
            is_terminal=is_t,
            get_transitions=get_t,
            compute_intrinsic_desire=comp_d,
            nest_level=10,
        )
        gds = result.game_design_score
        print(f"{hp:>4} {gds:>10.4f}", end="")
        for k in range(5):
            print(f" {result.gds_components[k]:>10.4f}", end="")
        print()

    # GoldGame (different structure — geometric rewards)
    print("\n--- GoldGame (turns = 4..14) ---")
    print(f"{'Turns':>6} {'GDS':>10}", end="")
    for k in range(5):
        print(f" {'GDS_'+str(k+1):>10}", end="")
    print()
    print("-" * 70)

    for max_turns in range(4, 15):
        hit = 0.68
        miss = 1 - hit
        mult = 1.2
        pen = 1.0 / 1.2

        def make_gold(mt):
            def is_terminal(state):
                return state[2] >= mt
            def get_transitions(state, config=None):
                p1g, p2g, turn = state
                if turn >= mt:
                    return []
                transitions = []
                for p1r in range(2):
                    for p2r in range(2):
                        new_p1 = int(p1g * (mult if p1r else pen))
                        new_p2 = int(p2g * (mult if p2r else pen))
                        prob = (hit if p1r else miss) * (hit if p2r else miss)
                        transitions.append((prob, (new_p1, new_p2, turn + 1)))
                # Sanitize: merge duplicate states
                merged = {}
                for prob, s in transitions:
                    if s in merged:
                        merged[s] += prob
                    else:
                        merged[s] = prob
                return [(p, s) for s, p in merged.items()]
            def compute_desire(state):
                p1g, p2g, turn = state
                if turn < mt:
                    return 0.0
                return 1.0 if p1g > p2g else 0.0
            return is_terminal, get_transitions, compute_desire
        is_t, get_t, comp_d = make_gold(max_turns)
        try:
            result = analyze(
                initial_state=(1000, 1000, 0),
                is_terminal=is_t,
                get_transitions=get_t,
                compute_intrinsic_desire=comp_d,
                nest_level=10,
            )
            gds = result.game_design_score
            print(f"{max_turns:>6} {gds:>10.4f}", end="")
            for k in range(5):
                print(f" {result.gds_components[k]:>10.4f}", end="")
            print()
        except Exception as e:
            print(f"{max_turns:>6} ERROR: {e}")


def main():
    experiment_a1_structure()
    experiment_exact_formulas()
    experiment_variance_amplification()
    experiment_recursive_amplification()
    experiment_growth_decomposition()
    experiment_cross_game_verification()


if __name__ == "__main__":
    main()
