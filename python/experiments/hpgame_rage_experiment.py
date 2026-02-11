"""HpGame_Rage experiments: verify against paper values + optimize critical chance.

Paper reference values:
- HpGame baseline GDS: 0.430 (A₁~A₅)
- HpGame_Rage (10% crit) GDS: 0.544 (A₁~A₅), improvement: +26.5%
- Optimal crit chance: 13%, GDS: 0.551
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.hpgame import HpGame
from toa.games.hpgame_rage import HpGameRage


def verify_baseline():
    """Verify HpGame baseline GDS matches expected 0.430."""
    print("=" * 70)
    print("1. HpGame Baseline Verification")
    print("=" * 70)

    result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )

    print(f"States: {len(result.states)}")
    print(f"GDS (A₁~A₅): {result.game_design_score:.6f}")
    print(f"Components: {' '.join(f'A{i+1}:{result.gds_components[i]:.4f}' for i in range(5))}")
    print(f"Expected: ~0.430")
    print(f"Match: {'YES' if abs(result.game_design_score - 0.430) < 0.005 else 'NO'}")

    # Top engaging states
    print("\nTop 10 most engaging states (by sum A₁~A₅):")
    state_engagement = []
    for s in result.states:
        node = result.state_nodes[s]
        total_a = sum(node.a[:5])
        if total_a > 0.01 and not HpGame.is_terminal(s):
            state_engagement.append((s, node.a[:5], total_a))

    state_engagement.sort(key=lambda x: x[2], reverse=True)
    print(f"{'State':>15}  {'A₁':>6}  {'A₂':>6}  {'A₃':>6}  {'A₄':>6}  {'A₅':>6}  {'Sum':>6}")
    for s, components, total in state_engagement[:10]:
        print(f"{HpGame.tostr(s):>15}  {components[0]:>6.3f}  {components[1]:>6.3f}  "
              f"{components[2]:>6.3f}  {components[3]:>6.3f}  {components[4]:>6.3f}  {total:>6.3f}")

    return result.game_design_score


def verify_rage():
    """Verify HpGame_Rage GDS matches expected 0.544."""
    print("\n" + "=" * 70)
    print("2. HpGame_Rage Verification (10% crit)")
    print("=" * 70)

    config = HpGameRage.Config(critical_chance=0.10)

    t0 = time.time()
    result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )
    elapsed = time.time() - t0

    print(f"States: {len(result.states)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"GDS (A₁~A₅): {result.game_design_score:.6f}")
    print(f"Components: {' '.join(f'A{i+1}:{result.gds_components[i]:.4f}' for i in range(5))}")
    print(f"Expected: ~0.544")
    print(f"Match: {'YES' if abs(result.game_design_score - 0.544) < 0.01 else 'APPROX' if abs(result.game_design_score - 0.544) < 0.05 else 'NO'}")

    # Top engaging states
    print("\nTop 15 most engaging states (by sum A₁~A₅):")
    state_engagement = []
    for s in result.states:
        node = result.state_nodes[s]
        total_a = sum(node.a[:5])
        if total_a > 0.01 and not HpGameRage.is_terminal(s):
            state_engagement.append((s, node.a[:5], total_a))

    state_engagement.sort(key=lambda x: x[2], reverse=True)
    print(f"{'State':>30}  {'A₁':>6}  {'A₂':>6}  {'A₃':>6}  {'A₄':>6}  {'A₅':>6}  {'Sum':>6}")
    for s, components, total in state_engagement[:15]:
        print(f"{HpGameRage.tostr(s):>30}  {components[0]:>6.3f}  {components[1]:>6.3f}  "
              f"{components[2]:>6.3f}  {components[3]:>6.3f}  {components[4]:>6.3f}  {total:>6.3f}")

    return result.game_design_score


def optimize_critical_chance():
    """Find optimal critical hit chance by brute-force search."""
    print("\n" + "=" * 70)
    print("3. Critical Hit Chance Optimization")
    print("=" * 70)

    print(f"{'Crit%':>6}  {'GDS':>10}  {'States':>8}  {'Time(s)':>8}")
    print("-" * 40)

    results = []

    for crit_pct in range(0, 41):
        crit = crit_pct / 100.0
        config = HpGameRage.Config(critical_chance=crit)

        t0 = time.time()
        result = analyze(
            initial_state=HpGameRage.initial_state(),
            is_terminal=HpGameRage.is_terminal,
            get_transitions=HpGameRage.get_transitions,
            compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
            config=config,
            nest_level=5,
        )
        elapsed = time.time() - t0

        results.append((crit_pct, result.game_design_score, len(result.states), elapsed))
        print(f"{crit_pct:>5}%  {result.game_design_score:>10.6f}  {len(result.states):>8}  {elapsed:>8.2f}")

        # Safety
        if elapsed > 60:
            print("Stopping — too slow")
            break

    # Find optimal
    best = max(results, key=lambda x: x[1])
    print(f"\nOptimal critical chance: {best[0]}%")
    print(f"Best GDS: {best[1]:.6f}")
    print(f"Expected optimal: 13% with GDS ~0.551")

    return results


def compare_improvement():
    """Compare baseline vs rage and show improvement."""
    print("\n" + "=" * 70)
    print("4. Improvement Summary")
    print("=" * 70)

    baseline_gds = verify_baseline()
    rage_gds = verify_rage()

    improvement = (rage_gds - baseline_gds) / baseline_gds * 100
    print(f"\nBaseline GDS: {baseline_gds:.6f}")
    print(f"Rage GDS:     {rage_gds:.6f}")
    print(f"Improvement:  {improvement:+.1f}%")
    print(f"Expected:     +26.5%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify against paper values")
    parser.add_argument("--optimize", action="store_true", help="Run critical chance optimization")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.all or (not args.verify and not args.optimize):
        compare_improvement()
        optimize_critical_chance()
    elif args.verify:
        compare_improvement()
    elif args.optimize:
        optimize_critical_chance()
