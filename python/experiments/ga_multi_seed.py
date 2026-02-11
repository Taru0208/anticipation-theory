"""Multi-seed GA analysis: verify that findings are robust across random seeds.

Also tests fixed HP values to see how GDS scales with game length.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genetic_optimizer_v2 import (
    SymmetricGenome, run, evaluate, random_genome, mutate,
    crossover, genome_to_functions, describe
)
from toa.engine import analyze
import random
import time


def multi_seed_experiment():
    """Run GA with multiple seeds to verify robustness."""
    print("Multi-Seed Robustness Test")
    print("=" * 70)

    results = []
    for seed in [42, 123, 456, 789, 1024, 2048, 3333, 7777]:
        t0 = time.time()
        best, fitness, _ = run(
            pop_size=80,
            generations=40,
            elite=8,
            seed=seed,
            verbose=False,
        )
        elapsed = time.time() - t0

        outcomes = best.get_outcomes()
        init_fn, is_term, _, _ = genome_to_functions(best)

        results.append({
            "seed": seed,
            "gds": fitness,
            "hp": best.max_hp,
            "outcomes": len(outcomes),
            "time": elapsed,
        })

        print(f"  Seed {seed:>5}: GDS={fitness:.4f}, HP={best.max_hp}, "
              f"Outcomes={len(outcomes)}, Time={elapsed:.1f}s")

    gds_values = [r["gds"] for r in results]
    print(f"\n  Mean GDS: {sum(gds_values)/len(gds_values):.4f}")
    print(f"  Min GDS:  {min(gds_values):.4f}")
    print(f"  Max GDS:  {max(gds_values):.4f}")
    print(f"  Std Dev:  {(sum((g - sum(gds_values)/len(gds_values))**2 for g in gds_values)/len(gds_values))**0.5:.4f}")

    hp_values = [r["hp"] for r in results]
    from collections import Counter
    hp_counts = Counter(hp_values)
    print(f"  HP distribution: {dict(hp_counts)}")


def fixed_hp_experiment():
    """Run GA with fixed HP values to see how game length affects GDS."""
    print()
    print("Fixed HP Experiment: GDS vs Game Length")
    print("=" * 70)
    print(f"{'HP':>4}  {'GDS':>8}  {'A₁':>8}  {'A₂':>8}  {'States':>8}  {'Time':>6}")
    print("-" * 50)

    for hp in [3, 4, 5, 6, 7, 8]:
        rng = random.Random(42)

        # Generate population with fixed HP
        population = []
        for _ in range(80):
            g = random_genome(rng)
            g.max_hp = hp
            population.append(g)

        best = None
        best_fitness = -1

        # Manual evolution with fixed HP
        for gen in range(40):
            fitness = [evaluate(g) for g in population]
            pairs = sorted(zip(fitness, population), key=lambda x: -x[0])
            f_sorted = [f for f, g in pairs]
            p_sorted = [g for f, g in pairs]

            if f_sorted[0] > best_fitness:
                best_fitness = f_sorted[0]
                best = p_sorted[0].copy()

            new_pop = [p_sorted[i].copy() for i in range(8)]
            while len(new_pop) < 80:
                parents = []
                for _ in range(2):
                    contestants = rng.sample(list(enumerate(f_sorted)), min(5, len(f_sorted)))
                    winner = max(contestants, key=lambda x: x[1])[0]
                    parents.append(p_sorted[winner])
                child = crossover(parents[0], parents[1], rng)
                child = mutate(child, rng, 0.3)
                child.max_hp = hp  # Force HP
                new_pop.append(child)
            population = new_pop

        # Analyze best
        init_fn, is_term, get_trans, desire = genome_to_functions(best)
        result = analyze(
            initial_state=init_fn(),
            is_terminal=is_term,
            get_transitions=get_trans,
            compute_intrinsic_desire=desire,
            nest_level=5,
        )

        print(f"{hp:>4}  {best_fitness:>8.4f}  {result.gds_components[0]:>8.4f}  "
              f"{result.gds_components[1]:>8.4f}  {len(result.states):>8}")

    print()
    print("  ANALYSIS:")
    print("  If GDS increases with HP → longer games are inherently more engaging")
    print("  If GDS decreases with HP → shorter intense games are more engaging")
    print("  If non-monotonic → there's an optimal game length")


if __name__ == "__main__":
    t0 = time.time()
    multi_seed_experiment()
    fixed_hp_experiment()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
