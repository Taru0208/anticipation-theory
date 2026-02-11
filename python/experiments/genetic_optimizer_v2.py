"""Genetic Algorithm Game Optimizer v2 — Symmetric games only.

V1 found that asymmetric games can trivially score high by giving one player
an advantage. V2 constrains the search to symmetric games where the
probability structure is fair (P1 and P2 have equivalent chances).

A symmetric combat game: for every outcome (d1, d2) with probability p,
there exists an outcome (d2, d1) with the same probability. This ensures
the game is fair from both perspectives.

Additionally explores a richer mechanic space:
- Variable damage (1-3)
- Healing
- Asymmetric outcomes within a turn (one gains, other loses)
"""

import sys
import os
import math
import random
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


@dataclass
class SymmetricGenome:
    """A genome for a symmetric 1v1 combat game.

    Outcomes are organized as:
    - Symmetric pairs: (d1, d2) and (d2, d1) with equal probability
    - Neutral outcomes: (d, d) with their own probability

    This guarantees initial D_global = 0.5 (perfectly fair game).
    """
    max_hp: int = 5
    # Symmetric pairs: [(weight, d1, d2), ...] — each generates TWO outcomes
    sym_pairs: list[tuple[float, int, int]] = field(default_factory=list)
    # Neutral outcomes: [(weight, d), ...] — generates ONE outcome (d, d)
    neutrals: list[tuple[float, int]] = field(default_factory=list)

    def get_outcomes(self) -> list[tuple[float, int, int]]:
        """Convert to list of (probability, d1, d2)."""
        raw = []
        for w, d1, d2 in self.sym_pairs:
            raw.append((w, d1, d2))
            if d1 != d2:
                raw.append((w, d2, d1))
        for w, d in self.neutrals:
            raw.append((w, d, d))

        # Normalize probabilities
        total = sum(w for w, _, _ in raw)
        if total <= 0:
            return [(1.0, 0, 0)]
        return [(w / total, d1, d2) for w, d1, d2 in raw]

    def copy(self) -> "SymmetricGenome":
        return SymmetricGenome(
            max_hp=self.max_hp,
            sym_pairs=[(w, d1, d2) for w, d1, d2 in self.sym_pairs],
            neutrals=[(w, d) for w, d in self.neutrals],
        )


def genome_to_functions(genome: SymmetricGenome):
    """Convert genome to analyzer-compatible functions."""
    outcomes = genome.get_outcomes()
    max_hp = genome.max_hp

    def initial_state():
        return (max_hp, max_hp)

    def is_terminal(state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = []
        for prob, d1, d2 in outcomes:
            new1 = max(0, min(max_hp, hp1 + d1))
            new2 = max(0, min(max_hp, hp2 + d2))
            if prob > 1e-9:
                transitions.append((prob, (new1, new2)))
        return sanitize_transitions(transitions)

    def desire(state):
        return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0

    return initial_state, is_terminal, get_transitions, desire


def evaluate(genome: SymmetricGenome, nest_level: int = 5) -> float:
    """Evaluate a symmetric genome."""
    init_fn, is_term, get_trans, desire = genome_to_functions(genome)
    try:
        result = analyze(
            initial_state=init_fn(),
            is_terminal=is_term,
            get_transitions=get_trans,
            compute_intrinsic_desire=desire,
            nest_level=nest_level,
        )

        non_terminal = [s for s in result.states if not is_term(s)]
        if len(non_terminal) < 3:
            return -1.0

        # Verify fairness
        d0 = result.state_nodes[init_fn()].d_global
        if abs(d0 - 0.5) > 0.05:
            return result.game_design_score * 0.3  # Heavy penalty for unfairness

        return result.game_design_score
    except Exception:
        return -1.0


def random_genome(rng: random.Random) -> SymmetricGenome:
    """Generate a random symmetric genome."""
    max_hp = rng.randint(3, 8)
    n_pairs = rng.randint(1, 4)
    n_neutrals = rng.randint(0, 2)

    pairs = []
    for _ in range(n_pairs):
        w = rng.uniform(0.1, 3.0)
        d1 = rng.randint(-2, 1)
        d2 = rng.randint(-2, 1)
        pairs.append((w, d1, d2))

    neutrals = []
    for _ in range(n_neutrals):
        w = rng.uniform(0.1, 3.0)
        d = rng.randint(-2, 1)
        neutrals.append((w, d))

    return SymmetricGenome(max_hp=max_hp, sym_pairs=pairs, neutrals=neutrals)


def mutate(g: SymmetricGenome, rng: random.Random, rate: float = 0.3) -> SymmetricGenome:
    """Mutate a symmetric genome."""
    g = g.copy()

    if rng.random() < rate:
        g.max_hp = max(3, min(8, g.max_hp + rng.choice([-1, 0, 1])))

    for i in range(len(g.sym_pairs)):
        w, d1, d2 = g.sym_pairs[i]
        if rng.random() < rate:
            w += rng.gauss(0, 0.5)
            w = max(0.01, w)
        if rng.random() < rate:
            d1 = max(-3, min(1, d1 + rng.choice([-1, 0, 1])))
        if rng.random() < rate:
            d2 = max(-3, min(1, d2 + rng.choice([-1, 0, 1])))
        g.sym_pairs[i] = (w, d1, d2)

    for i in range(len(g.neutrals)):
        w, d = g.neutrals[i]
        if rng.random() < rate:
            w += rng.gauss(0, 0.5)
            w = max(0.01, w)
        if rng.random() < rate:
            d = max(-3, min(1, d + rng.choice([-1, 0, 1])))
        g.neutrals[i] = (w, d)

    # Add/remove pairs
    if rng.random() < rate * 0.3:
        if len(g.sym_pairs) < 5 and rng.random() < 0.5:
            g.sym_pairs.append((rng.uniform(0.1, 2.0), rng.randint(-2, 1), rng.randint(-2, 1)))
        elif len(g.sym_pairs) > 1:
            g.sym_pairs.pop(rng.randint(0, len(g.sym_pairs) - 1))

    # Add/remove neutrals
    if rng.random() < rate * 0.2:
        if len(g.neutrals) < 3 and rng.random() < 0.5:
            g.neutrals.append((rng.uniform(0.1, 2.0), rng.randint(-2, 0)))
        elif len(g.neutrals) > 0:
            g.neutrals.pop(rng.randint(0, len(g.neutrals) - 1))

    return g


def crossover(g1: SymmetricGenome, g2: SymmetricGenome, rng: random.Random) -> SymmetricGenome:
    """Crossover two symmetric genomes."""
    child = g1.copy()
    child.max_hp = rng.choice([g1.max_hp, g2.max_hp])

    # Mix pairs from both parents
    all_pairs = g1.sym_pairs + g2.sym_pairs
    n_select = max(1, min(5, rng.randint(1, len(all_pairs))))
    child.sym_pairs = [all_pairs[rng.randint(0, len(all_pairs)-1)] for _ in range(n_select)]

    # Mix neutrals
    all_neutrals = g1.neutrals + g2.neutrals
    if all_neutrals:
        n_select = rng.randint(0, min(3, len(all_neutrals)))
        child.neutrals = [all_neutrals[rng.randint(0, len(all_neutrals)-1)] for _ in range(n_select)]
    else:
        child.neutrals = []

    return child


def run(pop_size=120, generations=60, elite=12, tournament=5,
        mutation_rate=0.3, nest_level=5, seed=42, verbose=True):
    """Run symmetric GA evolution."""

    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(pop_size)]

    best_ever = None
    best_fitness = -1.0
    history = []

    if verbose:
        print("Symmetric Game Optimizer (v2)")
        print("=" * 75)
        print(f"Pop: {pop_size}, Gen: {generations}, Elite: {elite}, Nest: {nest_level}")
        print("=" * 75)
        print(f"{'Gen':>4}  {'Best':>8}  {'Avg':>8}  {'HP':>4}  {'Out':>4}  {'D₀':>6}  {'Time':>6}")
        print("-" * 50)

    for gen in range(generations):
        t0 = time.time()

        fitness = [evaluate(g, nest_level) for g in population]

        pairs = sorted(zip(fitness, population), key=lambda x: -x[0])
        f_sorted = [f for f, g in pairs]
        p_sorted = [g for f, g in pairs]

        if f_sorted[0] > best_fitness:
            best_fitness = f_sorted[0]
            best_ever = p_sorted[0].copy()

        valid = [f for f in f_sorted if f > 0]
        avg = sum(valid) / max(1, len(valid))
        elapsed = time.time() - t0

        # Get D₀ for best
        init_fn, is_term, get_trans, desire = genome_to_functions(p_sorted[0])
        try:
            r = analyze(
                initial_state=init_fn(),
                is_terminal=is_term,
                get_transitions=get_trans,
                compute_intrinsic_desire=desire,
                nest_level=1,
            )
            d0 = r.state_nodes[init_fn()].d_global
        except Exception:
            d0 = -1

        outcomes = p_sorted[0].get_outcomes()

        if verbose:
            print(f"{gen:>4}  {f_sorted[0]:>8.4f}  {avg:>8.4f}  {p_sorted[0].max_hp:>4}  "
                  f"{len(outcomes):>4}  {d0:>6.3f}  {elapsed:>5.1f}s")

        history.append({"gen": gen, "best": f_sorted[0], "avg": avg})

        # Next generation
        new_pop = [p_sorted[i].copy() for i in range(elite)]
        while len(new_pop) < pop_size:
            parents = []
            for _ in range(2):
                contestants = rng.sample(list(enumerate(f_sorted)), min(tournament, len(f_sorted)))
                winner = max(contestants, key=lambda x: x[1])[0]
                parents.append(p_sorted[winner])
            child = crossover(parents[0], parents[1], rng)
            child = mutate(child, rng, mutation_rate)
            new_pop.append(child)
        population = new_pop

    if verbose and best_ever:
        print()
        print("=" * 75)
        print("BEST SYMMETRIC GAME")
        print("=" * 75)
        describe(best_ever, nest_level)
        compare(best_ever, best_fitness, nest_level)

    return best_ever, best_fitness, history


def describe(genome: SymmetricGenome, nest_level: int = 5):
    """Print detailed description of a symmetric genome."""
    outcomes = genome.get_outcomes()

    print(f"  Max HP: {genome.max_hp}")
    print(f"  Total outcomes: {len(outcomes)}")
    print()
    print(f"  {'#':<4} {'Prob':>8} {'P1 Δ':>8} {'P2 Δ':>8}  Description")
    print(f"  {'-'*55}")

    for i, (p, d1, d2) in enumerate(outcomes):
        desc = ""
        if d1 < 0 and d2 < 0:
            desc = "mutual damage"
        elif d1 > 0 and d2 < 0:
            desc = "P1 advantage"
        elif d1 < 0 and d2 > 0:
            desc = "P2 advantage"
        elif d1 >= 0 and d2 >= 0:
            desc = "peaceful"
        elif d1 == 0:
            desc = "only P2 affected"
        elif d2 == 0:
            desc = "only P1 affected"
        print(f"  {i+1:<4} {p:>8.3f} {d1:>+8} {d2:>+8}  {desc}")

    init_fn, is_term, get_trans, desire = genome_to_functions(genome)
    result = analyze(
        initial_state=init_fn(),
        is_terminal=is_term,
        get_transitions=get_trans,
        compute_intrinsic_desire=desire,
        nest_level=nest_level,
    )

    non_terminal = [s for s in result.states if not is_term(s)]
    print()
    print(f"  GDS: {result.game_design_score:.6f}")
    comps = [f"A{i+1}={result.gds_components[i]:.4f}"
             for i in range(nest_level) if result.gds_components[i] > 0.001]
    print(f"  Components: {' '.join(comps)}")
    print(f"  States: {len(result.states)} ({len(non_terminal)} non-terminal)")
    print(f"  D₀ (fairness): {result.state_nodes[init_fn()].d_global:.4f} (0.5 = perfect)")

    # A₁ distribution
    a1_vals = [result.state_nodes[s].a[0] for s in non_terminal]
    print(f"  A₁ range: [{min(a1_vals):.4f}, {max(a1_vals):.4f}]")
    near_max = sum(1 for a in a1_vals if a > 0.4) / len(a1_vals) * 100
    boring = sum(1 for a in a1_vals if a < 0.1) / len(a1_vals) * 100
    print(f"  States near A₁ max (>0.4): {near_max:.1f}%")
    print(f"  Boring states (A₁<0.1): {boring:.1f}%")

    print()
    print("  Most engaging states (by total A):")
    scores = [(s, result.state_nodes[s].sum_a()) for s in non_terminal]
    scores.sort(key=lambda x: -x[1])
    for s, total in scores[:5]:
        a1 = result.state_nodes[s].a[0]
        print(f"    HP({s[0]},{s[1]}): A₁={a1:.4f}, Total={total:.4f}")


def compare(genome: SymmetricGenome, fitness: float, nest_level: int = 5):
    """Compare with known games."""
    from toa.games.hpgame import HpGame
    from toa.games.hpgame_rage import HpGameRage

    hp = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=nest_level,
    )

    config = HpGameRage.Config(critical_chance=0.13)
    rage = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=nest_level,
    )

    print()
    print("=" * 75)
    print("COMPARISON")
    print("=" * 75)
    print(f"  {'Game':<30} {'GDS':>8} {'D₀':>6} {'States':>8}")
    print(f"  {'-'*55}")

    init_state = HpGame.initial_state()
    d0_hp = hp.state_nodes[init_state].d_global
    print(f"  {'HpGame (5,5)':<30} {hp.game_design_score:>8.4f} {d0_hp:>6.3f} {len(hp.states):>8}")

    init_state = HpGameRage.initial_state()
    d0_rage = rage.state_nodes[init_state].d_global
    print(f"  {'HpGame_Rage (13% crit)':<30} {rage.game_design_score:>8.4f} {d0_rage:>6.3f} {len(rage.states):>8}")

    print(f"  {'GA Symmetric Best':<30} {fitness:>8.4f}")

    if fitness > rage.game_design_score:
        pct = (fitness - rage.game_design_score) / rage.game_design_score * 100
        print(f"\n  → GA symmetric game beats HpGame_Rage by {pct:.1f}%!")
    elif fitness > hp.game_design_score:
        pct = (fitness - hp.game_design_score) / hp.game_design_score * 100
        print(f"\n  → GA symmetric game beats HpGame by {pct:.1f}% (below Rage)")
    else:
        print(f"\n  → GA didn't surpass known designs yet")

    # Insight: what makes the GA result different?
    print()
    print("  DESIGN INSIGHTS:")
    outcomes = genome.get_outcomes()
    total_damage_rate = sum(p * (abs(d1) + abs(d2)) for p, d1, d2 in outcomes if d1 < 0 or d2 < 0)
    healing_rate = sum(p * (max(0, d1) + max(0, d2)) for p, d1, d2 in outcomes)
    asymmetric_outcomes = sum(1 for _, d1, d2 in outcomes if d1 != d2)

    print(f"  - Expected damage per turn: {total_damage_rate:.3f}")
    print(f"  - Expected healing per turn: {healing_rate:.3f}")
    print(f"  - Asymmetric outcomes: {asymmetric_outcomes}/{len(outcomes)}")
    print(f"  - Max HP: {genome.max_hp} → expected game length: ~{genome.max_hp * 2 / max(0.01, total_damage_rate):.0f} turns")


if __name__ == "__main__":
    best, fitness, history = run(
        pop_size=120,
        generations=60,
        elite=12,
        tournament=5,
        mutation_rate=0.3,
        nest_level=5,
        seed=42,
        verbose=True,
    )
