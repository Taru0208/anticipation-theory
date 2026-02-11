"""Genetic Algorithm Game Optimizer

Uses evolutionary search to discover game mechanics that maximize GDS.
Instead of manually tweaking parameters, we let evolution find optimal designs.

The search space: parameterized combat games with configurable:
- Number of outcomes per turn (2-4)
- Probability distribution across outcomes
- Damage/healing values per outcome
- HP values
- Special mechanics (rage, critical, etc.)

This is the experiment hinted at in Vol 1 but never implemented.
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


# ---------------------------------------------------------------------------
# Parameterized Game: a flexible combat game defined by a genome
# ---------------------------------------------------------------------------

@dataclass
class CombatGenome:
    """A genome encoding a 1v1 combat game's mechanics.

    Each turn has N outcomes, each with a probability, and effects on both players.
    Effects are (hp1_delta, hp2_delta) pairs.

    The genome is a flat list of floats for easy crossover/mutation.
    """
    max_hp: int = 5  # Starting HP for each player (3-8)
    num_outcomes: int = 3  # Number of distinct outcomes per turn (2-5)
    # For each outcome: (raw_weight, hp1_delta, hp2_delta)
    # raw_weights are softmaxed to get probabilities
    outcome_weights: list[float] = field(default_factory=list)
    outcome_effects: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.outcome_weights:
            self.outcome_weights = [1.0] * self.num_outcomes
        if not self.outcome_effects:
            self.outcome_effects = [(0, -1), (-1, -1), (-1, 0)][:self.num_outcomes]

    @property
    def probabilities(self) -> list[float]:
        """Softmax the raw weights to get valid probabilities."""
        max_w = max(self.outcome_weights)
        exp_w = [math.exp(w - max_w) for w in self.outcome_weights]
        total = sum(exp_w)
        return [e / total for e in exp_w]

    def copy(self) -> "CombatGenome":
        return CombatGenome(
            max_hp=self.max_hp,
            num_outcomes=self.num_outcomes,
            outcome_weights=list(self.outcome_weights),
            outcome_effects=[tuple(e) for e in self.outcome_effects],
        )


def genome_to_game_functions(genome: CombatGenome):
    """Convert a genome into the 4 functions needed by the analyzer."""

    probs = genome.probabilities
    effects = genome.outcome_effects
    max_hp = genome.max_hp

    def initial_state():
        return (max_hp, max_hp)

    def is_terminal(state):
        hp1, hp2 = state
        return hp1 <= 0 or hp2 <= 0

    def get_transitions(state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = []
        for i, (d1, d2) in enumerate(effects):
            new_hp1 = max(0, min(max_hp, hp1 + d1))
            new_hp2 = max(0, min(max_hp, hp2 + d2))
            if probs[i] > 1e-9:
                transitions.append((probs[i], (new_hp1, new_hp2)))

        return sanitize_transitions(transitions)

    def compute_intrinsic_desire(state):
        hp1, hp2 = state
        return 1.0 if hp1 > 0 and hp2 <= 0 else 0.0

    return initial_state, is_terminal, get_transitions, compute_intrinsic_desire


def evaluate_genome(genome: CombatGenome, nest_level: int = 5) -> float:
    """Evaluate a genome by computing its GDS. Returns -1 if game is degenerate."""
    initial, is_term, get_trans, desire = genome_to_game_functions(genome)

    try:
        result = analyze(
            initial_state=initial(),
            is_terminal=is_term,
            get_transitions=get_trans,
            compute_intrinsic_desire=desire,
            nest_level=nest_level,
        )

        # Penalize degenerate games
        non_terminal = [s for s in result.states if not is_term(s)]
        if len(non_terminal) < 3:
            return -1.0  # Too simple

        # Penalize games where initial win probability is too skewed
        d0 = result.state_nodes[initial()].d_global
        if d0 > 0.8 or d0 < 0.2:
            return result.game_design_score * 0.5  # Unfair penalty

        return result.game_design_score
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Random genome generation
# ---------------------------------------------------------------------------

def random_genome(rng: random.Random) -> CombatGenome:
    """Generate a random combat game genome."""
    max_hp = rng.randint(3, 8)
    num_outcomes = rng.randint(2, 5)

    weights = [rng.uniform(0.1, 3.0) for _ in range(num_outcomes)]

    effects = []
    for _ in range(num_outcomes):
        d1 = rng.randint(-2, 1)  # -2 to +1 HP change for P1
        d2 = rng.randint(-2, 1)  # -2 to +1 HP change for P2
        effects.append((d1, d2))

    return CombatGenome(
        max_hp=max_hp,
        num_outcomes=num_outcomes,
        outcome_weights=weights,
        outcome_effects=effects,
    )


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def mutate(genome: CombatGenome, rng: random.Random, mutation_rate: float = 0.3) -> CombatGenome:
    """Mutate a genome with given probability per gene."""
    g = genome.copy()

    # Mutate HP
    if rng.random() < mutation_rate:
        g.max_hp = max(3, min(8, g.max_hp + rng.choice([-1, 0, 1])))

    # Mutate weights
    for i in range(g.num_outcomes):
        if rng.random() < mutation_rate:
            g.outcome_weights[i] += rng.gauss(0, 0.5)
            g.outcome_weights[i] = max(0.01, g.outcome_weights[i])

    # Mutate effects
    for i in range(g.num_outcomes):
        if rng.random() < mutation_rate:
            d1, d2 = g.outcome_effects[i]
            if rng.random() < 0.5:
                d1 = max(-2, min(1, d1 + rng.choice([-1, 0, 1])))
            else:
                d2 = max(-2, min(1, d2 + rng.choice([-1, 0, 1])))
            g.outcome_effects[i] = (d1, d2)

    # Occasionally add or remove an outcome
    if rng.random() < mutation_rate * 0.3:
        if g.num_outcomes < 5 and rng.random() < 0.5:
            g.num_outcomes += 1
            g.outcome_weights.append(rng.uniform(0.1, 2.0))
            g.outcome_effects.append((rng.randint(-2, 1), rng.randint(-2, 1)))
        elif g.num_outcomes > 2:
            idx = rng.randint(0, g.num_outcomes - 1)
            g.num_outcomes -= 1
            g.outcome_weights.pop(idx)
            g.outcome_effects.pop(idx)

    return g


def crossover(g1: CombatGenome, g2: CombatGenome, rng: random.Random) -> CombatGenome:
    """Single-point crossover between two genomes."""
    child = g1.copy()

    # HP from either parent
    child.max_hp = rng.choice([g1.max_hp, g2.max_hp])

    # Take outcome count from either parent
    if rng.random() < 0.5 and g1.num_outcomes == g2.num_outcomes:
        # Mix outcomes from both parents
        for i in range(child.num_outcomes):
            if rng.random() < 0.5:
                child.outcome_weights[i] = g2.outcome_weights[i]
                child.outcome_effects[i] = g2.outcome_effects[i]

    return child


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

def run_evolution(
    population_size: int = 100,
    generations: int = 50,
    elite_count: int = 10,
    tournament_size: int = 5,
    mutation_rate: float = 0.3,
    nest_level: int = 5,
    seed: int = 42,
    verbose: bool = True,
):
    """Run genetic algorithm to find optimal game designs."""

    rng = random.Random(seed)

    # Initialize population
    population = [random_genome(rng) for _ in range(population_size)]

    best_ever = None
    best_ever_fitness = -1.0
    history = []

    if verbose:
        print("Genetic Algorithm Game Optimizer")
        print("=" * 70)
        print(f"Population: {population_size}, Generations: {generations}")
        print(f"Elite: {elite_count}, Tournament: {tournament_size}")
        print(f"Mutation rate: {mutation_rate}, Nest level: {nest_level}")
        print("=" * 70)
        print(f"{'Gen':>4}  {'Best':>8}  {'Avg':>8}  {'Worst':>8}  {'BestHP':>6}  {'Outcomes':>8}  {'Time':>6}")
        print("-" * 60)

    for gen in range(generations):
        t0 = time.time()

        # Evaluate fitness
        fitness = []
        for g in population:
            f = evaluate_genome(g, nest_level)
            fitness.append(f)

        # Sort by fitness (descending)
        pairs = sorted(zip(fitness, population), key=lambda x: -x[0])
        fitness_sorted = [f for f, g in pairs]
        pop_sorted = [g for f, g in pairs]

        # Track best
        if fitness_sorted[0] > best_ever_fitness:
            best_ever_fitness = fitness_sorted[0]
            best_ever = pop_sorted[0].copy()

        # Stats
        valid = [f for f in fitness_sorted if f > 0]
        avg_fit = sum(valid) / max(1, len(valid))
        worst_fit = valid[-1] if valid else -1
        elapsed = time.time() - t0

        history.append({
            "gen": gen,
            "best": fitness_sorted[0],
            "avg": avg_fit,
            "worst": worst_fit,
        })

        if verbose:
            best_g = pop_sorted[0]
            print(f"{gen:>4}  {fitness_sorted[0]:>8.4f}  {avg_fit:>8.4f}  {worst_fit:>8.4f}  "
                  f"{best_g.max_hp:>6}  {best_g.num_outcomes:>8}  {elapsed:>5.1f}s")

        # --- Selection & reproduction ---
        new_population = []

        # Elitism: keep top N
        for i in range(elite_count):
            new_population.append(pop_sorted[i].copy())

        # Fill rest with tournament selection + crossover + mutation
        while len(new_population) < population_size:
            # Tournament selection (2 parents)
            parents = []
            for _ in range(2):
                contestants = rng.sample(list(enumerate(fitness_sorted)), min(tournament_size, len(fitness_sorted)))
                winner_idx = max(contestants, key=lambda x: x[1])[0]
                parents.append(pop_sorted[winner_idx])

            child = crossover(parents[0], parents[1], rng)
            child = mutate(child, rng, mutation_rate)
            new_population.append(child)

        population = new_population

    # Final analysis of best genome
    if verbose and best_ever:
        print()
        print("=" * 70)
        print("BEST GAME FOUND")
        print("=" * 70)
        describe_genome(best_ever, nest_level)

    return best_ever, best_ever_fitness, history


def describe_genome(genome: CombatGenome, nest_level: int = 5):
    """Print detailed analysis of a genome."""
    probs = genome.probabilities

    print(f"  Max HP: {genome.max_hp}")
    print(f"  Outcomes per turn: {genome.num_outcomes}")
    print()
    print(f"  {'Outcome':<10} {'Prob':>8} {'P1 HP Δ':>10} {'P2 HP Δ':>10}")
    print(f"  {'-'*40}")
    for i in range(genome.num_outcomes):
        d1, d2 = genome.outcome_effects[i]
        print(f"  {i+1:<10} {probs[i]:>8.3f} {d1:>+10} {d2:>+10}")

    # Evaluate
    initial, is_term, get_trans, desire = genome_to_game_functions(genome)
    result = analyze(
        initial_state=initial(),
        is_terminal=is_term,
        get_transitions=get_trans,
        compute_intrinsic_desire=desire,
        nest_level=nest_level,
    )

    non_terminal = [s for s in result.states if not is_term(s)]
    print()
    print(f"  GDS: {result.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={result.gds_components[i]:.4f}' for i in range(nest_level) if result.gds_components[i] > 0.001)}")
    print(f"  States: {len(result.states)} ({len(non_terminal)} non-terminal)")
    print(f"  Initial D_global: {result.state_nodes[initial()].d_global:.4f}")

    # Show most engaging states
    state_scores = [(s, result.state_nodes[s].sum_a()) for s in non_terminal]
    state_scores.sort(key=lambda x: -x[1])
    print()
    print("  Most engaging states:")
    for s, score in state_scores[:5]:
        a1 = result.state_nodes[s].a[0]
        print(f"    HP({s[0]},{s[1]}): A₁={a1:.4f}, Total={score:.4f}")


# ---------------------------------------------------------------------------
# Comparison: how does the evolved game compare to known games?
# ---------------------------------------------------------------------------

def compare_with_known_games(best_genome, best_fitness, nest_level=5):
    """Compare the GA result with manually-designed games."""
    from toa.games.hpgame import HpGame
    from toa.games.hpgame_rage import HpGameRage

    print()
    print("=" * 70)
    print("COMPARISON WITH KNOWN GAMES")
    print("=" * 70)

    # HpGame baseline
    hp_result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=nest_level,
    )

    # HpGame Rage optimal
    config = HpGameRage.Config(critical_chance=0.13)
    rage_result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=nest_level,
    )

    print(f"  {'Game':<25} {'GDS':>10} {'A₁':>8} {'A₂':>8} {'States':>8}")
    print(f"  {'-'*60}")
    print(f"  {'HpGame (baseline)':<25} {hp_result.game_design_score:>10.4f} "
          f"{hp_result.gds_components[0]:>8.4f} {hp_result.gds_components[1]:>8.4f} "
          f"{len(hp_result.states):>8}")
    print(f"  {'HpGame_Rage (13% crit)':<25} {rage_result.game_design_score:>10.4f} "
          f"{rage_result.gds_components[0]:>8.4f} {rage_result.gds_components[1]:>8.4f} "
          f"{len(rage_result.states):>8}")
    print(f"  {'GA Best':<25} {best_fitness:>10.4f}")

    if best_fitness > rage_result.game_design_score:
        improvement = (best_fitness - rage_result.game_design_score) / rage_result.game_design_score * 100
        print(f"\n  → GA found a game {improvement:.1f}% better than the best known hand-designed game!")
    elif best_fitness > hp_result.game_design_score:
        improvement = (best_fitness - hp_result.game_design_score) / hp_result.game_design_score * 100
        print(f"\n  → GA found a game {improvement:.1f}% better than HpGame baseline")
        print(f"    (but still below optimized HpGame_Rage)")
    else:
        print(f"\n  → GA didn't surpass known designs. Search space may need expansion.")


if __name__ == "__main__":
    best_genome, best_fitness, history = run_evolution(
        population_size=80,
        generations=40,
        elite_count=8,
        tournament_size=5,
        mutation_rate=0.3,
        nest_level=5,
        seed=42,
        verbose=True,
    )

    if best_genome:
        compare_with_known_games(best_genome, best_fitness)
