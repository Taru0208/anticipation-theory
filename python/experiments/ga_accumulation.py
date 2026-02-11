"""GA with Accumulating Mechanics — Can evolution discover rage-like systems?

The v2 GA found optimal "stateless" combat games (HP only).
This extends the genome to include an accumulating resource dimension:

State: (hp1, hp2, resource1, resource2)

The resource:
- Accumulates based on configurable triggers (dealing damage, receiving damage, etc.)
- Can modify outcome probabilities or damage values
- Can be spent or persistent

This is the experiment that tests whether evolution can independently discover
the rage mechanic (or something even better).
"""

import sys
import os
import math
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


class AccumulationGenome:
    """Genome for combat game with an accumulating resource.

    The resource starts at 0 and grows based on events. When it reaches
    a threshold, it modifies the game (extra damage, changed probabilities, etc.).

    To keep state space manageable: resource caps at max_resource.
    """

    def __init__(
        self,
        max_hp=5,
        max_resource=3,
        # Base combat: symmetric pairs (weight, d1, d2)
        base_pairs=None,
        # Resource accumulation triggers
        gain_on_deal_damage=True,     # +1 resource when dealing damage
        gain_on_take_damage=True,     # +1 resource when taking damage
        # Resource effect
        bonus_damage_per_resource=0,  # Extra damage on hit per resource level
        bonus_kill_chance=0.0,        # Extra probability of big hit per resource
        resource_spent_on_use=False,  # Whether resource resets after triggering
        # Threshold for effect activation
        effect_threshold=1,           # Minimum resource for bonus to apply
    ):
        self.max_hp = max_hp
        self.max_resource = max_resource
        self.base_pairs = base_pairs or [(1.0, 0, -1), (1.0, -1, 0)]
        self.gain_on_deal_damage = gain_on_deal_damage
        self.gain_on_take_damage = gain_on_take_damage
        self.bonus_damage_per_resource = bonus_damage_per_resource
        self.bonus_kill_chance = bonus_kill_chance
        self.resource_spent_on_use = resource_spent_on_use
        self.effect_threshold = effect_threshold

    def copy(self):
        return AccumulationGenome(
            max_hp=self.max_hp,
            max_resource=self.max_resource,
            base_pairs=[(w, d1, d2) for w, d1, d2 in self.base_pairs],
            gain_on_deal_damage=self.gain_on_deal_damage,
            gain_on_take_damage=self.gain_on_take_damage,
            bonus_damage_per_resource=self.bonus_damage_per_resource,
            bonus_kill_chance=self.bonus_kill_chance,
            resource_spent_on_use=self.resource_spent_on_use,
            effect_threshold=self.effect_threshold,
        )


def genome_to_functions(g: AccumulationGenome):
    """Convert genome to analyzer functions.

    State: (hp1, hp2, res1, res2)
    """
    # Compute base probabilities from pairs (symmetric)
    raw_outcomes = []
    for w, d1, d2 in g.base_pairs:
        raw_outcomes.append((w, d1, d2))
        if d1 != d2:
            raw_outcomes.append((w, d2, d1))  # Mirror for symmetry

    total_w = sum(w for w, _, _ in raw_outcomes)
    base_outcomes = [(w / total_w, d1, d2) for w, d1, d2 in raw_outcomes]

    def initial_state():
        return (g.max_hp, g.max_hp, 0, 0)

    def is_terminal(state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(state, config=None):
        hp1, hp2, r1, r2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = []

        for prob, d1, d2 in base_outcomes:
            new_hp1 = hp1 + d1
            new_hp2 = hp2 + d2
            new_r1 = r1
            new_r2 = r2

            # Apply resource-based bonus damage
            if d2 < 0 and r1 >= g.effect_threshold:
                # P1 is dealing damage to P2, P1 has enough resource
                bonus = g.bonus_damage_per_resource * r1
                new_hp2 -= bonus
                if g.resource_spent_on_use:
                    new_r1 = 0

            if d1 < 0 and r2 >= g.effect_threshold:
                # P2 is dealing damage to P1, P2 has enough resource
                bonus = g.bonus_damage_per_resource * r2
                new_hp1 -= bonus
                if g.resource_spent_on_use:
                    new_r2 = 0

            # Accumulate resource
            if d2 < 0 and g.gain_on_deal_damage:
                new_r1 = min(g.max_resource, new_r1 + 1)  # P1 dealt damage
            if d1 < 0 and g.gain_on_take_damage:
                new_r1 = min(g.max_resource, new_r1 + 1)  # P1 took damage

            if d1 < 0 and g.gain_on_deal_damage:
                new_r2 = min(g.max_resource, new_r2 + 1)  # P2 dealt damage
            if d2 < 0 and g.gain_on_take_damage:
                new_r2 = min(g.max_resource, new_r2 + 1)  # P2 took damage

            # Clamp
            new_hp1 = max(0, min(g.max_hp, new_hp1))
            new_hp2 = max(0, min(g.max_hp, new_hp2))

            # Resource-modified probability (bonus kill chance)
            if g.bonus_kill_chance > 0 and prob > 0.01:
                # Split: original outcome + bonus kill outcome
                kill_prob_1 = g.bonus_kill_chance * r1 / max(1, g.max_resource)
                kill_prob_2 = g.bonus_kill_chance * r2 / max(1, g.max_resource)

                # Keep it simple: just modify the hp directly via bonus damage
                pass

            transitions.append((prob, (new_hp1, new_hp2, new_r1, new_r2)))

        return sanitize_transitions(transitions)

    def desire(state):
        return 1.0 if state[0] > 0 and state[1] <= 0 else 0.0

    return initial_state, is_terminal, get_transitions, desire


def evaluate(g: AccumulationGenome, nest_level=5) -> float:
    """Evaluate a genome."""
    init, is_term, trans, desire = genome_to_functions(g)
    try:
        result = analyze(
            initial_state=init(),
            is_terminal=is_term,
            get_transitions=trans,
            compute_intrinsic_desire=desire,
            nest_level=nest_level,
        )

        non_terminal = [s for s in result.states if not is_term(s)]
        if len(non_terminal) < 3:
            return -1.0

        d0 = result.state_nodes[init()].d_global
        if abs(d0 - 0.5) > 0.1:
            return result.game_design_score * 0.3

        return result.game_design_score
    except Exception:
        return -1.0


def random_genome(rng: random.Random) -> AccumulationGenome:
    """Generate random accumulation genome."""
    n_pairs = rng.randint(1, 3)
    pairs = []
    for _ in range(n_pairs):
        w = rng.uniform(0.1, 3.0)
        d1 = rng.randint(-2, 1)
        d2 = rng.randint(-2, 1)
        pairs.append((w, d1, d2))

    return AccumulationGenome(
        max_hp=rng.randint(3, 6),
        max_resource=rng.randint(1, 4),
        base_pairs=pairs,
        gain_on_deal_damage=rng.random() < 0.5,
        gain_on_take_damage=rng.random() < 0.5,
        bonus_damage_per_resource=rng.randint(0, 2),
        resource_spent_on_use=rng.random() < 0.5,
        effect_threshold=rng.randint(1, 3),
    )


def mutate(g: AccumulationGenome, rng: random.Random, rate=0.3) -> AccumulationGenome:
    g = g.copy()

    if rng.random() < rate:
        g.max_hp = max(3, min(6, g.max_hp + rng.choice([-1, 0, 1])))
    if rng.random() < rate:
        g.max_resource = max(1, min(4, g.max_resource + rng.choice([-1, 0, 1])))

    for i in range(len(g.base_pairs)):
        w, d1, d2 = g.base_pairs[i]
        if rng.random() < rate:
            w = max(0.01, w + rng.gauss(0, 0.5))
        if rng.random() < rate:
            d1 = max(-3, min(1, d1 + rng.choice([-1, 0, 1])))
        if rng.random() < rate:
            d2 = max(-3, min(1, d2 + rng.choice([-1, 0, 1])))
        g.base_pairs[i] = (w, d1, d2)

    if rng.random() < rate * 0.3 and len(g.base_pairs) < 4:
        g.base_pairs.append((rng.uniform(0.1, 2.0), rng.randint(-2, 1), rng.randint(-2, 1)))
    elif rng.random() < rate * 0.2 and len(g.base_pairs) > 1:
        g.base_pairs.pop(rng.randint(0, len(g.base_pairs) - 1))

    if rng.random() < rate:
        g.gain_on_deal_damage = not g.gain_on_deal_damage
    if rng.random() < rate:
        g.gain_on_take_damage = not g.gain_on_take_damage
    if rng.random() < rate:
        g.bonus_damage_per_resource = max(0, min(3, g.bonus_damage_per_resource + rng.choice([-1, 0, 1])))
    if rng.random() < rate:
        g.resource_spent_on_use = not g.resource_spent_on_use
    if rng.random() < rate:
        g.effect_threshold = max(1, min(4, g.effect_threshold + rng.choice([-1, 0, 1])))

    return g


def crossover(g1: AccumulationGenome, g2: AccumulationGenome, rng: random.Random) -> AccumulationGenome:
    child = g1.copy()
    child.max_hp = rng.choice([g1.max_hp, g2.max_hp])
    child.max_resource = rng.choice([g1.max_resource, g2.max_resource])
    child.gain_on_deal_damage = rng.choice([g1.gain_on_deal_damage, g2.gain_on_deal_damage])
    child.gain_on_take_damage = rng.choice([g1.gain_on_take_damage, g2.gain_on_take_damage])
    child.bonus_damage_per_resource = rng.choice([g1.bonus_damage_per_resource, g2.bonus_damage_per_resource])
    child.resource_spent_on_use = rng.choice([g1.resource_spent_on_use, g2.resource_spent_on_use])
    child.effect_threshold = rng.choice([g1.effect_threshold, g2.effect_threshold])

    # Mix base pairs
    all_pairs = g1.base_pairs + g2.base_pairs
    n = max(1, min(4, rng.randint(1, len(all_pairs))))
    child.base_pairs = [all_pairs[rng.randint(0, len(all_pairs)-1)] for _ in range(n)]

    return child


def run(pop_size=80, generations=40, elite=8, tournament=5,
        mutation_rate=0.3, nest_level=5, seed=42, verbose=True):
    """Run GA for accumulation games."""
    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(pop_size)]

    best_ever = None
    best_fitness = -1.0

    if verbose:
        print("Accumulation Mechanic GA")
        print("=" * 80)
        print(f"Pop: {pop_size}, Gen: {generations}")
        print("=" * 80)
        print(f"{'Gen':>4}  {'Best':>8}  {'Avg':>8}  {'HP':>4}  {'Res':>4}  {'BonusDmg':>8}  {'Spend':>6}  {'States':>8}  {'Time':>6}")
        print("-" * 75)

    for gen in range(generations):
        t0 = time.time()
        fitness = [evaluate(g, nest_level) for g in population]

        pairs = sorted(zip(fitness, population), key=lambda x: -x[0])
        f_sorted = [f for f, _ in pairs]
        p_sorted = [g for _, g in pairs]

        if f_sorted[0] > best_fitness:
            best_fitness = f_sorted[0]
            best_ever = p_sorted[0].copy()

        valid = [f for f in f_sorted if f > 0]
        avg = sum(valid) / max(1, len(valid))
        elapsed = time.time() - t0

        if verbose:
            b = p_sorted[0]
            # Count states
            init, is_term, trans, desire = genome_to_functions(b)
            try:
                r = analyze(initial_state=init(), is_terminal=is_term,
                           get_transitions=trans, compute_intrinsic_desire=desire, nest_level=1)
                n_states = len(r.states)
            except Exception:
                n_states = -1

            print(f"{gen:>4}  {f_sorted[0]:>8.4f}  {avg:>8.4f}  {b.max_hp:>4}  {b.max_resource:>4}  "
                  f"{b.bonus_damage_per_resource:>8}  {'Y' if b.resource_spent_on_use else 'N':>6}  "
                  f"{n_states:>8}  {elapsed:>5.1f}s")

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
        describe(best_ever, nest_level)
        compare(best_ever, best_fitness, nest_level)

    return best_ever, best_fitness


def describe(g: AccumulationGenome, nest_level=5):
    """Describe the best genome found."""
    print("=" * 80)
    print("BEST ACCUMULATION GAME")
    print("=" * 80)

    init, is_term, trans, desire = genome_to_functions(g)
    result = analyze(
        initial_state=init(),
        is_terminal=is_term,
        get_transitions=trans,
        compute_intrinsic_desire=desire,
        nest_level=nest_level,
    )

    print(f"  HP: {g.max_hp}, Resource cap: {g.max_resource}")
    print(f"  Gain on dealing damage: {g.gain_on_deal_damage}")
    print(f"  Gain on taking damage: {g.gain_on_take_damage}")
    print(f"  Bonus damage per resource: {g.bonus_damage_per_resource}")
    print(f"  Resource spent on use: {g.resource_spent_on_use}")
    print(f"  Effect threshold: {g.effect_threshold}")
    print()

    # Show base outcome pairs
    raw = []
    for w, d1, d2 in g.base_pairs:
        raw.append((w, d1, d2))
        if d1 != d2:
            raw.append((w, d2, d1))
    total_w = sum(w for w, _, _ in raw)
    print(f"  Base outcomes (symmetric):")
    for w, d1, d2 in raw:
        print(f"    P={w/total_w:.3f}  P1:{d1:+d}  P2:{d2:+d}")

    non_terminal = [s for s in result.states if not is_term(s)]
    d0 = result.state_nodes[init()].d_global

    print()
    print(f"  GDS: {result.game_design_score:.6f}")
    comps = [f"A{i+1}={result.gds_components[i]:.4f}"
             for i in range(nest_level) if result.gds_components[i] > 0.001]
    print(f"  Components: {' '.join(comps)}")
    print(f"  States: {len(result.states)} ({len(non_terminal)} non-terminal)")
    print(f"  D₀: {d0:.4f}")

    # Resource distribution in most engaging states
    print()
    print("  Most engaging states:")
    scores = [(s, result.state_nodes[s].sum_a()) for s in non_terminal]
    scores.sort(key=lambda x: -x[1])
    for s, total in scores[:8]:
        a1 = result.state_nodes[s].a[0]
        print(f"    HP({s[0]},{s[1]}) R({s[2]},{s[3]}): A₁={a1:.4f}, Total={total:.4f}")


def compare(g: AccumulationGenome, fitness: float, nest_level=5):
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
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  {'Game':<35} {'GDS':>8} {'States':>8}")
    print(f"  {'-'*55}")
    print(f"  {'HpGame (5,5)':<35} {hp.game_design_score:>8.4f} {len(hp.states):>8}")
    print(f"  {'HpGame_Rage (13% crit)':<35} {rage.game_design_score:>8.4f} {len(rage.states):>8}")
    print(f"  {'GA v2 Symmetric (no resource)':<35} {'0.9794':>8} {'15':>8}")
    print(f"  {'GA Accumulation Best':<35} {fitness:>8.4f}")

    if fitness > 0.9794:
        print(f"\n  → Accumulation mechanic IMPROVES on stateless optimal!")
    elif fitness > rage.game_design_score:
        print(f"\n  → Better than hand-designed HpGame_Rage, but below stateless GA")
    else:
        print(f"\n  → Below known designs. Accumulation may need different parameterization.")


if __name__ == "__main__":
    best, fitness = run(
        pop_size=80,
        generations=40,
        elite=8,
        tournament=5,
        mutation_rate=0.3,
        nest_level=5,
        seed=42,
        verbose=True,
    )
