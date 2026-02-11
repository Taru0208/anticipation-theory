"""Unbound Conjecture — Comprehensive Investigation (v2)

The Unbound Conjecture: Can GDS grow without bound as game depth increases?

v1 only tested GoldGame (independent random accumulation).
This v2 tests multiple game classes to understand WHICH structural properties
allow unbounded growth:

1. GoldGame (independent outcomes per turn — multiplicative)
2. GoldGame Additive (independent outcomes — additive)
3. Combat (HP-bounded shared state)
4. Combat + Accumulation (HP + resource dimension)
5. Best-of-N Coin Toss (increasing depth with binary outcomes)
6. Multi-lane (parallel independent sub-games)

For each class, we measure GDS as depth increases and fit growth curves.

Key hypothesis: Games with INDEPENDENT outcome accumulation (GoldGame)
show unbounded growth, while games with SHARED STATE CONSTRAINTS (HP)
show bounded growth. The accumulation mechanic may bridge the gap.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


# ─── Game Class 1: GoldGame (Multiplicative) ─────────────────────────────────

class GoldGameMultiplicative:
    """Two players independently earn gold. Multiplicative growth."""

    def __init__(self, max_turns, success_chance=0.68, multiplier=1.2):
        self.max_turns = max_turns
        self.p = success_chance
        self.mult = multiplier
        self.div = 1.0 / multiplier

    def initial_state(self):
        return (1000, 1000, 0)

    def is_terminal(self, state):
        return state[2] >= self.max_turns

    def get_transitions(self, state, config=None):
        p1, p2, t = state
        if t >= self.max_turns:
            return []
        result = []
        for h1 in (True, False):
            for h2 in (True, False):
                np1 = int(p1 * (self.mult if h1 else self.div))
                np2 = int(p2 * (self.mult if h2 else self.div))
                prob = (self.p if h1 else 1-self.p) * (self.p if h2 else 1-self.p)
                result.append((prob, (np1, np2, t + 1)))
        return sanitize_transitions(result)

    def compute_intrinsic_desire(self, state):
        if state[2] < self.max_turns:
            return 0.0
        return 1.0 if state[0] > state[1] else (0.5 if state[0] == state[1] else 0.0)


# ─── Game Class 2: GoldGame (Additive) ───────────────────────────────────────

class GoldGameAdditive:
    """Two players independently earn gold. Additive growth (linear)."""

    def __init__(self, max_turns, success_chance=0.6, gain=100, loss=50):
        self.max_turns = max_turns
        self.p = success_chance
        self.gain = gain
        self.loss = loss

    def initial_state(self):
        return (500, 500, 0)

    def is_terminal(self, state):
        return state[2] >= self.max_turns

    def get_transitions(self, state, config=None):
        p1, p2, t = state
        if t >= self.max_turns:
            return []
        result = []
        for h1 in (True, False):
            for h2 in (True, False):
                np1 = p1 + (self.gain if h1 else -self.loss)
                np2 = p2 + (self.gain if h2 else -self.loss)
                prob = (self.p if h1 else 1-self.p) * (self.p if h2 else 1-self.p)
                result.append((prob, (np1, np2, t + 1)))
        return sanitize_transitions(result)

    def compute_intrinsic_desire(self, state):
        if state[2] < self.max_turns:
            return 0.0
        return 1.0 if state[0] > state[1] else (0.5 if state[0] == state[1] else 0.0)


# ─── Game Class 3: Combat (HP-bounded) ───────────────────────────────────────

class ScalableCombat:
    """Simple combat with configurable HP. Depth = max HP."""

    def __init__(self, max_hp):
        self.max_hp = max_hp

    def initial_state(self):
        return (self.max_hp, self.max_hp)

    def is_terminal(self, state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(self, state, config=None):
        hp1, hp2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []
        # Three outcomes: P1 hits P2, P2 hits P1, both hit
        return sanitize_transitions([
            (0.4, (hp1, hp2 - 1)),      # P1 hits P2
            (0.4, (hp1 - 1, hp2)),      # P2 hits P1
            (0.2, (hp1 - 1, hp2 - 1)),  # Both hit
        ])

    def compute_intrinsic_desire(self, state):
        if state[0] > 0 and state[1] <= 0:
            return 1.0
        return 0.0


# ─── Game Class 4: Combat + Resource ─────────────────────────────────────────

class CombatWithResource:
    """Combat with an accumulating resource. HP determines depth, resource adds dimension."""

    def __init__(self, max_hp, max_resource=3):
        self.max_hp = max_hp
        self.max_resource = max_resource

    def initial_state(self):
        return (self.max_hp, self.max_hp, 0, 0)

    def is_terminal(self, state):
        return state[0] <= 0 or state[1] <= 0

    def get_transitions(self, state, config=None):
        hp1, hp2, r1, r2 = state
        if hp1 <= 0 or hp2 <= 0:
            return []

        transitions = []
        # Base damage = 1, bonus = 1 per resource level above threshold
        bonus1 = max(0, r1 - 1)  # Bonus kicks in at resource >= 2
        bonus2 = max(0, r2 - 1)

        # P1 hits P2 (gains resource)
        new_r1 = min(self.max_resource, r1 + 1)
        transitions.append((0.4, (hp1, max(0, hp2 - 1 - bonus1), new_r1, r2)))

        # P2 hits P1 (gains resource)
        new_r2 = min(self.max_resource, r2 + 1)
        transitions.append((0.4, (max(0, hp1 - 1 - bonus2), hp2, r1, new_r2)))

        # Both hit (both gain resource)
        new_r1 = min(self.max_resource, r1 + 1)
        new_r2 = min(self.max_resource, r2 + 1)
        transitions.append((0.2, (
            max(0, hp1 - 1 - bonus2), max(0, hp2 - 1 - bonus1),
            new_r1, new_r2
        )))

        return sanitize_transitions(transitions)

    def compute_intrinsic_desire(self, state):
        if state[0] > 0 and state[1] <= 0:
            return 1.0
        return 0.0


# ─── Game Class 5: Best-of-N Coin Toss ───────────────────────────────────────

class BestOfN:
    """Best-of-N coin tosses. First to ceil(N/2) wins."""

    def __init__(self, n_rounds, win_prob=0.5):
        self.n = n_rounds
        self.target = (n_rounds + 1) // 2  # Wins needed
        self.p = win_prob

    def initial_state(self):
        return (0, 0)  # (p1_wins, p2_wins)

    def is_terminal(self, state):
        return state[0] >= self.target or state[1] >= self.target

    def get_transitions(self, state, config=None):
        w1, w2 = state
        if w1 >= self.target or w2 >= self.target:
            return []
        return [
            (self.p, (w1 + 1, w2)),
            (1 - self.p, (w1, w2 + 1)),
        ]

    def compute_intrinsic_desire(self, state):
        if state[0] >= self.target:
            return 1.0
        return 0.0


# ─── Game Class 6: Parallel Lanes ────────────────────────────────────────────

class ParallelLanes:
    """N independent coin tosses per turn, winner has most lanes.
    State: (p1_total, p2_total, turn) where totals are cumulative lane wins.
    """

    def __init__(self, max_turns, n_lanes=2):
        self.max_turns = max_turns
        self.n_lanes = n_lanes

    def initial_state(self):
        return (0, 0, 0)

    def is_terminal(self, state):
        return state[2] >= self.max_turns

    def get_transitions(self, state, config=None):
        s1, s2, t = state
        if t >= self.max_turns:
            return []

        # Each lane: 50% P1 wins, 50% P2 wins
        # With n_lanes=2: (0,2), (1,1), (2,0) with probs 0.25, 0.5, 0.25
        n = self.n_lanes
        transitions = []
        for p1_wins in range(n + 1):
            p2_wins = n - p1_wins
            # Binomial probability
            prob = _binomial_prob(n, p1_wins, 0.5)
            transitions.append((prob, (s1 + p1_wins, s2 + p2_wins, t + 1)))

        return sanitize_transitions(transitions)

    def compute_intrinsic_desire(self, state):
        if state[2] < self.max_turns:
            return 0.0
        return 1.0 if state[0] > state[1] else (0.5 if state[0] == state[1] else 0.0)


def _binomial_prob(n, k, p):
    """Binomial probability: C(n,k) * p^k * (1-p)^(n-k)."""
    coeff = math.comb(n, k)
    return coeff * (p ** k) * ((1 - p) ** (n - k))


# ─── Analysis Engine ──────────────────────────────────────────────────────────

def measure_gds(game, nest_level=10, timeout=60):
    """Measure GDS for a game, with timeout protection."""
    t0 = time.time()
    try:
        result = analyze(
            initial_state=game.initial_state(),
            is_terminal=game.is_terminal,
            get_transitions=game.get_transitions,
            compute_intrinsic_desire=game.compute_intrinsic_desire,
            nest_level=nest_level,
        )
        elapsed = time.time() - t0
        return {
            "gds": result.game_design_score,
            "states": len(result.states),
            "time": elapsed,
            "components": [result.gds_components[i] for i in range(nest_level)],
            "error": None,
        }
    except Exception as e:
        return {
            "gds": None,
            "states": None,
            "time": time.time() - t0,
            "components": None,
            "error": str(e),
        }


def fit_growth(depths, gds_values):
    """Fit growth curve to (depth, GDS) data. Returns best-fit model name and params."""
    if len(depths) < 3:
        return "insufficient data", {}

    # Try: constant, logarithmic, linear, power law
    n = len(depths)

    # Log fit: GDS = a * ln(depth) + b
    log_depths = [math.log(d) for d in depths]
    a_log, b_log = _linear_regression(log_depths, gds_values)
    residual_log = sum((gds_values[i] - (a_log * log_depths[i] + b_log)) ** 2 for i in range(n))

    # Linear fit: GDS = a * depth + b
    a_lin, b_lin = _linear_regression(depths, gds_values)
    residual_lin = sum((gds_values[i] - (a_lin * depths[i] + b_lin)) ** 2 for i in range(n))

    # Sqrt fit: GDS = a * sqrt(depth) + b
    sqrt_depths = [math.sqrt(d) for d in depths]
    a_sqrt, b_sqrt = _linear_regression(sqrt_depths, gds_values)
    residual_sqrt = sum((gds_values[i] - (a_sqrt * sqrt_depths[i] + b_sqrt)) ** 2 for i in range(n))

    fits = [
        ("logarithmic", residual_log, {"a": a_log, "b": b_log}),
        ("sqrt", residual_sqrt, {"a": a_sqrt, "b": b_sqrt}),
        ("linear", residual_lin, {"a": a_lin, "b": b_lin}),
    ]

    best = min(fits, key=lambda x: x[1])
    return best[0], best[2]


def _linear_regression(x, y):
    """Simple linear regression: y = ax + b."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(x[i] * y[i] for i in range(n))
    sx2 = sum(xi ** 2 for xi in x)

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, sy / n

    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


# ─── Experiments ──────────────────────────────────────────────────────────────

def run_game_class(name, game_factory, depth_range, nest_level=10, timeout_per=60):
    """Run depth scaling for a game class."""
    print(f"\n  {'Depth':>6}  {'GDS':>10}  {'States':>10}  {'Time':>8}  Components")
    print(f"  {'-'*70}")

    depths = []
    gds_values = []

    for depth in depth_range:
        game = game_factory(depth)
        r = measure_gds(game, nest_level=nest_level, timeout=timeout_per)

        if r["error"]:
            print(f"  {depth:>6}  {'ERROR':>10}  {'':>10}  {r['time']:>7.1f}s  {r['error']}")
            continue

        if r["time"] > timeout_per:
            print(f"  {depth:>6}  {'TIMEOUT':>10}")
            break

        comp_str = " ".join(
            f"A{i+1}={r['components'][i]:.3f}"
            for i in range(min(nest_level, 6))
            if r["components"][i] > 0.001
        )

        print(f"  {depth:>6}  {r['gds']:>10.4f}  {r['states']:>10}  {r['time']:>7.2f}s  {comp_str}")
        depths.append(depth)
        gds_values.append(r["gds"])

        if r["time"] > timeout_per * 0.5:
            print(f"  (stopping — approaching time limit)")
            break

    if len(depths) >= 3:
        model, params = fit_growth(depths, gds_values)
        print(f"\n  Best fit: {model}")
        for k, v in params.items():
            print(f"    {k} = {v:.6f}")

        # Predict GDS at depth 100 and 1000
        if model == "logarithmic":
            pred_100 = params["a"] * math.log(100) + params["b"]
            pred_1000 = params["a"] * math.log(1000) + params["b"]
        elif model == "sqrt":
            pred_100 = params["a"] * math.sqrt(100) + params["b"]
            pred_1000 = params["a"] * math.sqrt(1000) + params["b"]
        else:
            pred_100 = params["a"] * 100 + params["b"]
            pred_1000 = params["a"] * 1000 + params["b"]

        print(f"  Predicted GDS at depth 100: {pred_100:.4f}")
        print(f"  Predicted GDS at depth 1000: {pred_1000:.4f}")

        is_growing = gds_values[-1] > gds_values[0] * 1.05
        return {
            "name": name,
            "model": model,
            "params": params,
            "growing": is_growing,
            "depths": depths,
            "gds": gds_values,
        }

    return {"name": name, "model": "insufficient data", "growing": None, "depths": depths, "gds": gds_values}


def main():
    print("=" * 80)
    print("UNBOUND CONJECTURE — Comprehensive Investigation")
    print("=" * 80)
    print()
    print("Question: Can GDS grow without bound as game depth increases?")
    print("Testing 6 game classes to characterize growth patterns.")

    results = []

    # Class 1: GoldGame Multiplicative (exponential branching)
    print("\n" + "─" * 80)
    print("CLASS 1: GoldGame Multiplicative (independent exponential growth)")
    r = run_game_class(
        "GoldGame (mult)",
        lambda d: GoldGameMultiplicative(max_turns=d),
        range(3, 18),
        nest_level=10,
    )
    results.append(r)

    # Class 2: GoldGame Additive (linear branching)
    print("\n" + "─" * 80)
    print("CLASS 2: GoldGame Additive (independent linear growth)")
    r = run_game_class(
        "GoldGame (add)",
        lambda d: GoldGameAdditive(max_turns=d),
        range(3, 18),
        nest_level=10,
    )
    results.append(r)

    # Class 3: Combat (HP-bounded)
    print("\n" + "─" * 80)
    print("CLASS 3: Simple Combat (HP-bounded shared state)")
    r = run_game_class(
        "Combat",
        lambda d: ScalableCombat(max_hp=d),
        range(3, 16),
        nest_level=10,
    )
    results.append(r)

    # Class 4: Combat + Resource
    print("\n" + "─" * 80)
    print("CLASS 4: Combat + Accumulation (HP + resource dimension)")
    r = run_game_class(
        "Combat+Resource",
        lambda d: CombatWithResource(max_hp=d, max_resource=3),
        range(3, 12),
        nest_level=10,
    )
    results.append(r)

    # Class 5: Best-of-N
    print("\n" + "─" * 80)
    print("CLASS 5: Best-of-N Coin Toss (increasing rounds)")
    r = run_game_class(
        "Best-of-N",
        lambda d: BestOfN(n_rounds=d * 2 - 1),  # 5, 7, 9, 11, ...
        range(3, 20),
        nest_level=10,
    )
    results.append(r)

    # Class 6: Parallel Lanes
    print("\n" + "─" * 80)
    print("CLASS 6: Parallel Lanes (2 lanes, increasing turns)")
    r = run_game_class(
        "Parallel Lanes",
        lambda d: ParallelLanes(max_turns=d, n_lanes=2),
        range(3, 14),
        nest_level=10,
    )
    results.append(r)

    # ─── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"  {'Game Class':<25} {'Growth Model':<15} {'Growing?':<10} {'GDS Range'}")
    print(f"  {'-'*70}")

    for r in results:
        if r["gds"]:
            gds_range = f"{r['gds'][0]:.4f} → {r['gds'][-1]:.4f}"
        else:
            gds_range = "N/A"

        growing = "YES" if r.get("growing") else ("NO" if r.get("growing") is False else "?")
        print(f"  {r['name']:<25} {r['model']:<15} {growing:<10} {gds_range}")

    # Key findings
    print()
    print("  Key findings:")
    growing_classes = [r for r in results if r.get("growing")]
    bounded_classes = [r for r in results if r.get("growing") is False]

    if growing_classes:
        print(f"  → Growing: {', '.join(r['name'] for r in growing_classes)}")
    if bounded_classes:
        print(f"  → Bounded: {', '.join(r['name'] for r in bounded_classes)}")

    # Check if independent vs shared state predicts growth
    print()
    print("  Structural hypothesis:")
    print("    Independent outcome accumulation → unbounded growth")
    print("    Shared state constraints (HP) → bounded growth")
    print("    Resource accumulation + HP → ???")


if __name__ == "__main__":
    main()
