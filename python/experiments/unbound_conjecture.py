"""Experiment: Investigate the Unbound Conjecture.

The Unbound Conjecture states that the sum of higher-order anticipation
components (A₁ + A₂ + A₃ + ...) can grow without bound as the game
becomes deeper (more turns).

We test this with the GoldGame, varying turn count from 3 to 20+.

From C++ reference:
- 5 turns, geometric: GDS = 0.370
- 15 turns, linear: GDS = 0.438
- 20 turns, linear: GDS = 0.534
- 20 turns, geometric: GDS = 0.552

If GDS keeps growing → strong evidence for unboundedness.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze


class SimpleGoldGame:
    """Minimal gold game with configurable turns for conjecture testing."""

    def __init__(self, max_turns, success_chance=0.68, multiplier=1.2):
        self.max_turns = max_turns
        self.success_chance = success_chance
        self.multiplier = multiplier
        self.penalty = 1.0 / multiplier

    def initial_state(self):
        return (1000, 1000, 0)

    def is_terminal(self, state):
        return state[2] >= self.max_turns

    def get_transitions(self, state, config=None):
        _, _, turn = state
        if turn >= self.max_turns:
            return []
        p1, p2, t = state
        hit = self.success_chance
        miss = 1.0 - hit
        result = []
        for p1_hit in range(2):
            for p2_hit in range(2):
                new_p1 = int(p1 * (self.multiplier if p1_hit else self.penalty))
                new_p2 = int(p2 * (self.multiplier if p2_hit else self.penalty))
                prob = (hit if p1_hit else miss) * (hit if p2_hit else miss)
                result.append((prob, (new_p1, new_p2, t + 1)))
        return result

    def compute_intrinsic_desire(self, state):
        if state[2] < self.max_turns:
            return 0.0
        return 1.0 if state[0] > state[1] else 0.0


def run_experiment():
    print("Unbound Conjecture — GDS vs Turn Count")
    print("=" * 70)
    print(f"{'Turns':>5}  {'GDS':>10}  {'States':>8}  {'Time(s)':>8}  {'Components':}")
    print("-" * 70)

    results = []

    for turns in range(3, 16):
        game = SimpleGoldGame(max_turns=turns)
        nest = min(turns, 15)

        t0 = time.time()
        result = analyze(
            initial_state=game.initial_state(),
            is_terminal=game.is_terminal,
            get_transitions=game.get_transitions,
            compute_intrinsic_desire=game.compute_intrinsic_desire,
            nest_level=nest,
        )
        elapsed = time.time() - t0

        components_str = "  ".join(
            f"A{i+1}:{result.gds_components[i]:.4f}"
            for i in range(min(nest, 8))
            if result.gds_components[i] > 0.0001
        )

        results.append({
            "turns": turns,
            "gds": result.game_design_score,
            "states": len(result.states),
            "time": elapsed,
            "components": [result.gds_components[i] for i in range(nest)],
        })

        print(f"{turns:>5}  {result.game_design_score:>10.6f}  {len(result.states):>8}  {elapsed:>8.2f}  {components_str}")

        # Safety: stop if taking too long
        if elapsed > 30:
            print(f"\nStopping — computation becoming too expensive at {turns} turns")
            break

    # Analysis
    print()
    print("Analysis")
    print("=" * 70)

    if len(results) >= 2:
        gds_values = [r["gds"] for r in results]
        monotonic = all(gds_values[i] < gds_values[i + 1] for i in range(len(gds_values) - 1))
        print(f"Monotonically increasing: {monotonic}")
        print(f"GDS range: {gds_values[0]:.4f} → {gds_values[-1]:.4f}")

        if len(results) >= 3:
            # Check if growth rate is increasing, constant, or decreasing
            deltas = [gds_values[i+1] - gds_values[i] for i in range(len(gds_values)-1)]
            avg_delta_first_half = sum(deltas[:len(deltas)//2]) / max(1, len(deltas)//2)
            avg_delta_second_half = sum(deltas[len(deltas)//2:]) / max(1, len(deltas) - len(deltas)//2)
            print(f"Average delta (first half):  {avg_delta_first_half:.6f}")
            print(f"Average delta (second half): {avg_delta_second_half:.6f}")

            if avg_delta_second_half > avg_delta_first_half:
                print("→ Growth rate INCREASING — strong evidence for unboundedness")
            elif avg_delta_second_half > avg_delta_first_half * 0.5:
                print("→ Growth rate roughly constant — linear growth, evidence for unboundedness")
            else:
                print("→ Growth rate decreasing — may converge (needs more data)")


if __name__ == "__main__":
    run_experiment()
