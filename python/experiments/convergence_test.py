"""Convergence vs Divergence — Why games are Unbound but quizzes aren't.

Hypothesis: The Unbound Conjecture holds when state transitions maintain
uncertainty. It fails when state accumulation causes probability convergence.

Test: Build a "quiz" where knowledge DOESN'T converge probability, and see
if GDS grows with length (like a game) or shrinks (like a normal quiz).

Three models:
1. Normal quiz — knowledge increases P(correct) → convergence → Anti-Unbound
2. "Chaotic quiz" — knowledge doesn't affect probability → each question independent
3. "Resetting quiz" — knowledge resets periodically → breaks convergence

If the hypothesis is correct:
- Model 1: GDS decreases with length (Anti-Unbound) ✓ already confirmed
- Model 2: GDS increases with length (Unbound-like) → independent trials = Best-of-N
- Model 3: GDS increases with length if resets are frequent enough
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


def build_normal_quiz(num_q: int, mastery_threshold: int = None):
    """Standard quiz: knowledge accumulates, increases P(correct)."""
    if mastery_threshold is None:
        mastery_threshold = max(2, num_q // 2)
    max_k = num_q

    def initial():
        return (0, 0)

    def is_terminal(state):
        return state[1] >= num_q

    def get_transitions(state, _):
        k, q = state
        if q >= num_q:
            return []
        # Sigmoid probability: P = 1 / (1 + exp(-1.5 * (k - difficulty)))
        difficulty = 1.0 + q * 0.5  # ascending difficulty
        x = 1.5 * (k - difficulty)
        x = max(-10.0, min(10.0, x))
        p_correct = 1.0 / (1.0 + math.exp(-x))
        p_incorrect = 1.0 - p_correct

        new_k_correct = min(k + 1, max_k)
        new_k_incorrect = k
        if new_k_correct == new_k_incorrect:
            return [(1.0, (new_k_correct, q + 1))]
        return [(p_correct, (new_k_correct, q + 1)),
                (p_incorrect, (new_k_incorrect, q + 1))]

    def desire(state):
        k, q = state
        if q < num_q:
            return 0.0
        return 1.0 if k >= mastery_threshold else 0.0

    return initial, is_terminal, get_transitions, desire


def build_chaotic_quiz(num_q: int, mastery_threshold: int = None):
    """Each question is a fair coin flip — knowledge doesn't help.

    This is structurally identical to Best-of-N coin tosses.
    Knowledge still accumulates but probability is always 50%.
    """
    if mastery_threshold is None:
        mastery_threshold = max(2, num_q // 2)
    max_k = num_q

    def initial():
        return (0, 0)

    def is_terminal(state):
        return state[1] >= num_q

    def get_transitions(state, _):
        k, q = state
        if q >= num_q:
            return []
        # ALWAYS 50/50 regardless of knowledge
        new_k_correct = min(k + 1, max_k)
        new_k_incorrect = k
        if new_k_correct == new_k_incorrect:
            return [(1.0, (new_k_correct, q + 1))]
        return [(0.5, (new_k_correct, q + 1)),
                (0.5, (new_k_incorrect, q + 1))]

    def desire(state):
        k, q = state
        if q < num_q:
            return 0.0
        return 1.0 if k >= mastery_threshold else 0.0

    return initial, is_terminal, get_transitions, desire


def build_volatile_quiz(num_q: int, volatility: float = 0.3, mastery_threshold: int = None):
    """Quiz with volatile knowledge — can lose knowledge at any time.

    Each question: P(correct) based on knowledge, but even on correct answer,
    there's a chance of knowledge loss (representing interference/confusion).
    This keeps uncertainty high throughout.
    """
    if mastery_threshold is None:
        mastery_threshold = max(2, num_q // 2)
    max_k = num_q

    def initial():
        return (0, 0)

    def is_terminal(state):
        return state[1] >= num_q

    def get_transitions(state, _):
        k, q = state
        if q >= num_q:
            return []

        # Knowledge affects probability but volatility prevents convergence
        # P(correct) always between 30-70% regardless of knowledge
        base_p = 0.3 + 0.4 * min(k / max(max_k, 1), 1.0)  # caps at 70%
        p_correct = base_p
        p_incorrect = 1.0 - p_correct

        next_q = q + 1

        # On correct: gain 1, but with volatility chance lose 1 instead
        # On incorrect: lose 1 (if possible)
        transitions = []

        # Correct + keep knowledge
        new_k_gain = min(k + 1, max_k)
        # Correct + volatility loss
        new_k_volatile = max(k - 1, 0)

        p_gain = p_correct * (1.0 - volatility)
        p_volatile_correct = p_correct * volatility
        p_loss = p_incorrect

        # Merge same-state transitions
        state_probs = {}
        state_probs[(new_k_gain, next_q)] = state_probs.get((new_k_gain, next_q), 0) + p_gain
        state_probs[(new_k_volatile, next_q)] = state_probs.get((new_k_volatile, next_q), 0) + p_volatile_correct
        new_k_wrong = max(k - 1, 0)
        state_probs[(new_k_wrong, next_q)] = state_probs.get((new_k_wrong, next_q), 0) + p_loss

        for s, p in state_probs.items():
            if p > 1e-10:
                transitions.append((p, s))

        return transitions

    def desire(state):
        k, q = state
        if q < num_q:
            return 0.0
        return 1.0 if k >= mastery_threshold else 0.0

    return initial, is_terminal, get_transitions, desire


def run_model(name, builder, lengths, **kwargs):
    """Run a model across different lengths and track GDS."""
    results = []
    for n in lengths:
        init, is_term, get_trans, desire = builder(n, **kwargs)
        try:
            analysis = analyze(
                initial_state=init(),
                is_terminal=is_term,
                get_transitions=get_trans,
                compute_intrinsic_desire=desire,
                config=None,
                nest_level=10,
            )
            gds = analysis.game_design_score
            comps = analysis.gds_components[:5]
            results.append((n, gds, comps))
        except Exception as e:
            results.append((n, float('nan'), [0]*5))
    return results


def main():
    print("=" * 70)
    print("CONVERGENCE TEST: Why Games Are Unbound But Quizzes Aren't")
    print("=" * 70)

    lengths = [3, 5, 7, 9, 11, 13, 15, 17, 19]

    # --- Model 1: Normal quiz ---
    print("\n--- Model 1: Normal Quiz (knowledge → probability convergence) ---")
    r1 = run_model("Normal", build_normal_quiz, lengths)
    print(f"{'Length':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7}")
    print("-" * 40)
    for n, gds, comps in r1:
        if not math.isnan(gds):
            print(f"{n:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f}")

    # --- Model 2: Chaotic quiz (independent trials) ---
    print("\n--- Model 2: Chaotic Quiz (50/50, knowledge irrelevant) ---")
    r2 = run_model("Chaotic", build_chaotic_quiz, lengths)
    print(f"{'Length':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7}")
    print("-" * 40)
    for n, gds, comps in r2:
        if not math.isnan(gds):
            print(f"{n:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f}")

    # --- Model 3: Volatile quiz ---
    print("\n--- Model 3: Volatile Quiz (30% chance of knowledge loss on correct) ---")
    r3 = run_model("Volatile", build_volatile_quiz, lengths, volatility=0.3)
    print(f"{'Length':>7} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7}")
    print("-" * 40)
    for n, gds, comps in r3:
        if not math.isnan(gds):
            print(f"{n:>7} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: GDS Growth Direction")
    print("=" * 70)

    for name, results in [("Normal Quiz", r1), ("Chaotic Quiz", r2), ("Volatile Quiz", r3)]:
        valid = [(n, g) for n, g, _ in results if not math.isnan(g) and g > 0]
        if len(valid) >= 2:
            first_n, first_g = valid[0]
            last_n, last_g = valid[-1]
            growth = (last_g - first_g) / (last_n - first_n)
            direction = "GROWING (Unbound-like)" if growth > 0.001 else \
                        "SHRINKING (Anti-Unbound)" if growth < -0.001 else "FLAT"
            print(f"\n{name}:")
            print(f"  GDS at {first_n}q: {first_g:.4f}")
            print(f"  GDS at {last_n}q: {last_g:.4f}")
            print(f"  Growth rate: {growth:+.5f}/question")
            print(f"  Direction: {direction}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The key structural property:

UNBOUND: When each step adds INDEPENDENT uncertainty.
  → Best-of-N, combat games, chaotic quiz
  → Each new turn/question creates fresh variance
  → Higher-order components (A₂+) accumulate without bound

ANTI-UNBOUND: When steps reduce future uncertainty via state accumulation.
  → Standard quizzes, skill-building sequences
  → Knowledge compounds → outcomes become predictable
  → Law of large numbers kills variance

BOUNDARY: Volatile quiz tests the transition point.
  → Volatility prevents full convergence but allows knowledge buildup
  → Determines if partial convergence still kills growth

This suggests a general principle:
  GDS(depth) → ∞  iff  state transitions preserve or increase entropy
  GDS(depth) → C   iff  state transitions reduce entropy (convergence)
""")


if __name__ == "__main__":
    main()
