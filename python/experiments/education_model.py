"""Education Model — Applying ToA to Learning Sequences.

Models a quiz/learning sequence as a "game" to analyze engagement using
Theory of Anticipation. Key question: does adaptive difficulty kill
anticipation the same way Nash equilibrium kills GDS in games?

State: (knowledge_level, question_index)
Transitions: correct/incorrect answers based on knowledge vs difficulty
Terminal: all questions answered
Desire: reaching mastery = 1.0

Experiments:
1. Difficulty curve comparison (flat, ascending, descending, adaptive)
2. Adaptive difficulty paradox — does matching reduce anticipation?
3. Forgetting mechanics — does knowledge decay add engagement?
4. Binary vs graduated success criteria
"""

import sys
import os
import math
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze


# --- Model ---

@dataclass
class QuizConfig:
    """Configuration for a quiz sequence."""
    num_questions: int = 8          # total questions
    max_knowledge: int = 6          # maximum knowledge level
    mastery_threshold: int = 4      # knowledge >= this = success
    difficulties: list = None       # difficulty per question (0-indexed)
    difficulty_mode: str = "fixed"  # "fixed", "adaptive", "curve"
    base_difficulty: float = 3.0    # for fixed mode
    correct_gain: int = 1           # knowledge gained on correct
    incorrect_loss: int = 0         # knowledge lost on incorrect (0 = stays)
    sigmoid_steepness: float = 1.5  # how sharply probability changes
    graduated_desire: bool = False  # True = knowledge/max, False = binary pass/fail

    def __post_init__(self):
        if self.difficulties is None:
            if self.difficulty_mode == "fixed":
                self.difficulties = [self.base_difficulty] * self.num_questions
            # Other modes set difficulties dynamically


def correct_probability(knowledge: int, difficulty: float, steepness: float = 1.5) -> float:
    """Probability of answering correctly given knowledge and difficulty.

    Uses sigmoid: P = 1 / (1 + exp(-steepness * (knowledge - difficulty)))
    When knowledge == difficulty, P = 0.5
    """
    x = steepness * (knowledge - difficulty)
    # Clamp to avoid overflow
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def get_difficulty(config: QuizConfig, question_idx: int, knowledge: int) -> float:
    """Get difficulty for the current question."""
    if config.difficulty_mode == "adaptive":
        # Match difficulty to student's current knowledge
        return float(knowledge)
    elif config.difficulties and question_idx < len(config.difficulties):
        return config.difficulties[question_idx]
    return config.base_difficulty


def quiz_initial_state(config: QuizConfig):
    """Starting state: knowledge=0, question=0."""
    return (0, 0)


def quiz_is_terminal(state, config: QuizConfig):
    """Terminal when all questions answered."""
    knowledge, q_idx = state
    return q_idx >= config.num_questions


def quiz_get_transitions(state, config: QuizConfig):
    """Transitions based on correct/incorrect answer."""
    knowledge, q_idx = state
    if q_idx >= config.num_questions:
        return []

    difficulty = get_difficulty(config, q_idx, knowledge)
    p_correct = correct_probability(knowledge, difficulty, config.sigmoid_steepness)
    p_incorrect = 1.0 - p_correct

    transitions = []

    # Correct answer
    new_k_correct = min(knowledge + config.correct_gain, config.max_knowledge)
    transitions.append((p_correct, (new_k_correct, q_idx + 1)))

    # Incorrect answer
    new_k_incorrect = max(knowledge - config.incorrect_loss, 0)
    # If both outcomes lead to the same state, merge them
    if (new_k_correct, q_idx + 1) == (new_k_incorrect, q_idx + 1):
        return [(1.0, (new_k_correct, q_idx + 1))]
    transitions.append((p_incorrect, (new_k_incorrect, q_idx + 1)))

    return transitions


def quiz_compute_desire(state, config: QuizConfig):
    """Desire for terminal states."""
    knowledge, q_idx = state
    if q_idx < config.num_questions:
        return 0.0
    if config.graduated_desire:
        return knowledge / config.max_knowledge
    return 1.0 if knowledge >= config.mastery_threshold else 0.0


def analyze_quiz(config: QuizConfig):
    """Run ToA analysis on a quiz configuration."""
    return analyze(
        initial_state=quiz_initial_state(config),
        is_terminal=lambda s: quiz_is_terminal(s, config),
        get_transitions=lambda s, _: quiz_get_transitions(s, config),
        compute_intrinsic_desire=lambda s: quiz_compute_desire(s, config),
        config=None,
        nest_level=10,
    )


# --- Difficulty Curves ---

def make_flat_curve(num_q: int, level: float) -> list:
    """All questions same difficulty."""
    return [level] * num_q


def make_ascending_curve(num_q: int, start: float, end: float) -> list:
    """Gradually increasing difficulty."""
    if num_q <= 1:
        return [start]
    return [start + (end - start) * i / (num_q - 1) for i in range(num_q)]


def make_descending_curve(num_q: int, start: float, end: float) -> list:
    """Gradually decreasing difficulty."""
    return list(reversed(make_ascending_curve(num_q, end, start)))


def make_inverted_u_curve(num_q: int, start: float, peak: float) -> list:
    """Easy → hard → easy (peak in middle)."""
    mid = num_q // 2
    result = []
    for i in range(num_q):
        if i <= mid:
            t = i / max(mid, 1)
        else:
            t = (num_q - 1 - i) / max(num_q - 1 - mid, 1)
        result.append(start + (peak - start) * t)
    return result


def make_u_curve(num_q: int, peak: float, valley: float) -> list:
    """Hard → easy → hard (valley in middle)."""
    mid = num_q // 2
    result = []
    for i in range(num_q):
        if i <= mid:
            t = i / max(mid, 1)
        else:
            t = (num_q - 1 - i) / max(num_q - 1 - mid, 1)
        result.append(peak - (peak - valley) * t)
    return result


def make_step_curve(num_q: int, levels: list) -> list:
    """Step function — difficulty jumps at intervals."""
    result = []
    step_size = num_q / len(levels)
    for i in range(num_q):
        level_idx = min(int(i / step_size), len(levels) - 1)
        result.append(levels[level_idx])
    return result


# --- Experiments ---

def experiment1_difficulty_curves():
    """Compare engagement across different difficulty curve shapes.

    Key question: which difficulty progression maximizes GDS?
    """
    print("=" * 70)
    print("EXPERIMENT 1: Difficulty Curve Comparison")
    print("=" * 70)

    num_q = 8
    curves = {
        "Flat (easy, d=1)":      make_flat_curve(num_q, 1.0),
        "Flat (medium, d=3)":    make_flat_curve(num_q, 3.0),
        "Flat (hard, d=5)":      make_flat_curve(num_q, 5.0),
        "Ascending (1→5)":       make_ascending_curve(num_q, 1.0, 5.0),
        "Descending (5→1)":      make_descending_curve(num_q, 5.0, 1.0),
        "Inverted-U (1→5→1)":   make_inverted_u_curve(num_q, 1.0, 5.0),
        "U-shape (5→1→5)":      make_u_curve(num_q, 5.0, 1.0),
        "Step (1,3,5)":          make_step_curve(num_q, [1.0, 3.0, 5.0]),
    }

    results = []
    for name, diffs in curves.items():
        config = QuizConfig(
            num_questions=num_q,
            difficulties=diffs,
            difficulty_mode="curve",
            mastery_threshold=4,
        )
        analysis = analyze_quiz(config)
        gds = analysis.game_design_score
        components = analysis.gds_components[:5]
        results.append((name, gds, components, diffs))

    # Sort by GDS
    results.sort(key=lambda x: -x[1])

    print(f"\n{'Curve Shape':<25} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7} {'A₂+/Total':>10}")
    print("-" * 70)
    for name, gds, comps, diffs in results:
        a1 = comps[0]
        a2_plus = sum(comps[1:])
        depth_ratio = a2_plus / gds * 100 if gds > 0 else 0
        print(f"{name:<25} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f} {depth_ratio:>9.1f}%")

    best_name, best_gds, _, _ = results[0]
    worst_name, worst_gds, _, _ = results[-1]
    print(f"\nBest:  {best_name} (GDS = {best_gds:.4f})")
    print(f"Worst: {worst_name} (GDS = {worst_gds:.4f})")
    print(f"Ratio: {best_gds/worst_gds:.1f}x" if worst_gds > 0 else "")

    return results


def experiment2_adaptive_paradox():
    """The Adaptive Difficulty Paradox.

    Does matching difficulty to student knowledge reduce anticipation?
    Compare adaptive vs various fixed difficulties.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Adaptive Difficulty Paradox")
    print("=" * 70)

    num_q = 8
    results = []

    # Adaptive mode
    config_adaptive = QuizConfig(
        num_questions=num_q,
        difficulty_mode="adaptive",
        mastery_threshold=4,
    )
    analysis = analyze_quiz(config_adaptive)
    results.append(("Adaptive (d=knowledge)", analysis.game_design_score,
                     analysis.gds_components[:5]))

    # Fixed difficulties for comparison
    for d in [0, 1, 2, 3, 4, 5, 6]:
        config = QuizConfig(
            num_questions=num_q,
            difficulties=make_flat_curve(num_q, float(d)),
            difficulty_mode="curve",
            mastery_threshold=4,
        )
        analysis = analyze_quiz(config)
        results.append((f"Fixed (d={d})", analysis.game_design_score,
                         analysis.gds_components[:5]))

    # Slightly-above-adaptive: difficulty = knowledge + 1 (always slightly harder)
    # We need a custom mode for this
    config_above = QuizConfig(
        num_questions=num_q,
        difficulty_mode="adaptive",
        mastery_threshold=4,
    )
    # Override get_difficulty for "slightly above"
    original_get_diff = get_difficulty

    def above_adaptive_transitions(state, config_inner):
        knowledge, q_idx = state
        if q_idx >= num_q:
            return []
        # difficulty = knowledge + 1 (always slightly harder)
        difficulty = knowledge + 1.0
        p_correct = correct_probability(knowledge, difficulty, config_inner.sigmoid_steepness)
        p_incorrect = 1.0 - p_correct
        new_k_correct = min(knowledge + 1, config_inner.max_knowledge)
        new_k_incorrect = knowledge
        if new_k_correct == new_k_incorrect:
            return [(1.0, (new_k_correct, q_idx + 1))]
        return [(p_correct, (new_k_correct, q_idx + 1)),
                (p_incorrect, (new_k_incorrect, q_idx + 1))]

    analysis_above = analyze(
        initial_state=(0, 0),
        is_terminal=lambda s: s[1] >= num_q,
        get_transitions=lambda s, _: above_adaptive_transitions(s, config_above),
        compute_intrinsic_desire=lambda s: quiz_compute_desire(s, config_above),
        config=None,
        nest_level=10,
    )
    results.append(("Slightly-above (d=k+1)", analysis_above.game_design_score,
                     analysis_above.gds_components[:5]))

    # Slightly-below-adaptive: difficulty = knowledge - 1
    def below_adaptive_transitions(state, config_inner):
        knowledge, q_idx = state
        if q_idx >= num_q:
            return []
        difficulty = max(knowledge - 1.0, 0.0)
        p_correct = correct_probability(knowledge, difficulty, config_inner.sigmoid_steepness)
        p_incorrect = 1.0 - p_correct
        new_k_correct = min(knowledge + 1, config_inner.max_knowledge)
        new_k_incorrect = knowledge
        if new_k_correct == new_k_incorrect:
            return [(1.0, (new_k_correct, q_idx + 1))]
        return [(p_correct, (new_k_correct, q_idx + 1)),
                (p_incorrect, (new_k_incorrect, q_idx + 1))]

    analysis_below = analyze(
        initial_state=(0, 0),
        is_terminal=lambda s: s[1] >= num_q,
        get_transitions=lambda s, _: below_adaptive_transitions(s, config_above),
        compute_intrinsic_desire=lambda s: quiz_compute_desire(s, config_above),
        config=None,
        nest_level=10,
    )
    results.append(("Slightly-below (d=k-1)", analysis_below.game_design_score,
                     analysis_below.gds_components[:5]))

    results.sort(key=lambda x: -x[1])

    print(f"\n{'Mode':<28} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7} {'Depth%':>8}")
    print("-" * 70)
    for name, gds, comps in results:
        depth = sum(comps[1:]) / gds * 100 if gds > 0 else 0
        print(f"{name:<28} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f} {depth:>7.1f}%")

    # Find adaptive and best fixed
    adaptive_gds = next(gds for name, gds, _ in results if "Adaptive" in name)
    fixed_results = [(name, gds) for name, gds, _ in results if "Fixed" in name]
    best_fixed = max(fixed_results, key=lambda x: x[1])

    print(f"\nAdaptive GDS:    {adaptive_gds:.4f}")
    print(f"Best fixed GDS:  {best_fixed[1]:.4f} ({best_fixed[0]})")
    diff_pct = (adaptive_gds - best_fixed[1]) / best_fixed[1] * 100
    print(f"Difference:      {diff_pct:+.1f}%")

    if adaptive_gds < best_fixed[1]:
        print("\n*** ADAPTIVE DIFFICULTY PARADOX CONFIRMED ***")
        print("Matching difficulty to student reduces engagement!")
        print("Same pattern as Choice Paradox: reducing uncertainty reduces anticipation.")

    return results


def experiment3_forgetting():
    """Does knowledge decay between questions add engagement?

    Compare: no forgetting vs occasional forgetting (knowledge -1 on wrong).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Forgetting Mechanics")
    print("=" * 70)

    num_q = 8
    # Use ascending difficulty for a natural learning context
    diffs = make_ascending_curve(num_q, 1.0, 5.0)

    results = []
    for loss, label in [(0, "No forgetting"), (1, "Forget-1 on wrong"), (2, "Forget-2 on wrong")]:
        config = QuizConfig(
            num_questions=num_q,
            difficulties=diffs,
            difficulty_mode="curve",
            mastery_threshold=4,
            incorrect_loss=loss,
        )
        analysis = analyze_quiz(config)
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        results.append((label, gds, comps))

    print(f"\n{'Mode':<25} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7}")
    print("-" * 55)
    for name, gds, comps in results:
        print(f"{name:<25} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f}")

    base = results[0][1]
    for name, gds, _ in results[1:]:
        diff = (gds - base) / base * 100
        print(f"\n{name} vs No forgetting: {diff:+.1f}%")

    return results


def experiment4_graduated_vs_binary():
    """Binary pass/fail vs graduated success.

    Binary: knowledge >= threshold → desire 1.0, else 0.0
    Graduated: desire = knowledge / max_knowledge (continuous)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Binary vs Graduated Success")
    print("=" * 70)

    num_q = 8
    curves_to_test = {
        "Ascending (1→5)": make_ascending_curve(num_q, 1.0, 5.0),
        "Flat (d=3)":      make_flat_curve(num_q, 3.0),
        "Adaptive":        None,  # special handling
    }

    results = []
    for curve_name, diffs in curves_to_test.items():
        for graduated in [False, True]:
            mode = "Graduated" if graduated else "Binary"
            label = f"{curve_name} / {mode}"

            if diffs is None:  # adaptive
                config = QuizConfig(
                    num_questions=num_q,
                    difficulty_mode="adaptive",
                    mastery_threshold=4,
                    graduated_desire=graduated,
                )
            else:
                config = QuizConfig(
                    num_questions=num_q,
                    difficulties=diffs,
                    difficulty_mode="curve",
                    mastery_threshold=4,
                    graduated_desire=graduated,
                )

            analysis = analyze_quiz(config)
            gds = analysis.game_design_score
            comps = analysis.gds_components[:5]
            results.append((label, gds, comps))

    print(f"\n{'Mode':<35} {'GDS':>8} {'A₁':>7} {'A₂':>7}")
    print("-" * 55)
    for name, gds, comps in results:
        print(f"{name:<35} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f}")

    return results


def experiment5_question_count_scaling():
    """How does the number of questions affect GDS?

    Parallels Unbound Conjecture: does GDS grow with quiz length?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Question Count Scaling (Unbound Parallel)")
    print("=" * 70)

    results = []
    for num_q in [3, 5, 8, 10, 12, 15, 20]:
        # Scale mastery threshold proportionally
        threshold = max(2, num_q // 2)
        max_k = num_q  # Can potentially learn as much as there are questions
        diffs = make_ascending_curve(num_q, 1.0, float(num_q) * 0.6)

        config = QuizConfig(
            num_questions=num_q,
            max_knowledge=max_k,
            mastery_threshold=threshold,
            difficulties=diffs,
            difficulty_mode="curve",
        )
        analysis = analyze_quiz(config)
        gds = analysis.game_design_score
        comps = analysis.gds_components[:5]
        results.append((num_q, gds, comps))

    print(f"\n{'Questions':>10} {'GDS':>8} {'A₁':>7} {'A₂':>7} {'A₃':>7} {'States':>8}")
    print("-" * 55)
    for num_q, gds, comps in results:
        # Estimate states (rough)
        states_est = num_q * (num_q + 1)
        print(f"{num_q:>10} {gds:>8.4f} {comps[0]:>7.4f} {comps[1]:>7.4f} {comps[2]:>7.4f} {states_est:>8}")

    # Growth rate
    if len(results) >= 2:
        first_q, first_gds, _ = results[0]
        last_q, last_gds, _ = results[-1]
        growth = (last_gds - first_gds) / (last_q - first_q)
        print(f"\nGDS growth rate: ~{growth:.4f}/question")

    return results


def experiment6_optimal_difficulty_search():
    """Search for the optimal difficulty curve via parameter sweep.

    For 8-question quiz, sweep base difficulties to find peak GDS.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Optimal Difficulty Search")
    print("=" * 70)

    num_q = 8
    best_gds = 0
    best_params = None
    results = []

    # Sweep start and end difficulty for ascending curves
    for start_d in [x * 0.5 for x in range(0, 13)]:  # 0.0 to 6.0
        for end_d in [x * 0.5 for x in range(0, 13)]:
            diffs = make_ascending_curve(num_q, start_d, end_d)
            config = QuizConfig(
                num_questions=num_q,
                difficulties=diffs,
                difficulty_mode="curve",
                mastery_threshold=4,
            )
            analysis = analyze_quiz(config)
            gds = analysis.game_design_score
            results.append((start_d, end_d, gds))

            if gds > best_gds:
                best_gds = gds
                best_params = (start_d, end_d)

    print(f"\nBest linear curve: start={best_params[0]}, end={best_params[1]}")
    print(f"Best GDS: {best_gds:.4f}")

    # Show top 10
    results.sort(key=lambda x: -x[2])
    print(f"\n{'Start':>7} {'End':>7} {'GDS':>8}")
    print("-" * 25)
    for start_d, end_d, gds in results[:10]:
        print(f"{start_d:>7.1f} {end_d:>7.1f} {gds:>8.4f}")

    # Show heatmap-like summary
    print(f"\nHeatmap (Start × End difficulty, GDS values):")
    ends = sorted(set(x[1] for x in results))
    starts = sorted(set(x[0] for x in results))
    gds_map = {(s, e): g for s, e, g in results}

    print(f"{'':>5}", end="")
    for e in ends[::2]:  # every other for readability
        print(f" {e:>5.1f}", end="")
    print()

    for s in starts[::2]:
        print(f"{s:>5.1f}", end="")
        for e in ends[::2]:
            g = gds_map.get((s, e), 0)
            print(f" {g:>5.3f}", end="")
        print()

    return best_params, best_gds, results


def experiment7_controlled_mastery_probability():
    """Control for mastery probability — does easy win just because it's winnable?

    The key confound: easy quizzes have high GDS partly because the student
    can actually reach mastery. Hard quizzes have near-zero mastery probability,
    so the desire signal barely propagates.

    Fix: set mastery threshold to 1 (very low) so even hard quizzes are winnable.
    Then compare difficulty levels on equal footing.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Controlling for Mastery Probability")
    print("=" * 70)

    num_q = 8

    # Low threshold: just need knowledge >= 1 (almost everyone passes)
    print("\n--- Low threshold (mastery >= 1) ---")
    results_low = []
    for d in [0, 1, 2, 3, 4, 5]:
        config = QuizConfig(
            num_questions=num_q,
            difficulties=make_flat_curve(num_q, float(d)),
            difficulty_mode="curve",
            mastery_threshold=1,
        )
        analysis = analyze_quiz(config)
        # Also compute mastery probability from initial state
        d0 = analysis.state_nodes[(0, 0)].d_global
        gds = analysis.game_design_score
        results_low.append((d, gds, d0, analysis.gds_components[:5]))

    print(f"{'Diff':>5} {'GDS':>8} {'P(pass)':>8} {'A₁':>7} {'A₂':>7}")
    print("-" * 40)
    for d, gds, d0, comps in results_low:
        print(f"{d:>5} {gds:>8.4f} {d0:>8.3f} {comps[0]:>7.4f} {comps[1]:>7.4f}")

    # Mid threshold: need knowledge >= 4
    print("\n--- Mid threshold (mastery >= 4) ---")
    results_mid = []
    for d in [0, 1, 2, 3, 4, 5]:
        config = QuizConfig(
            num_questions=num_q,
            difficulties=make_flat_curve(num_q, float(d)),
            difficulty_mode="curve",
            mastery_threshold=4,
        )
        analysis = analyze_quiz(config)
        d0 = analysis.state_nodes[(0, 0)].d_global
        gds = analysis.game_design_score
        results_mid.append((d, gds, d0, analysis.gds_components[:5]))

    print(f"{'Diff':>5} {'GDS':>8} {'P(pass)':>8} {'A₁':>7} {'A₂':>7}")
    print("-" * 40)
    for d, gds, d0, comps in results_mid:
        print(f"{d:>5} {gds:>8.4f} {d0:>8.3f} {comps[0]:>7.4f} {comps[1]:>7.4f}")

    # Key insight: plot GDS vs P(pass)
    print("\n--- GDS vs Mastery Probability (all configs) ---")
    all_results = results_low + results_mid
    all_results.sort(key=lambda x: x[2])  # sort by P(pass)
    print(f"{'P(pass)':>8} {'GDS':>8} {'Config':>20}")
    print("-" * 40)
    for d, gds, d0, _ in all_results:
        threshold = 1 if (d, gds, d0, _) in [(x[0], x[1], x[2], x[3]) for x in results_low] else 4
        print(f"{d0:>8.3f} {gds:>8.4f}   d={d}, t={threshold}")

    return results_low, results_mid


def experiment8_mastery_proximity():
    """What creates the most anticipation: proximity to mastery.

    Like HP games where the exciting state is (1,1), the exciting learning
    state should be "one question away from passing/failing."

    Measure: for each state, what's its anticipation? Map the "excitement
    landscape" of the quiz.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Excitement Landscape — State-Level Anticipation")
    print("=" * 70)

    num_q = 8
    # Use a moderate difficulty to see the full landscape
    config = QuizConfig(
        num_questions=num_q,
        difficulties=make_ascending_curve(num_q, 0.5, 3.0),
        difficulty_mode="curve",
        mastery_threshold=4,
    )
    analysis = analyze_quiz(config)

    print(f"\nAnticipation map (knowledge × question):")
    print(f"{'':>8}", end="")
    for q in range(num_q):
        print(f"  Q{q:>2}", end="")
    print()

    for k in range(config.max_knowledge + 1):
        marker = " ←mastery" if k == config.mastery_threshold else ""
        print(f"K={k:>2}   ", end="")
        for q in range(num_q):
            state = (k, q)
            if state in analysis.state_nodes:
                a1 = analysis.state_nodes[state].a[0]
                print(f" {a1:.3f}" if a1 > 0.001 else "     .", end="")
            else:
                print("     -", end="")
        print(marker)

    # Find peak excitement states
    peak_states = []
    for state, node in analysis.state_nodes.items():
        if not quiz_is_terminal(state, config):
            peak_states.append((state, node.a[0], node.sum_a()))

    peak_states.sort(key=lambda x: -x[2])
    print(f"\nTop 5 most exciting states (by total A):")
    for state, a1, total_a in peak_states[:5]:
        k, q = state
        diff = get_difficulty(config, q, k)
        p_correct = correct_probability(k, diff, config.sigmoid_steepness)
        print(f"  K={k}, Q={q}: A₁={a1:.4f}, Total={total_a:.4f}, P(correct)={p_correct:.3f}, diff={diff:.1f}")

    # Find the "boring" states
    print(f"\nBottom 5 least exciting states (by total A):")
    non_zero = [(s, a1, ta) for s, a1, ta in peak_states if ta > 0.0001]
    for state, a1, total_a in non_zero[-5:]:
        k, q = state
        diff = get_difficulty(config, q, k)
        p_correct = correct_probability(k, diff, config.sigmoid_steepness)
        print(f"  K={k}, Q={q}: A₁={a1:.4f}, Total={total_a:.4f}, P(correct)={p_correct:.3f}, diff={diff:.1f}")

    return analysis


def main():
    """Run all experiments."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  ToA Education Model — Learning Sequence Engagement Analysis  ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    r1 = experiment1_difficulty_curves()
    r2 = experiment2_adaptive_paradox()
    r3 = experiment3_forgetting()
    r4 = experiment4_graduated_vs_binary()
    r5 = experiment5_question_count_scaling()
    r6 = experiment6_optimal_difficulty_search()
    r7 = experiment7_controlled_mastery_probability()
    r8 = experiment8_mastery_proximity()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    # Adaptive paradox
    adaptive_data = next((name, gds, c) for name, gds, c in r2 if "Adaptive" in name)
    best_fixed = max(((name, gds) for name, gds, _ in r2 if "Fixed" in name), key=lambda x: x[1])
    adaptive_diff = (adaptive_data[1] - best_fixed[1]) / best_fixed[1] * 100

    print(f"\n1. Difficulty Curves:")
    print(f"   Best curve: {r1[0][0]} (GDS = {r1[0][1]:.4f})")
    print(f"   Worst curve: {r1[-1][0]} (GDS = {r1[-1][1]:.4f})")

    print(f"\n2. Adaptive Difficulty:")
    print(f"   Adaptive GDS = {adaptive_data[1]:.4f}")
    print(f"   Best fixed = {best_fixed[1]:.4f}")
    print(f"   {'PARADOX CONFIRMED' if adaptive_data[1] < best_fixed[1] else 'No paradox'}: {adaptive_diff:+.1f}%")

    print(f"\n3. Forgetting:")
    base_gds = r3[0][1]
    for name, gds, _ in r3[1:]:
        diff = (gds - base_gds) / base_gds * 100
        print(f"   {name}: {diff:+.1f}% vs baseline")

    print(f"\n4. Success Criteria: Binary vs Graduated")
    for i in range(0, len(r4), 2):
        bin_gds = r4[i][1]
        grad_gds = r4[i+1][1]
        diff = (grad_gds - bin_gds) / bin_gds * 100 if bin_gds > 0 else 0
        curve = r4[i][0].split("/")[0].strip()
        print(f"   {curve}: binary={bin_gds:.4f}, graduated={grad_gds:.4f} ({diff:+.1f}%)")

    print(f"\n5. Question Count Scaling:")
    for num_q, gds, _ in r5:
        print(f"   {num_q} questions: GDS = {gds:.4f}")

    print(f"\n6. Optimal Difficulty Curve:")
    print(f"   Start={r6[0][0]}, End={r6[0][1]}, GDS={r6[1]:.4f}")


if __name__ == "__main__":
    main()
