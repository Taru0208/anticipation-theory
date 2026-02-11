"""Narrative Structure Conjecture — Do optimal games produce narrative arcs?

The paper conjectures that optimizing higher-order anticipation (A₂+) creates
natural trade-offs with A₁, producing alternating high/low tension states that
mirror classical narrative structures (Freytag's Pyramid, Three-Act Structure).

This experiment:
1. Traces A₁ values along most-probable game trajectories in HpGame and HpGame_Rage
2. Checks if the A₁ trajectory exhibits narrative-like patterns:
   - Rising action (increasing tension)
   - Climax (peak tension)
   - Falling action / Resolution
3. Compares narrative quality between baseline and optimized games
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.hpgame import HpGame
from toa.games.hpgame_rage import HpGameRage


def trace_trajectories(game_cls, result, config=None, num_traces=10000, label="Game"):
    """Simulate random game trajectories and record A₁ at each step."""
    trajectories = []

    for _ in range(num_traces):
        state = game_cls.initial_state()
        path = []

        while not game_cls.is_terminal(state):
            node = result.state_nodes.get(state)
            if node is None:
                break
            path.append((state, node.a[0], sum(node.a[:5])))

            transitions = game_cls.get_transitions(state, config)
            if not transitions:
                break

            # Random walk based on probabilities
            r = random.random()
            cumulative = 0.0
            for prob, next_state in transitions:
                cumulative += prob
                if r <= cumulative:
                    state = next_state
                    break

        if len(path) >= 2:
            trajectories.append(path)

    return trajectories


def analyze_narrative_pattern(trajectories, label="Game"):
    """Analyze trajectories for narrative structure patterns."""
    print(f"\n{'='*70}")
    print(f"Narrative Analysis: {label}")
    print(f"{'='*70}")
    print(f"Total trajectories: {len(trajectories)}")

    # Bin trajectories by length
    length_bins = {}
    for t in trajectories:
        length = len(t)
        if length not in length_bins:
            length_bins[length] = []
        length_bins[length].append(t)

    # For each length, compute average A₁ at each step position
    # Normalize to [0, 1] position to compare across lengths
    print(f"\nA₁ by step (averaged across trajectories):")
    print(f"{'Length':>7} {'Count':>6} {'Pattern':}")

    all_normalized_arcs = []

    for length in sorted(length_bins.keys()):
        if length < 3:
            continue
        traces = length_bins[length]
        if len(traces) < 10:
            continue

        # Average A₁ at each step
        avg_a1 = [0.0] * length
        for t in traces:
            for i, (state, a1, sum_a) in enumerate(t):
                avg_a1[i] += a1
        avg_a1 = [a / len(traces) for a in avg_a1]

        # Normalize positions to [0, 1]
        normalized = [(i / (length - 1), a) for i, a in enumerate(avg_a1)]
        all_normalized_arcs.append((length, len(traces), normalized, avg_a1))

        # Detect pattern
        peak_idx = avg_a1.index(max(avg_a1))
        peak_position = peak_idx / (length - 1)

        if peak_position < 0.3:
            pattern = "Front-loaded (peak early)"
        elif peak_position < 0.6:
            pattern = "CLIMACTIC (peak middle) — narrative-like"
        elif peak_position < 0.85:
            pattern = "Back-loaded (peak late) — rising tension"
        else:
            pattern = "Final burst (peak at end)"

        a1_str = " ".join(f"{a:.3f}" for a in avg_a1)
        print(f"{length:>7} {len(traces):>6} {pattern}")
        print(f"{'':>14} A₁: {a1_str}")

    # Overall narrative arc analysis
    print(f"\n--- Overall Arc Shape ---")

    # Combine all trajectories normalized to percentage-of-game
    num_bins = 10
    bin_values = [[] for _ in range(num_bins)]

    for t in trajectories:
        if len(t) < 3:
            continue
        for i, (state, a1, sum_a) in enumerate(t):
            pos = i / (len(t) - 1)
            bin_idx = min(int(pos * num_bins), num_bins - 1)
            bin_values[bin_idx].append(a1)

    avg_arc = []
    print(f"{'Position':>10} {'Avg A₁':>8} {'Samples':>8} {'Visual':}")
    for i in range(num_bins):
        if bin_values[i]:
            avg = sum(bin_values[i]) / len(bin_values[i])
            avg_arc.append(avg)
            bar = "█" * int(avg * 60)
            print(f"{i*10:>8}%-{(i+1)*10:<3}% {avg:>8.4f} {len(bin_values[i]):>8} {bar}")
        else:
            avg_arc.append(0)
            print(f"{i*10:>8}%-{(i+1)*10:<3}% {'N/A':>8}")

    # Detect narrative shape
    if len(avg_arc) >= 5:
        first_quarter = sum(avg_arc[:3]) / 3
        middle = sum(avg_arc[3:7]) / 4
        last_quarter = sum(avg_arc[7:]) / max(1, len(avg_arc[7:]))

        print(f"\nFirst 30%  avg A₁: {first_quarter:.4f}")
        print(f"Middle 40% avg A₁: {middle:.4f}")
        print(f"Last 30%   avg A₁: {last_quarter:.4f}")

        if middle > first_quarter and middle > last_quarter:
            shape = "FREYTAG'S PYRAMID — rises, peaks in middle, falls"
        elif last_quarter > middle > first_quarter:
            shape = "RISING ACTION — continuous increase toward climax"
        elif first_quarter > middle and first_quarter > last_quarter:
            shape = "FRONT-LOADED — decreasing tension"
        elif last_quarter > first_quarter and abs(middle - first_quarter) < 0.01:
            shape = "THREE-ACT — setup, confrontation, resolution"
        else:
            shape = "MIXED — no clear classical pattern"

        print(f"Overall shape: {shape}")

    return avg_arc


def compare_games():
    """Compare narrative arcs between HpGame and HpGame_Rage."""

    random.seed(42)

    # Analyze baseline
    print("Analyzing HpGame baseline...")
    baseline_result = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )
    baseline_traces = trace_trajectories(HpGame, baseline_result, label="HpGame")
    arc_baseline = analyze_narrative_pattern(baseline_traces, "HpGame (baseline, GDS=0.430)")

    # Analyze rage (10% crit)
    print("\nAnalyzing HpGame_Rage (10% crit)...")
    config_10 = HpGameRage.Config(critical_chance=0.10)
    rage_result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config_10,
        nest_level=5,
    )
    rage_traces = trace_trajectories(HpGameRage, rage_result, config=config_10, label="HpGame_Rage")
    arc_rage = analyze_narrative_pattern(rage_traces, "HpGame_Rage (10% crit, GDS=0.544)")

    # Analyze optimal rage (13% crit)
    print("\nAnalyzing HpGame_Rage (13% crit, optimal)...")
    config_13 = HpGameRage.Config(critical_chance=0.13)
    opt_result = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config_13,
        nest_level=5,
    )
    opt_traces = trace_trajectories(HpGameRage, opt_result, config=config_13, label="HpGame_Rage_Opt")
    arc_opt = analyze_narrative_pattern(opt_traces, "HpGame_Rage (13% crit, optimal GDS=0.551)")

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*70}")

    def arc_variance(arc):
        """Variance of the arc — higher = more dramatic shape."""
        if not arc:
            return 0
        mean = sum(arc) / len(arc)
        return sum((a - mean) ** 2 for a in arc) / len(arc)

    def arc_peak_position(arc):
        """Position of the peak (0=start, 1=end)."""
        if not arc:
            return 0
        peak = max(arc)
        idx = arc.index(peak)
        return idx / max(1, len(arc) - 1)

    for name, arc, gds in [
        ("HpGame baseline", arc_baseline, 0.430),
        ("HpGame_Rage 10%", arc_rage, 0.544),
        ("HpGame_Rage 13%", arc_opt, 0.551),
    ]:
        var = arc_variance(arc)
        peak_pos = arc_peak_position(arc)
        print(f"\n{name} (GDS={gds:.3f}):")
        print(f"  Arc variance:    {var:.6f} (higher = more dramatic)")
        print(f"  Peak position:   {peak_pos:.2f} (0=start, 0.5=middle, 1=end)")
        print(f"  Arc shape:       {' '.join(f'{a:.3f}' for a in arc)}")


if __name__ == "__main__":
    compare_games()
