"""GameFlow ↔ ToA Bridge Analysis

The GameFlow paper (Sweetser & Wyeth, 2005) identifies 8 elements of game enjoyment
based on Csikszentmihalyi's Flow theory. ToA provides a mathematical metric.

This experiment maps GameFlow concepts to ToA measurements:

1. Challenge ↔ A₁ (immediate engagement from uncertainty)
   - GameFlow: "challenges must match player skill levels"
   - ToA: A₁ peaks at p=0.5 (perfectly matched challenge)

2. Immersion/Depth ↔ A₂+ (higher-order engagement)
   - GameFlow: "deep but effortless involvement"
   - ToA: Higher-order components capture strategic depth

3. Flow Channel ↔ GDS trajectory
   - GameFlow: balance between boredom (too easy) and anxiety (too hard)
   - ToA: A₁ variance along trajectory shows the "flow channel"

This analysis demonstrates that ToA formalizes many of GameFlow's qualitative
criteria into computable metrics.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.hpgame import HpGame
from toa.games.hpgame_rage import HpGameRage


def analyze_flow_channel():
    """Map GameFlow's Challenge/Skill balance to ToA's A₁ distribution."""

    print("=" * 70)
    print("GameFlow ↔ ToA Bridge Analysis")
    print("=" * 70)

    # Analyze both games
    baseline = analyze(
        initial_state=HpGame.initial_state(),
        is_terminal=HpGame.is_terminal,
        get_transitions=HpGame.get_transitions,
        compute_intrinsic_desire=HpGame.compute_intrinsic_desire,
        nest_level=5,
    )

    config = HpGameRage.Config(critical_chance=0.13)
    optimal = analyze(
        initial_state=HpGameRage.initial_state(),
        is_terminal=HpGameRage.is_terminal,
        get_transitions=HpGameRage.get_transitions,
        compute_intrinsic_desire=HpGameRage.compute_intrinsic_desire,
        config=config,
        nest_level=5,
    )

    # 1. Challenge Mapping (A₁ distribution)
    print("\n1. CHALLENGE MAPPING (GameFlow: Challenge/Skill Balance)")
    print("-" * 60)
    print("  GameFlow says: 'challenges must match player skill levels'")
    print("  ToA says: A₁ peaks when win probability ≈ 0.5")
    print()

    for name, result, game_cls, cfg in [
        ("HpGame", baseline, HpGame, None),
        ("HpGame_Rage", optimal, HpGameRage, config),
    ]:
        a1_values = []
        for s in result.states:
            if not game_cls.is_terminal(s):
                a1_values.append(result.state_nodes[s].a[0])

        if a1_values:
            avg_a1 = sum(a1_values) / len(a1_values)
            max_a1 = max(a1_values)
            min_a1 = min(a1_values)
            # States near A₁=0.5 (theoretical max) = perfect balance
            near_optimal = sum(1 for a in a1_values if a > 0.4) / len(a1_values) * 100
            # States near A₁=0 = boring/certain
            boring = sum(1 for a in a1_values if a < 0.1) / len(a1_values) * 100

            print(f"  {name}:")
            print(f"    Avg A₁ across states: {avg_a1:.4f}")
            print(f"    A₁ range: [{min_a1:.4f}, {max_a1:.4f}]")
            print(f"    States near optimal (A₁>0.4): {near_optimal:.1f}%")
            print(f"    Boring states (A₁<0.1): {boring:.1f}%")
            print()

    # 2. Depth Mapping (Higher-order components)
    print("2. DEPTH MAPPING (GameFlow: Immersion/Deep Involvement)")
    print("-" * 60)
    print("  GameFlow says: 'deep but effortless involvement'")
    print("  ToA says: Higher A₂+ components = deeper strategic engagement")
    print()

    for name, result in [("HpGame", baseline), ("HpGame_Rage", optimal)]:
        components = result.gds_components[:5]
        total = sum(components)
        depth_ratio = sum(components[1:]) / max(total, 0.001)  # A₂-A₅ / total

        print(f"  {name}:")
        print(f"    Components: {' '.join(f'A{i+1}={c:.4f}' for i, c in enumerate(components))}")
        print(f"    Total GDS: {total:.4f}")
        print(f"    Depth ratio (A₂+/total): {depth_ratio:.3f} ({depth_ratio*100:.1f}%)")
        print(f"    → {'SHALLOW' if depth_ratio < 0.4 else 'MODERATE' if depth_ratio < 0.6 else 'DEEP'} engagement")
        print()

    # 3. Concentration Mapping (Workload distribution)
    print("3. CONCENTRATION MAPPING (GameFlow: Player Workload)")
    print("-" * 60)
    print("  GameFlow says: 'high workload but within cognitive limits'")
    print("  ToA insight: State space size × transition count = decision complexity")
    print()

    for name, result, game_cls, cfg in [
        ("HpGame", baseline, HpGame, None),
        ("HpGame_Rage", optimal, HpGameRage, config),
    ]:
        non_terminal = [s for s in result.states if not game_cls.is_terminal(s)]
        total_transitions = sum(len(game_cls.get_transitions(s, cfg)) for s in non_terminal)
        avg_branching = total_transitions / max(1, len(non_terminal))

        print(f"  {name}:")
        print(f"    Total states: {len(result.states)}")
        print(f"    Non-terminal states: {len(non_terminal)}")
        print(f"    Avg branching factor: {avg_branching:.1f}")
        print(f"    → Decision complexity per turn: {avg_branching:.0f} possible outcomes")
        print()

    # 4. Flow Channel Visualization
    print("4. FLOW CHANNEL (GameFlow: Balance Between Boredom and Anxiety)")
    print("-" * 60)
    print("  GameFlow says: skill/challenge mismatch → anxiety (too hard) or apathy (too easy)")
    print("  ToA mapping: D_global shows win probability; A₁ shows engagement")
    print("  'In the zone': 0.3 < D_global < 0.7 AND A₁ > 0.15")
    print()

    for name, result, game_cls in [
        ("HpGame", baseline, HpGame),
        ("HpGame_Rage", optimal, HpGameRage),
    ]:
        in_zone = 0
        total_nonterminal = 0
        for s in result.states:
            if not game_cls.is_terminal(s):
                total_nonterminal += 1
                d = result.state_nodes[s].d_global
                a = result.state_nodes[s].a[0]
                if 0.3 < d < 0.7 and a > 0.15:
                    in_zone += 1

        pct = in_zone / max(1, total_nonterminal) * 100
        print(f"  {name}: {in_zone}/{total_nonterminal} states in flow zone ({pct:.1f}%)")

    # 5. Summary: GameFlow Elements → ToA Metrics
    print(f"\n{'='*70}")
    print("SUMMARY: GameFlow → ToA Metric Mapping")
    print(f"{'='*70}")
    print("""
  GameFlow Element       ToA Metric                Insight
  ─────────────────────  ────────────────────────   ───────────────────────
  Challenge/Skill        A₁ (local anticipation)    Max at p=0.5 (matched)
  Immersion/Depth        A₂+ (nested anticipation)  Captures strategic depth
  Concentration          State space × branching     Decision complexity
  Clear Goals            D_local (binary desire)     Win condition = goal
  Feedback               D_global trajectory         Progress toward win
  Flow (overall)         GDS (game design score)     Single objective metric

  KEY ADVANTAGE of ToA over GameFlow:
  - GameFlow: qualitative criteria, subjective evaluation (1-5 scores)
  - ToA: computable from game definition alone, no subjective input needed
  - ToA can PREDICT engagement; GameFlow can only EVALUATE after the fact
  - ToA enables automated optimization; GameFlow requires human evaluators
""")


if __name__ == "__main__":
    analyze_flow_channel()
