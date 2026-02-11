"""MOBA Lane Phase Model — Extending LaneGame toward realism.

Builds on LaneGame (simple RPS minion contest) by adding MOBA-specific mechanics:
1. Gold from last-hitting (economic dimension)
2. Level-ups from experience (power scaling)
3. Item power spikes (gold thresholds give combat advantage)
4. Kill potential (low HP → risk of being killed)

The goal is to understand which MOBA mechanics contribute most to engagement
(measured by GDS) and whether ToA can explain why MOBAs are addictive.

State: (gold1, gold2, level1, level2, hp1, hp2, turn)
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


class MobaLane:
    """Simplified MOBA laning model.

    Each turn: both players attempt to last-hit minions.
    Outcomes depend on level/item advantage.

    Simplifications:
    - HP is abstract (3 levels: safe/risky/danger)
    - Gold is discretized into tiers (0-4)
    - Levels from 1-3
    - 10 minion waves (turns)
    """

    class Config:
        def __init__(
            self,
            max_waves=8,
            gold_per_cs=1,       # Gold tiers gained per successful last-hit
            kill_gold=2,          # Gold tiers gained from a kill
            item_threshold=3,     # Gold tier for item spike
            item_advantage=0.10,  # Extra win prob from item
            level_advantage=0.08, # Extra win prob per level diff
        ):
            self.max_waves = max_waves
            self.gold_per_cs = gold_per_cs
            self.kill_gold = kill_gold
            self.item_threshold = item_threshold
            self.item_advantage = item_advantage
            self.level_advantage = level_advantage

    @staticmethod
    def initial_state():
        # (gold1, gold2, level1, level2, hp1, hp2, wave)
        # gold: 0-6, level: 1-3, hp: 1-3 (3=safe, 1=danger), wave: 0-max
        return (0, 0, 1, 1, 3, 3, 0)

    @staticmethod
    def is_terminal(state):
        # Default terminal check — for custom max_waves, use make_is_terminal()
        return state[6] >= 8 or state[4] <= 0 or state[5] <= 0

    @staticmethod
    def make_is_terminal(config):
        """Create a terminal check function with config-specific max_waves."""
        def is_terminal(state):
            return state[6] >= config.max_waves or state[4] <= 0 or state[5] <= 0
        return is_terminal

    @staticmethod
    def get_transitions(state, config=None):
        if config is None:
            config = MobaLane.Config()

        g1, g2, l1, l2, hp1, hp2, wave = state
        if wave >= config.max_waves or hp1 <= 0 or hp2 <= 0:
            return []

        # Calculate advantage
        level_diff = l1 - l2
        has_item1 = 1 if g1 >= config.item_threshold else 0
        has_item2 = 1 if g2 >= config.item_threshold else 0

        # Base probability: 1/3 each for win/draw/loss (like LaneGame)
        # Modified by advantages
        advantage = (level_diff * config.level_advantage +
                     (has_item1 - has_item2) * config.item_advantage)

        p_win = max(0.05, min(0.60, 1.0 / 3.0 + advantage))
        p_loss = max(0.05, min(0.60, 1.0 / 3.0 - advantage))
        p_draw = 1.0 - p_win - p_loss

        # Level up every 3 waves (simplified)
        new_l1 = min(3, l1 + (1 if (wave + 1) % 3 == 0 else 0))
        new_l2 = min(3, l2 + (1 if (wave + 1) % 3 == 0 else 0))

        transitions = []

        # Outcome 1: P1 wins trade (gets CS, P2 takes HP damage)
        new_g1_w = min(6, g1 + config.gold_per_cs)
        new_hp2_w = max(0, hp2 - 1)
        transitions.append((p_win, (new_g1_w, g2, new_l1, new_l2, hp1, new_hp2_w, wave + 1)))

        # Outcome 2: Draw (both get some CS, no HP change)
        new_g1_d = min(6, g1 + config.gold_per_cs)
        new_g2_d = min(6, g2 + config.gold_per_cs)
        transitions.append((p_draw, (new_g1_d, new_g2_d, new_l1, new_l2, hp1, hp2, wave + 1)))

        # Outcome 3: P1 loses trade (P2 gets CS, P1 takes HP damage)
        new_g2_l = min(6, g2 + config.gold_per_cs)
        new_hp1_l = max(0, hp1 - 1)
        transitions.append((p_loss, (g1, new_g2_l, new_l1, new_l2, new_hp1_l, hp2, wave + 1)))

        return sanitize_transitions(transitions)

    @staticmethod
    def compute_intrinsic_desire(state):
        g1, g2, l1, l2, hp1, hp2, wave = state
        if hp1 <= 0 and hp2 > 0:
            return 0.0  # P1 died
        if hp2 <= 0 and hp1 > 0:
            return 1.0  # P1 killed P2
        if hp1 > 0 and hp2 > 0:
            return 0.0  # Not terminal (or wave end not reached)
        # End of laning: compare gold + level advantage
        score1 = g1 + l1 * 2
        score2 = g2 + l2 * 2
        return 1.0 if score1 > score2 else 0.0

    @staticmethod
    def make_desire(config):
        """Create desire function with config-specific behavior."""
        def desire(state):
            g1, g2, l1, l2, hp1, hp2, wave = state
            if hp1 <= 0 and hp2 > 0:
                return 0.0
            if hp2 <= 0 and hp1 > 0:
                return 1.0
            if wave < config.max_waves and hp1 > 0 and hp2 > 0:
                return 0.0
            score1 = g1 + l1 * 2
            score2 = g2 + l2 * 2
            return 1.0 if score1 > score2 else 0.0
        return desire

    @staticmethod
    def tostr(state):
        return f"G({state[0]},{state[1]}) L({state[2]},{state[3]}) HP({state[4]},{state[5]}) W{state[6]}"


def run_experiment():
    print("MOBA Lane Phase Model — ToA Analysis")
    print("=" * 75)

    # Baseline analysis
    config = MobaLane.Config()
    is_term = MobaLane.make_is_terminal(config)
    desire_fn = MobaLane.make_desire(config)
    result = analyze(
        initial_state=MobaLane.initial_state(),
        is_terminal=is_term,
        get_transitions=MobaLane.get_transitions,
        compute_intrinsic_desire=desire_fn,
        config=config,
        nest_level=5,
    )

    non_terminal = [s for s in result.states if not is_term(s)]
    d0 = result.state_nodes[MobaLane.initial_state()].d_global

    print(f"\nBaseline MOBA Lane:")
    print(f"  GDS: {result.game_design_score:.6f}")
    print(f"  Components: {' '.join(f'A{i+1}={result.gds_components[i]:.4f}' for i in range(5) if result.gds_components[i] > 0.001)}")
    print(f"  States: {len(result.states)} ({len(non_terminal)} non-terminal)")
    print(f"  D₀: {d0:.4f}")

    # A₁ distribution
    a1_vals = [result.state_nodes[s].a[0] for s in non_terminal]
    near_max = sum(1 for a in a1_vals if a > 0.4) / max(1, len(a1_vals)) * 100
    boring = sum(1 for a in a1_vals if a < 0.1) / max(1, len(a1_vals)) * 100
    print(f"  A₁ range: [{min(a1_vals):.4f}, {max(a1_vals):.4f}]")
    print(f"  States near max (A₁>0.4): {near_max:.1f}%")
    print(f"  Boring (A₁<0.1): {boring:.1f}%")

    # Compare: what happens with different mechanics?
    print()
    print("=" * 75)
    print("Mechanic Impact Analysis")
    print("=" * 75)
    print(f"{'Config':<40} {'GDS':>8} {'A₁':>8} {'A₂':>8} {'States':>8}")
    print("-" * 70)

    configs = [
        ("Baseline (items + levels)", MobaLane.Config()),
        ("No items (item_threshold=99)", MobaLane.Config(item_threshold=99)),
        ("No level advantage", MobaLane.Config(level_advantage=0.0)),
        ("No items, no levels", MobaLane.Config(item_threshold=99, level_advantage=0.0)),
        ("Strong items (15% adv)", MobaLane.Config(item_advantage=0.15)),
        ("Strong levels (12% adv)", MobaLane.Config(level_advantage=0.12)),
        ("Kill gold = 3", MobaLane.Config(kill_gold=3)),
        ("Early item spike (tier 2)", MobaLane.Config(item_threshold=2)),
        ("Short lane (5 waves)", MobaLane.Config(max_waves=5)),
    ]

    for name, cfg in configs:
        is_term = MobaLane.make_is_terminal(cfg)
        desire_fn = MobaLane.make_desire(cfg)
        r = analyze(
            initial_state=MobaLane.initial_state(),
            is_terminal=is_term,
            get_transitions=MobaLane.get_transitions,
            compute_intrinsic_desire=desire_fn,
            config=cfg,
            nest_level=5,
        )
        print(f"{name:<40} {r.game_design_score:>8.4f} "
              f"{r.gds_components[0]:>8.4f} {r.gds_components[1]:>8.4f} "
              f"{len(r.states):>8}")

    # Most engaging moments in the MOBA model
    print()
    print("=" * 75)
    print("Most Engaging Moments (Baseline)")
    print("=" * 75)

    state_scores = [(s, result.state_nodes[s].sum_a()) for s in non_terminal]
    state_scores.sort(key=lambda x: -x[1])

    print(f"{'State':<45} {'A₁':>8} {'Total':>8}")
    print("-" * 65)
    for s, total in state_scores[:10]:
        a1 = result.state_nodes[s].a[0]
        print(f"  {MobaLane.tostr(s):<43} {a1:>8.4f} {total:>8.4f}")

    # Comparison with other games
    print()
    print("=" * 75)
    print("Cross-Game Comparison")
    print("=" * 75)

    from toa.games.hpgame import HpGame
    from toa.games.hpgame_rage import HpGameRage
    from toa.games.lanegame import LaneGame

    comparisons = [
        ("HpGame", HpGame, None),
        ("HpGame_Rage (13%)", HpGameRage, HpGameRage.Config(critical_chance=0.13)),
        ("LaneGame (basic)", LaneGame, None),
        ("MOBA Lane", MobaLane, MobaLane.Config()),
    ]

    print(f"{'Game':<30} {'GDS':>8} {'A₁':>8} {'A₂':>8} {'Depth%':>8}")
    print("-" * 60)
    for name, cls, cfg in comparisons:
        r = analyze(
            initial_state=cls.initial_state(),
            is_terminal=cls.is_terminal,
            get_transitions=cls.get_transitions,
            compute_intrinsic_desire=cls.compute_intrinsic_desire,
            config=cfg,
            nest_level=5,
        )
        total = r.game_design_score
        depth = sum(r.gds_components[1:5]) / max(0.001, total) * 100
        print(f"{name:<30} {total:>8.4f} {r.gds_components[0]:>8.4f} "
              f"{r.gds_components[1]:>8.4f} {depth:>7.1f}%")

    print()
    print("INSIGHTS:")
    print("  Depth% = fraction of engagement from higher-order (A₂+) components")
    print("  Higher depth% → more strategic depth, longer engagement potential")


if __name__ == "__main__":
    run_experiment()
