"""Test: Do comeback mechanics reduce GDS in HpGame?

Hypothesis: Artificial comeback mechanics in HpGame reduce GDS,
consistent with the CoinDuel finding that desperation bonuses
decrease GDS by ~6%.

Compares:
  - Base HpGame (HP=5, no healing)
  - HpGameHeal (HP=5, comeback bonus — trailing player deals 2 dmg with 30% chance)
  - Sweep across comeback chances (10%-50%)
  - Per-component breakdown to see WHERE tension is lost
"""

import sys
sys.path.insert(0, "/agent/projects/anticipation-theory-local/python")

from toa.engine import analyze
from toa.games.hpgame import HpGame
from toa.games.hpgame_heal import HpGameHeal


def run_analysis(game_cls, config=None, nest_level=5):
    """Run ToA analysis on a game and return results."""
    result = analyze(
        initial_state=game_cls.initial_state(),
        is_terminal=game_cls.is_terminal,
        get_transitions=game_cls.get_transitions,
        compute_intrinsic_desire=game_cls.compute_intrinsic_desire,
        config=config,
        nest_level=nest_level,
    )
    return result


def get_p_win(result):
    """Get P(win) from initial state's d_global."""
    init = result.states[0]
    return result.state_nodes[init].d_global


def main():
    print("=" * 70)
    print("HpGame Comeback Mechanics — GDS Impact Analysis")
    print("=" * 70)
    print()
    print("Mechanic: trailing player (lower HP) deals 2 damage instead of 1")
    print("          with 'comeback_chance' probability. When tied, no bonus.")
    print()

    # 1. Base HpGame
    base = run_analysis(HpGame)
    base_pwin = get_p_win(base)
    print(f"Base HpGame (HP=5)")
    print(f"  GDS     = {base.game_design_score:.6f}")
    print(f"  P(win)  = {base_pwin:.4f}")
    print(f"  A_k     = {[f'{c:.6f}' for c in base.gds_components[:5]]}")
    print(f"  States  = {len(base.states)}")
    print()

    # 2. HpGameHeal with default 30% comeback chance
    heal_config = HpGameHeal.Config(max_hp=5, heal_chance=0.3)
    heal30 = run_analysis(HpGameHeal, config=heal_config)
    heal30_pwin = get_p_win(heal30)
    print(f"HpGameHeal (HP=5, 30% comeback bonus)")
    print(f"  GDS     = {heal30.game_design_score:.6f}")
    print(f"  P(win)  = {heal30_pwin:.4f}")
    print(f"  A_k     = {[f'{c:.6f}' for c in heal30.gds_components[:5]]}")
    print(f"  States  = {len(heal30.states)}")
    diff_pct = (heal30.game_design_score - base.game_design_score) / base.game_design_score * 100
    print(f"  GDS change: {diff_pct:+.2f}%")
    print()

    # 3. Per-component comparison
    print("-" * 70)
    print("Per-Component Breakdown (where is tension lost?)")
    print("-" * 70)
    print(f"{'Component':>10} {'Base':>10} {'Comeback':>10} {'Diff':>10} {'% Change':>10}")
    print(f"{'─'*10:>10} {'─'*10:>10} {'─'*10:>10} {'─'*10:>10} {'─'*10:>10}")
    for i in range(5):
        b = base.gds_components[i]
        h = heal30.gds_components[i]
        d = h - b
        pct = (d / b * 100) if b != 0 else 0
        print(f"{'A_' + str(i+1):>10} {b:>10.6f} {h:>10.6f} {d:>+10.6f} {pct:>+9.2f}%")
    print()

    # 4. Sweep across comeback chances
    print("-" * 70)
    print("Comeback Chance Sweep")
    print("-" * 70)
    print(f"{'CB %':>6} {'GDS':>10} {'vs Base':>10} {'P(win)':>8} {'A1':>10} {'A2':>10}")
    print(f"{'────':>6} {'───':>10} {'──────':>10} {'──────':>8} {'──':>10} {'──':>10}")

    for cb_pct in [0, 10, 20, 30, 40, 50]:
        if cb_pct == 0:
            result = base
        else:
            cfg = HpGameHeal.Config(max_hp=5, heal_chance=cb_pct / 100.0)
            result = run_analysis(HpGameHeal, config=cfg)

        diff = (result.game_design_score - base.game_design_score) / base.game_design_score * 100
        pwin = get_p_win(result)
        a1 = result.gds_components[0]
        a2 = result.gds_components[1]
        print(f"{cb_pct:>5}% {result.game_design_score:>10.6f} {diff:>+9.2f}% {pwin:>8.4f} {a1:>10.6f} {a2:>10.6f}")

    print()

    # 5. Verdict
    print("=" * 70)
    heal30_diff = (heal30.game_design_score - base.game_design_score) / base.game_design_score * 100
    if heal30_diff < 0:
        print(f"CONFIRMED: Comeback mechanics reduce GDS by {abs(heal30_diff):.2f}%")
        print()
        # Find which component lost the most
        max_loss_idx = 0
        max_loss_pct = 0
        for i in range(5):
            b = base.gds_components[i]
            h = heal30.gds_components[i]
            if b > 0:
                loss = (b - h) / b * 100
                if loss > max_loss_pct:
                    max_loss_pct = loss
                    max_loss_idx = i
        print(f"Largest impact: A_{max_loss_idx+1} reduced by {max_loss_pct:.1f}%")
        print(f"P(win) shifted from {base_pwin:.4f} to {heal30_pwin:.4f} (symmetric game stays ~0.5)")
        print()
        print("Interpretation: Comeback mechanics compress outcome variance.")
        print("When the trailing player can catch up more easily, each")
        print("individual turn matters less — reducing anticipation at")
        print("every nest level, especially A2 (anticipation of anticipation).")
        print()
        print("This confirms the CoinDuel finding generalizes to HP games:")
        print("artificial rubber-banding reduces structural tension.")
    elif heal30_diff > 0:
        print(f"UNEXPECTED: Comeback mechanics increase GDS by {heal30_diff:.2f}%")
        print("This contradicts the CoinDuel finding. Investigate further.")
    else:
        print("NEUTRAL: No significant GDS change.")
    print("=" * 70)


if __name__ == "__main__":
    main()
