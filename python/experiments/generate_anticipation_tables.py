"""Generate precalculated anticipation tables for web demos.

Outputs JSON-compatible JavaScript objects mapping game states
to their A1-A5 anticipation values + sum.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toa.engine import analyze
from toa.games.hpgame import HpGame
from toa.games.hpgame_rage import HpGameRage


def generate_table(game_cls, nest_level=5, config=None):
    """Generate anticipation table for a game."""
    kwargs = {
        "initial_state": game_cls.initial_state(),
        "is_terminal": game_cls.is_terminal,
        "get_transitions": game_cls.get_transitions,
        "compute_intrinsic_desire": game_cls.compute_intrinsic_desire,
        "nest_level": nest_level,
    }
    if config:
        kwargs["config"] = config

    analysis = analyze(**kwargs)

    table = {}
    for state, node in analysis.state_nodes.items():
        if isinstance(state, tuple):
            key = ",".join(str(s) for s in state)
        else:
            key = str(state)

        values = [node.a[i] for i in range(nest_level)]
        total = sum(values)
        values.append(total)
        table[key] = values

    return table, analysis.game_design_score, analysis.gds_components[:nest_level]


def format_js_table(table, var_name):
    """Format as JavaScript object."""
    lines = [f"const {var_name} = {{"]
    for key, values in sorted(table.items()):
        formatted = ",".join(f"{v:.6f}" for v in values)
        lines.append(f'    "{key}": [{formatted}],')
    lines.append("};")
    return "\n".join(lines)


if __name__ == "__main__":
    # Standard HpGame
    print("=== Standard HpGame (5,5) ===")
    std_table, std_gds, std_comps = generate_table(HpGame, nest_level=5)
    print(f"GDS: {std_gds:.6f}")
    print(f"Components: {[f'{c:.6f}' for c in std_comps]}")
    print(f"States: {len(std_table)}")
    print()
    print(format_js_table(std_table, "STD_ANTICIPATION"))

    print("\n\n")

    # HpGame Rage (default config)
    print("=== HpGame Rage (5,5,0,0) default config ===")
    rage_table, rage_gds, rage_comps = generate_table(HpGameRage, nest_level=5)
    print(f"GDS: {rage_gds:.6f}")
    print(f"Components: {[f'{c:.6f}' for c in rage_comps]}")
    print(f"States: {len(rage_table)}")
    print()
    print(format_js_table(rage_table, "RAGE_ANTICIPATION"))
