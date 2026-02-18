"""ToA engine verification of Spark Duel v5 candidates.

Reuses the PI/CPG computation from spark_duel_v4_analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toa.engine import analyze
from toa.games.spark_duel_v4 import SparkDuelV4
from toa.game import sanitize_transitions


def analyze_game(config):
    initial = SparkDuelV4.initial_state(config)
    result = analyze(
        initial_state=initial,
        is_terminal=SparkDuelV4.is_terminal,
        get_transitions=lambda s, c=None, cf=config: SparkDuelV4.get_transitions(s, cf),
        compute_intrinsic_desire=SparkDuelV4.compute_intrinsic_desire,
        nest_level=5,
    )
    gds = result.game_design_score
    d0 = result.state_nodes[initial].d_global
    comps = result.gds_components[:5]
    depth = sum(comps[1:]) / gds if gds > 0 else 0
    states = len(result.state_nodes)
    return gds, d0, comps, depth, states


def compute_pi_cpg(config, resolution=5):
    initial = SparkDuelV4.initial_state(config)
    gc = SparkDuelV4

    best_gds = (0, 0, None)
    best_d0 = (0, 0, None)
    all_results = []

    for bi in range(resolution + 1):
        for di in range(resolution + 1):
            blast_pct = bi / resolution
            dodge_pct = di / resolution

            def make_transitions(state, _c=None, cf=config,
                                 bp=blast_pct, dp=dodge_pct):
                if gc.is_terminal(state):
                    return []
                hp1, hp2, cd1, cd2, phase = state

                if phase == 0:
                    attacks = gc._available_attacks(cd1)
                else:
                    attacks = gc._available_attacks(cd2)
                defends = [gc.BRACE, gc.DODGE]

                if len(attacks) == 1:
                    atk_weights = {attacks[0]: 1.0}
                else:
                    atk_weights = {gc.BLAST: bp, gc.ZAP: 1.0 - bp}

                def_weights = {gc.DODGE: dp, gc.BRACE: 1.0 - dp}

                transitions = []
                for attack, aw in atk_weights.items():
                    if aw <= 0:
                        continue
                    for defend, dw in def_weights.items():
                        if dw <= 0:
                            continue
                        new_cd_atk = 1 if attack == gc.BLAST else 0
                        outcomes = gc._resolve_attack(attack, defend, cf)

                        if phase == 0:
                            hp1_c = max(0, hp1 - cf.chip_damage)
                            hp2_c = max(0, hp2 - cf.chip_damage)
                            if hp1_c <= 0 or hp2_c <= 0:
                                transitions.append((aw * dw, (hp1_c, hp2_c, cd1, cd2, 0)))
                                continue
                            for prob, dmg_def, dmg_atk in outcomes:
                                new_hp2 = max(0, hp2_c - dmg_def)
                                new_hp1 = max(0, hp1_c - dmg_atk)
                                next_state = (new_hp1, new_hp2, new_cd_atk, cd2, 1)
                                transitions.append((aw * dw * prob, next_state))
                        else:
                            for prob, dmg_def, dmg_atk in outcomes:
                                new_hp1 = max(0, hp1 - dmg_def)
                                new_hp2 = max(0, hp2 - dmg_atk)
                                next_state = (new_hp1, new_hp2, cd1, new_cd_atk, 0)
                                transitions.append((aw * dw * prob, next_state))

                return sanitize_transitions(transitions)

            result = analyze(
                initial_state=initial,
                is_terminal=gc.is_terminal,
                get_transitions=make_transitions,
                compute_intrinsic_desire=gc.compute_intrinsic_desire,
                nest_level=5,
            )
            gds = result.game_design_score
            d0 = result.state_nodes[initial].d_global

            all_results.append((blast_pct, dodge_pct, gds, d0))

            if gds > best_gds[0]:
                best_gds = (gds, d0, (blast_pct, dodge_pct))
            if d0 > best_d0[0]:
                best_d0 = (d0, gds, (blast_pct, dodge_pct))

    pi_values = [r[2] for r in all_results]
    pi = max(pi_values) - min(pi_values)
    cpg = abs(best_gds[1] - best_d0[0])

    return pi, cpg, best_gds, best_d0


def run_config(name, config):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  HP={config.max_hp} B={config.blast_damage}@{config.blast_hit_rate:.0%} "
          f"Z={config.zap_damage} Br=-{config.brace_reduction} "
          f"D={config.dodge_chance:.0%}+C={config.dodge_counter}")
    print(f"{'='*60}")

    gds, d0, comps, depth, states = analyze_game(config)
    print(f"  GDS={gds:.4f}  D₀={d0:.4f}  Depth={depth:.0%}  States={states}")
    print(f"  A₁-A₅: {[f'{a:.4f}' for a in comps]}")

    pi, cpg, fun_opt, win_opt = compute_pi_cpg(config, resolution=5)
    print(f"  PI={pi:.4f} ({pi/gds*100:.1f}% of GDS)")
    print(f"  CPG={cpg:.4f}")
    print(f"  Fun-opt: B={fun_opt[2][0]:.0%} D={fun_opt[2][1]:.0%}  GDS={fun_opt[0]:.4f} D₀={fun_opt[1]:.4f}")
    print(f"  Win-opt: B={win_opt[2][0]:.0%} D={win_opt[2][1]:.0%}  D₀={win_opt[0]:.4f} GDS={win_opt[1]:.4f}")

    return {'gds': gds, 'd0': d0, 'depth': depth, 'pi': pi, 'cpg': cpg,
            'fun': fun_opt, 'win': win_opt, 'states': states}


def main():
    print("Spark Duel v4 → v5 ToA Verification")
    print("Comparing current config vs D₀-balanced candidates\n")

    configs = [
        ("v4 CURRENT", SparkDuelV4.Config(
            max_hp=7, blast_damage=4, zap_damage=2,
            dodge_chance=0.30, dodge_counter=1)),

        ("v5a: B=3, D=35%, C=2", SparkDuelV4.Config(
            max_hp=7, blast_damage=3, zap_damage=2,
            dodge_chance=0.35, dodge_counter=2)),

        ("v5b: B=3, D=30%, C=2", SparkDuelV4.Config(
            max_hp=7, blast_damage=3, zap_damage=2,
            dodge_chance=0.30, dodge_counter=2)),

        ("v5c: B=3, D=40%, C=2", SparkDuelV4.Config(
            max_hp=7, blast_damage=3, zap_damage=2,
            dodge_chance=0.40, dodge_counter=2)),

        ("v5d: HP=9, B=4, D=30%, C=1", SparkDuelV4.Config(
            max_hp=9, blast_damage=4, zap_damage=2,
            dodge_chance=0.30, dodge_counter=1)),

        ("v5e: B=4, D=40%, C=1", SparkDuelV4.Config(
            max_hp=7, blast_damage=4, zap_damage=2,
            dodge_chance=0.40, dodge_counter=1)),
    ]

    results = {}
    for name, config in configs:
        results[name] = run_config(name, config)

    # Summary
    print("\n\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)
    print(f"{'Config':<28} {'GDS':>6} {'D₀':>6} {'Depth':>6} {'PI':>6} {'PI%':>5} {'CPG':>6} {'States':>7}")
    print("-" * 90)
    for name, r in results.items():
        short = name[:27]
        pi_pct = r['pi']/r['gds']*100 if r['gds'] > 0 else 0
        d0_mark = "✓" if abs(r['d0'] - 0.50) <= 0.03 else " "
        cpg_mark = "✓" if r['cpg'] <= 0.02 else " "
        print(f"{short:<28} {r['gds']:>6.3f} {r['d0']:>6.3f}{d0_mark} {r['depth']:>5.0%}  {r['pi']:>6.3f} {pi_pct:>4.0f}% {r['cpg']:>6.3f}{cpg_mark} {r['states']:>7}")


if __name__ == '__main__':
    main()
