"""Mode system v2 — trying to create a proper RPS cycle.

Key insight from v1: linear stat buffs create hierarchies, not cycles.
For RPS we need: each mode's strength is specifically countered by another.

Approach: define modes by their dominant strategy, then make each dominant
strategy specifically weak against another mode's strength.

- Blaster: Blast=5@55%, Zap=2. Best at burst damage. Weak to high dodge.
- Tank: HP=8, Brace=2, Blast=4@60%. Best at absorbing. Weak to chip/consistent.
- Phantom: Dodge=55%, Counter=2, HP=6. Best at evasion. Weak to sure-hits.

Desired cycle: Blaster > Tank (burst overwhelms defense)
              Tank > Phantom (HP survives counters, brace negates dodges)
              Phantom > Blaster (dodge the big blast, counter punishes)
"""

import random

class AsymSim:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        self.chip = 1

    def resolve(self, attack, defend, atk_stats, def_stats):
        if attack == 'blast':
            if random.random() >= atk_stats['blast_hit']:
                return 0, 0
            if defend == 'brace':
                return max(0, atk_stats['blast_dmg'] - def_stats['brace_reduce']), 0
            else:
                if random.random() < def_stats['dodge_chance']:
                    return 0, def_stats['dodge_counter']
                return atk_stats['blast_dmg'], 0
        else:
            if defend == 'brace':
                return max(0, atk_stats['zap_dmg'] - def_stats['brace_reduce']), 0
            else:
                if random.random() < def_stats['dodge_chance']:
                    return 0, def_stats['dodge_counter']
                return atk_stats['zap_dmg'], 0

    def play(self, p1, p2):
        hp1 = self.m1['max_hp']
        hp2 = self.m2['max_hp']
        cd1, cd2 = 0, 0
        for _ in range(20):
            hp1 -= self.chip; hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0: break

            atk1 = p1('attack', cd1, hp1, hp2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)
            d1, d2 = self.resolve(atk1, def2, self.m1, self.m2)
            hp2 -= d1; hp1 -= d2
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
            if hp1 <= 0 or hp2 <= 0: break

            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            d3, d4 = self.resolve(atk2, def1, self.m2, self.m1)
            hp1 -= d3; hp2 -= d4
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
            if hp1 <= 0 or hp2 <= 0: break

        if hp1 <= 0 and hp2 <= 0: return 'draw'
        return 'p1' if hp2 <= 0 else ('p2' if hp1 <= 0 else 'timeout')


def make_fun(bp=0.80, dp=0.70):
    def s(role, cd, my, opp, opp_atk=None):
        if role == 'attack':
            return 'blast' if cd == 0 and random.random() < bp else 'zap'
        return 'dodge' if random.random() < dp else 'brace'
    return s


def test_config(name, modes, n=20000):
    fun = make_fun()
    mode_names = list(modes.keys())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    print(f"\n  {'Mode':<10} {'HP':>3} {'Blast':>8} {'Zap':>4} {'Brace':>6} {'Dodge':>6} {'Counter':>8}")
    for mn, m in modes.items():
        print(f"  {mn:<10} {m['max_hp']:>3} {m['blast_dmg']}@{m['blast_hit']:.0%}{'':<2} {m['zap_dmg']:>4} {m['brace_reduce']:>6} {m['dodge_chance']:.0%}     {m['dodge_counter']:>8}")

    print(f"\n  Matchup Matrix (P1 win%):")
    header = f"  {'':>10}"
    for mn in mode_names:
        header += f" {mn:>10}"
    print(header)

    wins = {}
    for n1 in mode_names:
        row = f"  {n1:>10}"
        for n2 in mode_names:
            sim = AsymSim(modes[n1], modes[n2])
            p1w = sum(1 for _ in range(n) if sim.play(fun, fun) == 'p1')
            wr = p1w / n * 100
            row += f" {wr:>9.1f}%"
            wins[(n1, n2)] = wr
        print(row)

    # RPS analysis
    print(f"\n  Pairwise (corrected for first-player advantage):")
    non_mirror = [(a, b) for i, a in enumerate(mode_names) for b in mode_names[i+1:]]
    for a, b in non_mirror:
        # Average of A-as-P1 and B-as-P1 to cancel first-player effect
        a_wr = (wins[(a, b)] + (100 - wins[(b, a)])) / 2
        advantage = a_wr - 50
        winner = a if advantage > 2 else (b if advantage < -2 else 'EVEN')
        arrow = '→' if abs(advantage) > 2 else '≈'
        print(f"    {a} vs {b}: {a_wr:.1f}% {arrow} {winner} ({advantage:+.1f}%)")

    # Check RPS
    pairs = {}
    for a, b in non_mirror:
        a_wr = (wins[(a, b)] + (100 - wins[(b, a)])) / 2
        pairs[(a, b)] = a_wr

    return pairs


def main():
    random.seed(42)

    # V2: designed for RPS cycle
    v2 = {
        'Blaster': dict(max_hp=7, blast_dmg=5, blast_hit=0.55, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
        'Tank':    dict(max_hp=8, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=2, dodge_chance=0.30, dodge_counter=1),
        'Phantom': dict(max_hp=6, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.55, dodge_counter=2),
    }
    test_config("V2: Blaster/Tank/Phantom", v2)

    # V3: try different numbers
    v3 = {
        'Blaster': dict(max_hp=7, blast_dmg=5, blast_hit=0.50, zap_dmg=2, brace_reduce=1, dodge_chance=0.35, dodge_counter=1),
        'Tank':    dict(max_hp=8, blast_dmg=4, blast_hit=0.55, zap_dmg=2, brace_reduce=2, dodge_chance=0.35, dodge_counter=1),
        'Phantom': dict(max_hp=6, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.55, dodge_counter=2),
    }
    test_config("V3: adjusted hit rates", v3)

    # V4: more extreme differentiation
    v4 = {
        'Blaster': dict(max_hp=6, blast_dmg=5, blast_hit=0.60, zap_dmg=1, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
        'Tank':    dict(max_hp=9, blast_dmg=3, blast_hit=0.60, zap_dmg=2, brace_reduce=2, dodge_chance=0.30, dodge_counter=1),
        'Phantom': dict(max_hp=6, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=0, dodge_chance=0.60, dodge_counter=2),
    }
    test_config("V4: extreme differentiation", v4)

    # V5: simpler — only change 1 stat each
    v5 = {
        'Blaster': dict(max_hp=7, blast_dmg=5, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
        'Tank':    dict(max_hp=7, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=2, dodge_chance=0.40, dodge_counter=1),
        'Phantom': dict(max_hp=7, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=2),
    }
    test_config("V5: minimal — 1 stat each (B=5/Brace=2/Counter=2)", v5)

    # V6: HP + offensive tradeoff
    v6 = {
        'Blaster': dict(max_hp=6, blast_dmg=5, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
        'Tank':    dict(max_hp=9, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
        'Phantom': dict(max_hp=7, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.55, dodge_counter=2),
    }
    test_config("V6: HP tradeoff (HP=6/B=5, HP=9/base, HP=7/dodge=55%+C=2)", v6)


if __name__ == '__main__':
    main()
