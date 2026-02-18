"""Test class/mode system for Spark Duel differentiation.

3 modes that modify base stats, hopefully forming a soft RPS:
- Striker: Zap=3, Brace=0. Aggressive.
- Guardian: Brace=2, Blast=50%. Defensive.
- Trickster: Dodge=55%, HP=6. Evasive.

Run matchup matrix to verify interesting meta-game.
"""

import random

class SparkDuelSim:
    def __init__(self, max_hp=7, blast_dmg=4, blast_hit=0.60,
                 zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1):
        self.max_hp = max_hp
        self.chip = 1
        self.blast_dmg = blast_dmg
        self.blast_hit = blast_hit
        self.zap_dmg = zap_dmg
        self.brace_reduce = brace_reduce
        self.dodge_chance = dodge_chance
        self.dodge_counter = dodge_counter

    def resolve(self, attack, defend):
        if attack == 'blast':
            if random.random() >= self.blast_hit:
                return 0, 0
            if defend == 'brace':
                return max(0, self.blast_dmg - self.brace_reduce), 0
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter
                return self.blast_dmg, 0
        else:
            if defend == 'brace':
                return max(0, self.zap_dmg - self.brace_reduce), 0
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter
                return self.zap_dmg, 0

    def play(self, p1, p2):
        hp1, hp2, cd1, cd2 = self.max_hp, self.max_hp, 0, 0
        for _ in range(20):
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0: break

            atk1 = p1('attack', cd1, hp1, hp2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)
            d1, d2 = self.resolve(atk1, def2)
            hp2 -= d1; hp1 -= d2
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
            if hp1 <= 0 or hp2 <= 0: break

            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            d3, d4 = self.resolve(atk2, def1)
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


MODES = {
    'Base':     dict(max_hp=7, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.40, dodge_counter=1),
    'Striker':  dict(max_hp=7, blast_dmg=4, blast_hit=0.60, zap_dmg=3, brace_reduce=0, dodge_chance=0.40, dodge_counter=1),
    'Guardian': dict(max_hp=7, blast_dmg=4, blast_hit=0.50, zap_dmg=2, brace_reduce=2, dodge_chance=0.40, dodge_counter=1),
    'Trickster':dict(max_hp=6, blast_dmg=4, blast_hit=0.60, zap_dmg=2, brace_reduce=1, dodge_chance=0.55, dodge_counter=1),
}


def make_asymmetric_sim(mode1_name, mode2_name):
    """Create a sim where P1 and P2 have different stats.

    For asymmetric matchups, we need to handle each side's stats separately.
    Since our sim class only has one set of stats, we create a custom play function.
    """
    m1 = MODES[mode1_name]
    m2 = MODES[mode2_name]

    class AsymSim:
        def __init__(self):
            self.chip = 1

        def resolve(self, attack, defend, attacker_stats, defender_stats):
            if attack == 'blast':
                if random.random() >= attacker_stats['blast_hit']:
                    return 0, 0
                if defend == 'brace':
                    return max(0, attacker_stats['blast_dmg'] - defender_stats['brace_reduce']), 0
                else:
                    if random.random() < defender_stats['dodge_chance']:
                        return 0, defender_stats['dodge_counter']
                    return attacker_stats['blast_dmg'], 0
            else:
                if defend == 'brace':
                    return max(0, attacker_stats['zap_dmg'] - defender_stats['brace_reduce']), 0
                else:
                    if random.random() < defender_stats['dodge_chance']:
                        return 0, defender_stats['dodge_counter']
                    return attacker_stats['zap_dmg'], 0

        def play(self, p1, p2):
            hp1 = m1['max_hp']
            hp2 = m2['max_hp']
            cd1, cd2 = 0, 0
            for _ in range(20):
                hp1 -= self.chip
                hp2 -= self.chip
                if hp1 <= 0 or hp2 <= 0: break

                atk1 = p1('attack', cd1, hp1, hp2)
                def2 = p2('defend', cd2, hp2, hp1, atk1)
                d1, d2 = self.resolve(atk1, def2, m1, m2)
                hp2 -= d1; hp1 -= d2
                cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
                if hp1 <= 0 or hp2 <= 0: break

                atk2 = p2('attack', cd2, hp2, hp1)
                def1 = p1('defend', cd1, hp1, hp2, atk2)
                d3, d4 = self.resolve(atk2, def1, m2, m1)
                hp1 -= d3; hp2 -= d4
                cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
                if hp1 <= 0 or hp2 <= 0: break

            if hp1 <= 0 and hp2 <= 0: return 'draw'
            return 'p1' if hp2 <= 0 else ('p2' if hp1 <= 0 else 'timeout')

    return AsymSim()


def main():
    random.seed(42)
    N = 20000
    fun = make_fun()

    mode_names = list(MODES.keys())

    print("=" * 70)
    print("SPARK DUEL — MODE SYSTEM TEST")
    print("=" * 70)

    # Show mode stats
    print("\nMode Stats:")
    print(f"{'Mode':<12} {'HP':>3} {'Blast':>8} {'Zap':>4} {'Brace':>6} {'Dodge':>6}")
    for name, m in MODES.items():
        print(f"{name:<12} {m['max_hp']:>3} {m['blast_dmg']}@{m['blast_hit']:.0%}{'':<2} {m['zap_dmg']:>4} {m['brace_reduce']:>6} {m['dodge_chance']:.0%}+{m['dodge_counter']}")

    # Mirror matchups (same mode vs itself)
    print(f"\nMirror Matchups (same mode vs itself, Fun strategy):")
    for name in mode_names:
        sim = SparkDuelSim(**MODES[name])
        p1w = sum(1 for _ in range(N) if sim.play(fun, fun) == 'p1')
        print(f"  {name:<12} P1 win: {p1w/N*100:.1f}%")

    # Cross-matchup matrix
    print(f"\nCross-Matchup Matrix (P1 win%, N={N}):")
    p1p2 = "P1 \\ P2"
    header = f"{p1p2:<12}"
    for n in mode_names:
        header += f" {n:>10}"
    print(header)
    print("-" * (12 + 11 * len(mode_names)))

    rps_score = {n: 0 for n in mode_names}
    for n1 in mode_names:
        row = f"{n1:<12}"
        for n2 in mode_names:
            if n1 == n2:
                sim = SparkDuelSim(**MODES[n1])
                p1w = sum(1 for _ in range(N) if sim.play(fun, fun) == 'p1')
            else:
                sim = make_asymmetric_sim(n1, n2)
                p1w = sum(1 for _ in range(N) if sim.play(fun, fun) == 'p1')

            wr = p1w / N * 100
            row += f" {wr:>9.1f}%"

            if n1 != n2:
                if wr > 52: rps_score[n1] += 1; rps_score[n2] -= 1
                elif wr < 48: rps_score[n1] -= 1; rps_score[n2] += 1
        print(row)

    print(f"\nRPS Score (positive = tends to win, negative = tends to lose):")
    for n, s in sorted(rps_score.items(), key=lambda x: -x[1]):
        print(f"  {n:<12} {s:>+3}")

    # Check if we have a proper RPS
    print(f"\nRPS Analysis:")
    # Check triangular dominance
    non_base = [n for n in mode_names if n != 'Base']
    for i, a in enumerate(non_base):
        for b in non_base[i+1:]:
            sim_ab = make_asymmetric_sim(a, b)
            sim_ba = make_asymmetric_sim(b, a)
            ab_wr = sum(1 for _ in range(N) if sim_ab.play(fun, fun) == 'p1') / N * 100
            ba_wr = sum(1 for _ in range(N) if sim_ba.play(fun, fun) == 'p1') / N * 100
            avg = (ab_wr + (100 - ba_wr)) / 2
            winner = a if avg > 52 else (b if avg < 48 else 'EVEN')
            print(f"  {a} vs {b}: {avg:.1f}% → {winner}")


if __name__ == '__main__':
    main()
