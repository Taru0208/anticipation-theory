"""Analyze D₀ asymmetry fixes for Spark Duel.

The core problem: P1 attacks first → can kill P2 before P2 gets a chance.
This creates D₀ = 0.553 (P1 advantage) and 52% blowouts.

Approaches tested:
A. Current: sequential (P1 atk first)
B. Simultaneous attacks (both attack at once, then both defend)
C. Higher HP (dilute single-hit impact)
D. Lower blast damage
E. "Retaliation" — if killed during attack phase, still get your attack
"""

import random
from collections import Counter

class SparkDuelSim:
    def __init__(self, max_hp=7, chip=1, blast_dmg=4, blast_hit=0.60,
                 zap_dmg=2, brace_reduce=1, dodge_chance=0.30, dodge_counter=1,
                 simultaneous=False, retaliation=False):
        self.max_hp = max_hp
        self.chip = chip
        self.blast_dmg = blast_dmg
        self.blast_hit = blast_hit
        self.zap_dmg = zap_dmg
        self.brace_reduce = brace_reduce
        self.dodge_chance = dodge_chance
        self.dodge_counter = dodge_counter
        self.simultaneous = simultaneous
        self.retaliation = retaliation

    def resolve(self, attack, defend):
        if attack == 'blast':
            if random.random() >= self.blast_hit:
                return 0, 0, 'blast-miss'
            if defend == 'brace':
                return max(0, self.blast_dmg - self.brace_reduce), 0, 'blast-brace'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'blast-dodged'
                return self.blast_dmg, 0, 'blast-dodge-fail'
        else:
            if defend == 'brace':
                dmg = max(0, self.zap_dmg - self.brace_reduce)
                return dmg, 0, 'zap-brace'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'zap-dodged'
                return self.zap_dmg, 0, 'zap-hit'

    def play_sequential(self, p1, p2):
        hp1, hp2, cd1, cd2 = self.max_hp, self.max_hp, 0, 0
        for rnd in range(20):
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0:
                break

            # P1 attacks P2
            atk1 = p1('attack', cd1, hp1, hp2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)
            d_def, d_atk, _ = self.resolve(atk1, def2)
            hp2 -= d_def
            hp1 -= d_atk
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)

            if not self.retaliation and (hp1 <= 0 or hp2 <= 0):
                break

            # P2 attacks P1
            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            d_def2, d_atk2, _ = self.resolve(atk2, def1)
            hp1 -= d_def2
            hp2 -= d_atk2
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
            if hp1 <= 0 or hp2 <= 0:
                break

        return 'draw' if hp1 <= 0 and hp2 <= 0 else ('p1' if hp2 <= 0 else 'p2' if hp1 <= 0 else 'timeout'), hp1, hp2

    def play_simultaneous(self, p1, p2):
        """Both players attack simultaneously, then both defend simultaneously."""
        hp1, hp2, cd1, cd2 = self.max_hp, self.max_hp, 0, 0
        for rnd in range(20):
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0:
                break

            # Both choose attack and defense simultaneously
            atk1 = p1('attack', cd1, hp1, hp2)
            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)

            # Resolve both at once
            d_def1, d_atk1, _ = self.resolve(atk1, def2)  # P1 attacks P2
            d_def2, d_atk2, _ = self.resolve(atk2, def1)  # P2 attacks P1

            hp2 -= d_def1  # P2 takes damage from P1's attack
            hp1 -= d_atk1  # P1 takes counter from P2's dodge
            hp1 -= d_def2  # P1 takes damage from P2's attack
            hp2 -= d_atk2  # P2 takes counter from P1's dodge

            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
            if hp1 <= 0 or hp2 <= 0:
                break

        return 'draw' if hp1 <= 0 and hp2 <= 0 else ('p1' if hp2 <= 0 else 'p2' if hp1 <= 0 else 'timeout'), hp1, hp2

    def play(self, p1, p2):
        if self.simultaneous:
            return self.play_simultaneous(p1, p2)
        return self.play_sequential(p1, p2)


def make_mixed(blast_pct, dodge_pct):
    def strategy(role, cd, my_hp, opp_hp, opp_atk=None):
        if role == 'attack':
            if cd > 0:
                return 'zap'
            return 'blast' if random.random() < blast_pct else 'zap'
        else:
            return 'dodge' if random.random() < dodge_pct else 'brace'
    return strategy


def run_analysis(sim, n=20000):
    fun = make_mixed(0.80, 0.70)
    win = make_mixed(0.00, 0.50)

    # Mirror
    p1_wins = 0
    blowouts = 0
    lead_changes_total = 0
    for _ in range(n):
        w, hp1, hp2 = sim.play(fun, fun)
        if w == 'p1':
            p1_wins += 1
        if max(hp1, hp2) >= 3:
            blowouts += 1

    # Fun vs Win (both directions)
    fvw_p1 = sum(1 for _ in range(n) if sim.play(fun, win)[0] == 'p1')
    wvf_p2 = sum(1 for _ in range(n) if sim.play(win, fun)[0] == 'p2')
    fun_wr = (fvw_p1 + wvf_p2) / (2 * n)

    return {
        'p1_wr': p1_wins / n,
        'blowout': blowouts / n,
        'fun_wr': fun_wr,
    }


def main():
    random.seed(42)
    N = 20000

    configs = [
        ("A. Current (HP=7,B=4,seq)", SparkDuelSim()),
        ("B. Simultaneous attacks", SparkDuelSim(simultaneous=True)),
        ("C. Retaliation rule", SparkDuelSim(retaliation=True)),
        ("D. HP=9, Blast=4", SparkDuelSim(max_hp=9)),
        ("E. HP=7, Blast=3", SparkDuelSim(blast_dmg=3)),
        ("F. HP=9, Blast=5", SparkDuelSim(max_hp=9, blast_dmg=5)),
        ("G. HP=7, Dodge=40%", SparkDuelSim(dodge_chance=0.40)),
        ("H. HP=7, Counter=2", SparkDuelSim(dodge_counter=2)),
        ("I. Simul + HP=7,B=4", SparkDuelSim(simultaneous=True)),
        ("J. Simul + HP=9,B=5", SparkDuelSim(simultaneous=True, max_hp=9, blast_dmg=5)),
        ("K. HP=7,B=4,Zap=1", SparkDuelSim(zap_dmg=1)),
        ("L. HP=8,B=4", SparkDuelSim(max_hp=8)),
    ]

    print(f"{'Config':<30} {'P1 win%':>8} {'Blowout%':>9} {'Fun WR':>8}")
    print("-" * 60)
    for name, sim in configs:
        r = run_analysis(sim, N)
        d0_marker = "⚠️" if abs(r['p1_wr'] - 0.50) > 0.04 else "✓"
        blow_marker = "⚠️" if r['blowout'] > 0.40 else "✓"
        print(f"{name:<30} {r['p1_wr']*100:>7.1f}% {r['blowout']*100:>8.1f}% {r['fun_wr']*100:>7.1f}% {d0_marker} {blow_marker}")


if __name__ == '__main__':
    main()
