"""Refined parameter search around promising configurations.

Focus: D₀ ≈ 0.50, blowout < 45%, Fun WR > 55%, nothing% = 0%, 2-3 rounds avg.
"""

import random
from collections import Counter

class SparkDuelSim:
    def __init__(self, max_hp=7, chip=1, blast_dmg=4, blast_hit=0.60,
                 zap_dmg=2, brace_reduce=1, dodge_chance=0.30, dodge_counter=1):
        self.max_hp = max_hp
        self.chip = chip
        self.blast_dmg = blast_dmg
        self.blast_hit = blast_hit
        self.zap_dmg = zap_dmg
        self.brace_reduce = brace_reduce
        self.dodge_chance = dodge_chance
        self.dodge_counter = dodge_counter

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

    def play(self, p1, p2):
        hp1, hp2, cd1, cd2 = self.max_hp, self.max_hp, 0, 0
        rounds = 0
        nothing = 0
        total_events = 0
        for rnd in range(20):
            rounds += 1
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0:
                break

            atk1 = p1('attack', cd1, hp1, hp2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)
            d_def, d_atk, ev1 = self.resolve(atk1, def2)
            hp2 -= d_def
            hp1 -= d_atk
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
            total_events += 1
            if ev1 == 'zap-brace' and d_def == 0:
                nothing += 1
            if hp1 <= 0 or hp2 <= 0:
                break

            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            d_def2, d_atk2, ev2 = self.resolve(atk2, def1)
            hp1 -= d_def2
            hp2 -= d_atk2
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
            total_events += 1
            if ev2 == 'zap-brace' and d_def2 == 0:
                nothing += 1
            if hp1 <= 0 or hp2 <= 0:
                break

        winner = 'draw' if hp1 <= 0 and hp2 <= 0 else ('p1' if hp2 <= 0 else 'p2' if hp1 <= 0 else 'timeout')
        return winner, hp1, hp2, rounds, nothing, total_events


def make_mixed(blast_pct, dodge_pct):
    def strategy(role, cd, my_hp, opp_hp, opp_atk=None):
        if role == 'attack':
            if cd > 0:
                return 'zap'
            return 'blast' if random.random() < blast_pct else 'zap'
        else:
            return 'dodge' if random.random() < dodge_pct else 'brace'
    return strategy


def analyze(sim, n=15000):
    fun = make_mixed(0.80, 0.70)
    win = make_mixed(0.00, 0.50)

    p1w = 0
    blowouts = 0
    total_rounds = 0
    total_nothing = 0
    total_events = 0
    for _ in range(n):
        w, hp1, hp2, rnds, noth, evts = sim.play(fun, fun)
        if w == 'p1': p1w += 1
        if max(hp1, hp2) >= 3: blowouts += 1
        total_rounds += rnds
        total_nothing += noth
        total_events += evts

    fvw = sum(1 for _ in range(n) if sim.play(fun, win)[0] == 'p1')
    wvf = sum(1 for _ in range(n) if sim.play(win, fun)[0] == 'p2')
    fun_wr = (fvw + wvf) / (2 * n)

    return {
        'p1_wr': p1w / n,
        'blowout': blowouts / n,
        'fun_wr': fun_wr,
        'avg_rounds': total_rounds / n,
        'nothing_pct': total_nothing / total_events if total_events > 0 else 0,
    }


def score(r):
    """Score a config: higher is better."""
    d0_penalty = abs(r['p1_wr'] - 0.50) * 10  # 0 is ideal
    blowout_penalty = max(0, r['blowout'] - 0.35) * 5
    fun_bonus = r['fun_wr']
    nothing_penalty = r['nothing_pct'] * 10
    round_penalty = abs(r['avg_rounds'] - 2.5) * 2  # 2.5 rounds ideal
    return fun_bonus - d0_penalty - blowout_penalty - nothing_penalty - round_penalty


def main():
    random.seed(42)
    N = 15000

    configs = []
    # Systematic sweep
    for hp in [7, 8, 9, 10]:
        for blast in [3, 4, 5]:
            for zap in [1, 2]:
                for dodge in [0.25, 0.30, 0.35, 0.40]:
                    for counter in [0, 1, 2]:
                        brace_reduce = 1
                        # Skip obviously bad combos
                        if zap <= brace_reduce and counter == 0:
                            continue  # nothing events
                        if blast >= hp:
                            continue  # one-shot
                        configs.append((
                            f"HP={hp} B={blast} Z={zap} D={dodge:.0%} C={counter}",
                            SparkDuelSim(max_hp=hp, blast_dmg=blast, zap_dmg=zap,
                                        dodge_chance=dodge, dodge_counter=counter)
                        ))

    print(f"Testing {len(configs)} configurations...")
    results = []
    for name, sim in configs:
        r = analyze(sim, N)
        s = score(r)
        results.append((name, r, s))

    # Sort by score
    results.sort(key=lambda x: -x[2])

    print(f"\n{'Config':<30} {'P1%':>5} {'Blow%':>6} {'FunWR':>6} {'Rnds':>5} {'Noth%':>6} {'Score':>7}")
    print("-" * 72)
    for name, r, s in results[:20]:
        d0 = "✓" if abs(r['p1_wr'] - 0.50) <= 0.03 else " "
        print(f"{name:<30} {r['p1_wr']*100:>4.1f}% {r['blowout']*100:>5.1f}% {r['fun_wr']*100:>5.1f}% {r['avg_rounds']:>5.2f} {r['nothing_pct']*100:>5.1f}% {s:>7.3f} {d0}")

    # Also show current config for comparison
    print("\n--- Current config for comparison ---")
    current = SparkDuelSim()
    r = analyze(current, N)
    s = score(r)
    print(f"{'HP=7 B=4 Z=2 D=30% C=1':<30} {r['p1_wr']*100:>4.1f}% {r['blowout']*100:>5.1f}% {r['fun_wr']*100:>5.1f}% {r['avg_rounds']:>5.2f} {r['nothing_pct']*100:>5.1f}% {s:>7.3f}")


if __name__ == '__main__':
    main()
