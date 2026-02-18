"""Quick sim verification of v5e: HP=7, B=4@60%, Z=2, Dodge=40%, Counter=1."""

import random

class SparkDuelSim:
    def __init__(self, dodge_chance=0.40):
        self.max_hp = 7
        self.chip = 1
        self.blast_dmg = 4
        self.blast_hit = 0.60
        self.zap_dmg = 2
        self.brace_reduce = 1
        self.dodge_chance = dodge_chance
        self.dodge_counter = 1

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
                return max(0, self.zap_dmg - self.brace_reduce), 0, 'zap-brace'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'zap-dodged'
                return self.zap_dmg, 0, 'zap-hit'

    def play(self, p1, p2):
        hp1, hp2, cd1, cd2 = self.max_hp, self.max_hp, 0, 0
        nothing = 0
        total_events = 0
        rounds = 0
        for _ in range(20):
            rounds += 1
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0: break

            atk1 = p1('attack', cd1, hp1, hp2)
            def2 = p2('defend', cd2, hp2, hp1, atk1)
            d1, d2, ev = self.resolve(atk1, def2)
            hp2 -= d1; hp1 -= d2
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)
            total_events += 1
            if hp1 <= 0 or hp2 <= 0: break

            atk2 = p2('attack', cd2, hp2, hp1)
            def1 = p1('defend', cd1, hp1, hp2, atk2)
            d3, d4, ev2 = self.resolve(atk2, def1)
            hp1 -= d3; hp2 -= d4
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)
            total_events += 1
            if hp1 <= 0 or hp2 <= 0: break

        winner = 'draw' if hp1 <= 0 and hp2 <= 0 else ('p1' if hp2 <= 0 else 'p2' if hp1 <= 0 else 'timeout')
        return winner, hp1, hp2, rounds


def make_mixed(bp, dp):
    def s(role, cd, my_hp, opp_hp, opp_atk=None):
        if role == 'attack':
            return 'blast' if cd == 0 and random.random() < bp else 'zap'
        return 'dodge' if random.random() < dp else 'brace'
    return s


def main():
    random.seed(42)
    N = 20000

    for dodge in [0.30, 0.35, 0.40]:
        sim = SparkDuelSim(dodge_chance=dodge)
        print(f"\n=== Dodge={dodge:.0%} ===")

        strats = {
            'Fun (80%B/70%D)': make_mixed(0.80, 0.70),
            'Fun (100%B/80%D)': make_mixed(1.00, 0.80),
            'Win (0%B/50%D)': make_mixed(0.00, 0.50),
            'Win (100%B/60%D)': make_mixed(1.00, 0.60),
            'Balanced': make_mixed(0.50, 0.50),
        }

        # Mirror (Fun vs Fun)
        fun = strats['Fun (80%B/70%D)']
        p1w = sum(1 for _ in range(N) if sim.play(fun, fun)[0] == 'p1')
        print(f"  Mirror P1 win: {p1w/N*100:.1f}%")

        # Fun vs Win
        win = strats['Win (0%B/50%D)']
        fvw = sum(1 for _ in range(N) if sim.play(fun, win)[0] == 'p1')
        wvf = sum(1 for _ in range(N) if sim.play(win, fun)[0] == 'p2')
        fun_wr = (fvw + wvf) / (2 * N) * 100
        print(f"  Fun avg WR vs Win(0%B/50%D): {fun_wr:.1f}%")

        # ToA-matched Fun vs Win
        fun2 = strats['Fun (100%B/80%D)']
        win2 = strats['Win (100%B/60%D)']
        fvw2 = sum(1 for _ in range(N) if sim.play(fun2, win2)[0] == 'p1')
        wvf2 = sum(1 for _ in range(N) if sim.play(win2, fun2)[0] == 'p2')
        fun_wr2 = (fvw2 + wvf2) / (2 * N) * 100
        print(f"  Fun(100%B/80%D) vs Win(100%B/60%D): {fun_wr2:.1f}%")

        # Blowout
        blowouts = sum(1 for _ in range(N) if max(sim.play(fun, fun)[1], sim.play(fun, fun)[2]) >= 3)
        print(f"  Blowout (3+ HP): {blowouts/N*100:.1f}%")

        # Avg rounds
        total_r = sum(sim.play(fun, fun)[3] for _ in range(N))
        print(f"  Avg rounds: {total_r/N:.2f}")

        # Dodge counter kills
        dodge_kills = 0
        for _ in range(N):
            w, h1, h2, r = sim.play(fun, fun)
            # approximate — if winner has low HP and counter exists
            if w == 'p1' and h1 == 1: dodge_kills += 1
            if w == 'p2' and h2 == 1: dodge_kills += 1
        print(f"  Clutch wins (1 HP): {dodge_kills/N*100:.1f}%")


if __name__ == '__main__':
    main()
