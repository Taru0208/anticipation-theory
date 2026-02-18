"""Spark Duel iteration #2 — deeper gameplay feel analysis.

Analyzes round-by-round patterns, comeback frequency, and
first-player advantage mitigation strategies.

Config: HP=7, Blast=4@60%, Zap=2, Brace=-1, Dodge=30%+counter=1
"""

import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
        """Returns (dmg_to_defender, dmg_to_attacker, event_name)."""
        if attack == 'blast':
            if random.random() >= self.blast_hit:
                return 0, 0, 'blast-miss'
            if defend == 'brace':
                return max(0, self.blast_dmg - self.brace_reduce), 0, 'blast-brace'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'blast-dodged'
                return self.blast_dmg, 0, 'blast-dodge-fail'
        else:  # zap
            if defend == 'brace':
                dmg = max(0, self.zap_dmg - self.brace_reduce)
                return dmg, 0, 'zap-brace'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'zap-dodged'
                return self.zap_dmg, 0, 'zap-hit'

    def play_game(self, p1_strategy, p2_strategy, record_detail=False):
        """Play one game. Returns detailed game record if requested."""
        hp1 = self.max_hp
        hp2 = self.max_hp
        cd1 = 0
        cd2 = 0
        round_records = []

        for rnd in range(20):
            # Chip damage
            hp1 -= self.chip
            hp2 -= self.chip
            if hp1 <= 0 or hp2 <= 0:
                round_records.append({
                    'round': rnd + 1, 'phase': 'chip-kill',
                    'hp1': hp1, 'hp2': hp2
                })
                break

            round_start_hp1 = hp1
            round_start_hp2 = hp2

            # P1 attacks P2
            atk1 = p1_strategy('attack', cd1, hp1, hp2)
            def2 = p2_strategy('defend', cd2, hp2, hp1, atk1)
            dmg_def, dmg_atk, event1 = self.resolve(atk1, def2)
            hp2 -= dmg_def
            hp1 -= dmg_atk
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)

            if hp1 <= 0 or hp2 <= 0:
                round_records.append({
                    'round': rnd + 1, 'phase': 'p1-atk',
                    'hp1_start': round_start_hp1, 'hp2_start': round_start_hp2,
                    'hp1': hp1, 'hp2': hp2,
                    'p1_atk': atk1, 'p2_def': def2, 'event1': event1,
                })
                break

            # P2 attacks P1
            atk2 = p2_strategy('attack', cd2, hp2, hp1)
            def1 = p1_strategy('defend', cd1, hp1, hp2, atk2)
            dmg_def2, dmg_atk2, event2 = self.resolve(atk2, def1)
            hp1 -= dmg_def2
            hp2 -= dmg_atk2
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)

            round_records.append({
                'round': rnd + 1,
                'hp1_start': round_start_hp1, 'hp2_start': round_start_hp2,
                'hp1': hp1, 'hp2': hp2,
                'p1_atk': atk1, 'p2_def': def2, 'event1': event1,
                'p2_atk': atk2, 'p1_def': def1, 'event2': event2,
            })

            if hp1 <= 0 or hp2 <= 0:
                break

        winner = 'draw' if hp1 <= 0 and hp2 <= 0 else ('p1' if hp2 <= 0 else 'p2' if hp1 <= 0 else 'timeout')
        return {
            'winner': winner,
            'rounds': len(round_records),
            'records': round_records,
            'final_hp1': hp1,
            'final_hp2': hp2,
        }


def make_mixed(blast_pct, dodge_pct):
    def strategy(role, cd, my_hp, opp_hp, opp_atk=None):
        if role == 'attack':
            if cd > 0:
                return 'zap'
            return 'blast' if random.random() < blast_pct else 'zap'
        else:
            return 'dodge' if random.random() < dodge_pct else 'brace'
    return strategy


def analyze_games(sim, p1, p2, n=20000):
    """Run n games and return deep analysis."""
    results = {'p1': 0, 'p2': 0, 'draw': 0, 'timeout': 0}
    round_counts = Counter()
    event_counts = Counter()
    total_events = 0
    comeback_count = 0  # games where trailing player won
    blowout_count = 0  # games ending with 3+ HP difference
    hp_deltas_per_round = {}  # round -> list of (p1_delta, p2_delta)
    lead_changes = 0
    dramatic_games = []  # games with 3+ lead changes

    for _ in range(n):
        g = sim.play_game(p1, p2)
        results[g['winner']] += 1
        round_counts[g['rounds']] += 1

        for rec in g['records']:
            if 'event1' in rec:
                event_counts[rec['event1']] += 1
                total_events += 1
            if 'event2' in rec:
                event_counts[rec['event2']] += 1
                total_events += 1

            rnd = rec['round']
            if rnd not in hp_deltas_per_round:
                hp_deltas_per_round[rnd] = []
            if 'hp1_start' in rec:
                hp_deltas_per_round[rnd].append((
                    rec['hp1'] - rec['hp1_start'],
                    rec['hp2'] - rec['hp2_start']
                ))

        # Comeback analysis
        if len(g['records']) >= 2:
            changes = 0
            prev_leader = None
            was_trailing = False
            for rec in g['records']:
                hp1 = rec.get('hp1', sim.max_hp)
                hp2 = rec.get('hp2', sim.max_hp)
                leader = 'p1' if hp1 > hp2 else ('p2' if hp2 > hp1 else 'tied')
                if prev_leader and leader != 'tied' and prev_leader != 'tied' and leader != prev_leader:
                    changes += 1
                if leader != 'tied':
                    prev_leader = leader
                if g['winner'] == 'p1' and hp2 > hp1:
                    was_trailing = True
                if g['winner'] == 'p2' and hp1 > hp2:
                    was_trailing = True

            lead_changes += changes
            if was_trailing:
                comeback_count += 1
            if changes >= 3:
                dramatic_games.append(g)

        # Blowout check
        final_hp = max(g['final_hp1'], g['final_hp2'])
        if final_hp >= 3:
            blowout_count += 1

    return {
        'results': results,
        'round_dist': round_counts,
        'events': {k: v/total_events for k, v in sorted(event_counts.items(), key=lambda x: -x[1])},
        'comebacks': comeback_count / n,
        'blowouts': blowout_count / n,
        'avg_lead_changes': lead_changes / n,
        'dramatic_pct': len(dramatic_games) / n,
        'hp_deltas': hp_deltas_per_round,
        'n': n,
    }


def main():
    sim = SparkDuelSim()  # HP=7, Blast=4@60%, Zap=2, Dodge=30%+counter=1
    N = 20000
    random.seed(42)

    strategies = {
        'Fun (80%B/70%D)': make_mixed(0.80, 0.70),
        'Win (0%B/50%D)': make_mixed(0.00, 0.50),
        'Balanced (50%B/50%D)': make_mixed(0.50, 0.50),
        'All-Blast (100%B/0%D)': make_mixed(1.00, 0.00),
        'All-Dodge (50%B/100%D)': make_mixed(0.50, 1.00),
    }

    print("=" * 80)
    print("Spark Duel v4 — Iteration #2 Deep Analysis")
    print(f"Config: HP={sim.max_hp}, Blast={sim.blast_dmg}@{sim.blast_hit:.0%}, "
          f"Zap={sim.zap_dmg}, Dodge={sim.dodge_chance:.0%}+counter={sim.dodge_counter}")
    print(f"Games: {N}")
    print("=" * 80)

    # 1. Fun vs Fun — baseline feel
    print("\n=== 1. FUN vs FUN (80%B/70%D vs itself) ===")
    a = analyze_games(sim, strategies['Fun (80%B/70%D)'], strategies['Fun (80%B/70%D)'], N)
    print(f"  P1 win: {a['results']['p1']/N*100:.1f}%  P2 win: {a['results']['p2']/N*100:.1f}%  Draw: {a['results']['draw']/N*100:.1f}%")
    print(f"  Comebacks: {a['comebacks']*100:.1f}%  Blowouts (3+ HP): {a['blowouts']*100:.1f}%")
    print(f"  Avg lead changes: {a['avg_lead_changes']:.2f}  Dramatic games (3+ changes): {a['dramatic_pct']*100:.1f}%")
    print(f"\n  Round distribution:")
    for rnd in sorted(a['round_dist']):
        pct = a['round_dist'][rnd] / N * 100
        bar = '#' * int(pct)
        print(f"    Round {rnd}: {pct:>5.1f}% {bar}")
    print(f"\n  Event breakdown:")
    for event, pct in a['events'].items():
        print(f"    {event:<20} {pct*100:>5.1f}%")

    # 2. Fun vs Win
    print("\n=== 2. FUN vs WIN ===")
    b = analyze_games(sim, strategies['Fun (80%B/70%D)'], strategies['Win (0%B/50%D)'], N)
    print(f"  Fun(P1) wins: {b['results']['p1']/N*100:.1f}%  Win(P2) wins: {b['results']['p2']/N*100:.1f}%")
    print(f"  Comebacks: {b['comebacks']*100:.1f}%")
    b2 = analyze_games(sim, strategies['Win (0%B/50%D)'], strategies['Fun (80%B/70%D)'], N)
    print(f"  Win(P1) wins: {b2['results']['p1']/N*100:.1f}%  Fun(P2) wins: {b2['results']['p2']/N*100:.1f}%")
    avg_fun_wr = (b['results']['p1']/N + b2['results']['p2']/N) / 2 * 100
    print(f"  → Fun avg winrate vs Win: {avg_fun_wr:.1f}%")

    # 3. Cross-matchup matrix
    print("\n=== 3. CROSS-MATCHUP MATRIX (P1 win%) ===")
    snames = list(strategies.keys())
    short_names = ['Fun', 'Win', 'Bal', 'AllB', 'AllD']
    p1p2 = 'P1\\P2'
    header = f"{p1p2:<25}"
    for s in short_names:
        header += f" {s:>8}"
    print(header)
    for i, n1 in enumerate(snames):
        row = f"{short_names[i]:<25}"
        for j, n2 in enumerate(snames):
            r = analyze_games(sim, strategies[n1], strategies[n2], 5000)
            row += f" {r['results']['p1']/5000*100:>7.1f}%"
        print(row)

    # 4. First-player advantage deep dive
    print("\n=== 4. FIRST-PLAYER ADVANTAGE (D₀ = 0.553) ===")
    print("  Testing: alternate first-attacker across best-of-3 matches")
    fun = make_mixed(0.80, 0.70)
    total_matches = 10000
    p1_match_wins = 0
    for _ in range(total_matches):
        p1_games = 0
        p2_games = 0
        for game_num in range(5):  # max 5 games in best-of-3
            if game_num % 2 == 0:
                g = sim.play_game(fun, fun)
                if g['winner'] == 'p1':
                    p1_games += 1
                elif g['winner'] == 'p2':
                    p2_games += 1
            else:
                # Swap: P2 gets first attack
                g = sim.play_game(fun, fun)
                if g['winner'] == 'p1':
                    p2_games += 1
                elif g['winner'] == 'p2':
                    p1_games += 1
            if p1_games >= 2 or p2_games >= 2:
                break
        if p1_games >= 2:
            p1_match_wins += 1
    print(f"  Best-of-3 with alternating: P1 wins {p1_match_wins/total_matches*100:.1f}% of matches")

    # 5. HP trajectory analysis
    print("\n=== 5. AVERAGE HP TRAJECTORY (Fun vs Fun) ===")
    # Re-run with full tracking
    hp_trajectories = {'p1': {}, 'p2': {}}
    for _ in range(N):
        g = sim.play_game(fun, fun)
        for rec in g['records']:
            rnd = rec['round']
            if rnd not in hp_trajectories['p1']:
                hp_trajectories['p1'][rnd] = []
                hp_trajectories['p2'][rnd] = []
            hp_trajectories['p1'][rnd].append(rec['hp1'])
            hp_trajectories['p2'][rnd].append(rec['hp2'])

    print(f"  {'Round':<8} {'P1 avg HP':>10} {'P2 avg HP':>10} {'HP diff':>10}")
    for rnd in sorted(hp_trajectories['p1']):
        avg1 = sum(hp_trajectories['p1'][rnd]) / len(hp_trajectories['p1'][rnd])
        avg2 = sum(hp_trajectories['p2'][rnd]) / len(hp_trajectories['p2'][rnd])
        print(f"  {rnd:<8} {avg1:>10.2f} {avg2:>10.2f} {avg1-avg2:>+10.2f}")

    # 6. Blast miss frustration
    print("\n=== 6. BLAST MISS PATTERNS ===")
    consecutive_misses = Counter()
    max_miss_streak = 0
    total_blast_count = 0
    for _ in range(N):
        g = sim.play_game(fun, fun)
        streak = 0
        for rec in g['records']:
            for key in ['event1', 'event2']:
                if key not in rec:
                    continue
                ev = rec[key]
                if 'blast' in ev:
                    total_blast_count += 1
                if ev == 'blast-miss':
                    streak += 1
                    max_miss_streak = max(max_miss_streak, streak)
                elif 'blast' in ev:
                    if streak > 0:
                        consecutive_misses[streak] += 1
                    streak = 0
        if streak > 0:
            consecutive_misses[streak] += 1

    print(f"  Total blasts: {total_blast_count}")
    print(f"  Miss streaks:")
    for streak in sorted(consecutive_misses):
        print(f"    {streak} consecutive misses: {consecutive_misses[streak]} times ({consecutive_misses[streak]/N*100:.1f}% of games)")
    print(f"  Longest miss streak: {max_miss_streak}")

    # 7. Dodge counter dramatic moments
    print("\n=== 7. DRAMATIC MOMENTS ===")
    dodge_kills = 0  # counter-attack kills
    clutch_wins = 0  # won at 1 HP
    for _ in range(N):
        g = sim.play_game(fun, fun)
        for rec in g['records']:
            for key in ['event1', 'event2']:
                if key not in rec:
                    continue
                if 'dodged' in rec[key]:
                    # Check if counter-damage killed
                    if rec['hp1'] <= 0 or rec['hp2'] <= 0:
                        dodge_kills += 1
            if g['winner'] == 'p1' and g['final_hp1'] == 1:
                clutch_wins += 1
            elif g['winner'] == 'p2' and g['final_hp2'] == 1:
                clutch_wins += 1

    print(f"  Dodge counter-kills: {dodge_kills} ({dodge_kills/N*100:.1f}% of games)")
    print(f"  Clutch wins (1 HP): {clutch_wins} ({clutch_wins/N*100:.1f}% of games)")


if __name__ == '__main__':
    main()
