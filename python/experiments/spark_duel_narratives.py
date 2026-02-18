"""Generate narrative game logs to evaluate "feel" of Spark Duel v5e.

Simulates games and prints them as readable stories to identify:
1. Does the game tell interesting stories?
2. Are there meaningful decision points?
3. What's frustrating vs exciting?
4. Is 2-3 rounds enough for a satisfying arc?
"""

import random

class SparkDuelSim:
    def __init__(self):
        self.max_hp = 7
        self.chip = 1
        self.blast_dmg = 4
        self.blast_hit = 0.60
        self.zap_dmg = 2
        self.brace_reduce = 1
        self.dodge_chance = 0.40
        self.dodge_counter = 1

    def resolve(self, attack, defend):
        if attack == 'blast':
            if random.random() >= self.blast_hit:
                return 0, 0, 'blast-miss'
            if defend == 'brace':
                return max(0, self.blast_dmg - self.brace_reduce), 0, 'blast-braced'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'blast-dodged'
                return self.blast_dmg, 0, 'blast-hit'
        else:
            if defend == 'brace':
                return max(0, self.zap_dmg - self.brace_reduce), 0, 'zap-braced'
            else:
                if random.random() < self.dodge_chance:
                    return 0, self.dodge_counter, 'zap-dodged'
                return self.zap_dmg, 0, 'zap-hit'

    def play_narrative(self, p1_strat, p2_strat, p1_name="Alice", p2_name="Bob"):
        hp1, hp2 = self.max_hp, self.max_hp
        cd1, cd2 = 0, 0
        lines = []
        lines.append(f"  {p1_name} ({hp1} HP) vs {p2_name} ({hp2} HP)")
        lines.append("")

        for rnd in range(1, 20):
            hp1 -= self.chip
            hp2 -= self.chip
            lines.append(f"  Round {rnd}: Chip damage → {p1_name} {hp1} HP, {p2_name} {hp2} HP")

            if hp1 <= 0 or hp2 <= 0:
                break

            # P1 attacks P2
            atk1 = p1_strat('attack', cd1, hp1, hp2)
            def2 = p2_strat('defend', cd2, hp2, hp1, atk1)
            d1, d2, ev1 = self.resolve(atk1, def2)
            hp2 -= d1
            hp1 -= d2
            cd1 = 1 if atk1 == 'blast' else max(0, cd1 - 1)

            atk_str = atk1.upper()
            def_str = def2.upper()
            if ev1 == 'blast-miss':
                lines.append(f"    {p1_name} fires BLAST → misses! {p2_name} breathes easy.")
            elif ev1 == 'blast-braced':
                lines.append(f"    {p1_name} fires BLAST → {p2_name} BRACES, takes {d1} dmg. [{p2_name}: {hp2} HP]")
            elif ev1 == 'blast-hit':
                lines.append(f"    {p1_name} fires BLAST → {p2_name} tries to DODGE... fails! {d1} dmg! [{p2_name}: {hp2} HP]")
            elif ev1 == 'blast-dodged':
                lines.append(f"    {p1_name} fires BLAST → {p2_name} DODGES! Counter {d2} dmg! [{p1_name}: {hp1} HP]")
            elif ev1 == 'zap-braced':
                lines.append(f"    {p1_name} ZAPs → {p2_name} BRACES, takes {d1} dmg. [{p2_name}: {hp2} HP]")
            elif ev1 == 'zap-hit':
                lines.append(f"    {p1_name} ZAPs → hits {p2_name} for {d1} dmg. [{p2_name}: {hp2} HP]")
            elif ev1 == 'zap-dodged':
                lines.append(f"    {p1_name} ZAPs → {p2_name} DODGES! Counter {d2}! [{p1_name}: {hp1} HP]")

            if hp1 <= 0 or hp2 <= 0:
                break

            # P2 attacks P1
            atk2 = p2_strat('attack', cd2, hp2, hp1)
            def1 = p1_strat('defend', cd1, hp1, hp2, atk2)
            d3, d4, ev2 = self.resolve(atk2, def1)
            hp1 -= d3
            hp2 -= d4
            cd2 = 1 if atk2 == 'blast' else max(0, cd2 - 1)

            if ev2 == 'blast-miss':
                lines.append(f"    {p2_name} fires BLAST → misses!")
            elif ev2 == 'blast-braced':
                lines.append(f"    {p2_name} fires BLAST → {p1_name} BRACES, takes {d3} dmg. [{p1_name}: {hp1} HP]")
            elif ev2 == 'blast-hit':
                lines.append(f"    {p2_name} fires BLAST → {p1_name} tries to DODGE... fails! {d3} dmg! [{p1_name}: {hp1} HP]")
            elif ev2 == 'blast-dodged':
                lines.append(f"    {p2_name} fires BLAST → {p1_name} DODGES! Counter {d4}! [{p2_name}: {hp2} HP]")
            elif ev2 == 'zap-braced':
                lines.append(f"    {p2_name} ZAPs → {p1_name} BRACES, takes {d3} dmg. [{p1_name}: {hp1} HP]")
            elif ev2 == 'zap-hit':
                lines.append(f"    {p2_name} ZAPs → hits {p1_name} for {d3} dmg. [{p1_name}: {hp1} HP]")
            elif ev2 == 'zap-dodged':
                lines.append(f"    {p2_name} ZAPs → {p1_name} DODGES! Counter {d4}! [{p2_name}: {hp2} HP]")

            if hp1 <= 0 or hp2 <= 0:
                break

            # End of round status
            diff = hp1 - hp2
            if abs(diff) >= 3:
                leader = p1_name if diff > 0 else p2_name
                lines.append(f"    --- {leader} dominates ({hp1} vs {hp2}) ---")
            elif abs(diff) <= 1:
                lines.append(f"    --- neck and neck ({hp1} vs {hp2}) ---")
            lines.append("")

        # Result
        lines.append("")
        if hp1 <= 0 and hp2 <= 0:
            lines.append(f"  ★ DRAW! Both fall. ★")
        elif hp2 <= 0:
            lines.append(f"  ★ {p1_name} WINS with {hp1} HP remaining! ★")
        elif hp1 <= 0:
            lines.append(f"  ★ {p2_name} WINS with {hp2} HP remaining! ★")

        return '\n'.join(lines), hp1, hp2


def make_fun(blast_pct=0.80, dodge_pct=0.70):
    def s(role, cd, my_hp, opp_hp, opp_atk=None):
        if role == 'attack':
            return 'blast' if cd == 0 and random.random() < blast_pct else 'zap'
        return 'dodge' if random.random() < dodge_pct else 'brace'
    return s


def classify_game(narrative, hp1, hp2):
    """Tag a game narrative with experience markers."""
    tags = []
    if 'DODGES! Counter' in narrative:
        tags.append('DODGE_COUNTER')
    if narrative.count('misses!') >= 2:
        tags.append('MISS_FRUSTRATION')
    if hp1 == 1 or hp2 == 1:
        tags.append('CLUTCH')
    if hp1 <= 0 and hp2 <= 0:
        tags.append('DRAW')
    if 'dominates' in narrative:
        tags.append('BLOWOUT')
    if 'neck and neck' in narrative:
        tags.append('CLOSE')
    return tags


def main():
    random.seed(42)
    sim = SparkDuelSim()
    fun = make_fun()

    print("=" * 70)
    print("SPARK DUEL v5e — NARRATIVE PLAYTEST")
    print("Config: HP=7, Blast=4@60%, Zap=2, Dodge=40%, Counter=1")
    print("Strategy: Fun (80% Blast, 70% Dodge)")
    print("=" * 70)

    # Generate 20 sample games
    tag_counts = {}
    for i in range(20):
        print(f"\n{'─'*60}")
        print(f"GAME {i+1}")
        print(f"{'─'*60}")
        narr, hp1, hp2 = sim.play_narrative(fun, fun)
        print(narr)
        tags = classify_game(narr, hp1, hp2)
        if tags:
            print(f"  Tags: {', '.join(tags)}")
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIENCE SUMMARY (20 games)")
    print(f"{'='*60}")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:<20} {count:>3} ({count/20*100:.0f}%)")

    # Critical question analysis
    print(f"\n{'='*60}")
    print("CRITICAL QUESTIONS")
    print(f"{'='*60}")
    print("""
1. STORY ARC: Do games have a beginning, middle, and end?
   → With 2-3 rounds, the arc is: chip damage sets stage →
     attack exchanges create drama → someone falls.

2. MEANINGFUL CHOICE: Did strategy choices matter in these narratives?
   → Attack choice (Blast vs Zap) matters when cooldown forces Zap.
   → Defense choice (Brace vs Dodge) creates the "gamble" moment.
   → But: both players use the same strategy, so it's mostly dice.

3. FRUSTRATION POINTS:
   → Double Blast miss = wasted turns (40% × 40% = 16% per pair)
   → Getting Blasted for 4 dmg through failed Dodge = punishing
   → Chip damage killing you (no agency) can feel anticlimactic

4. EXCITEMENT POINTS:
   → Dodge + Counter = best moment (risk rewarded!)
   → Clutch 1-HP wins = dramatic
   → Blast hitting through Brace for 3 dmg = impactful

5. DIFFERENTIATION GAP:
   → These games could be described as "luck-heavy RPS combat"
   → The attack/defense split is interesting but not unique
   → Missing: progression within a match, meta-strategy, surprise
""")


if __name__ == '__main__':
    main()
