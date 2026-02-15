# Theory of Anticipation — Extended Research

[![Tests](https://github.com/Taru0208/anticipation-theory/actions/workflows/test.yml/badge.svg)](https://github.com/Taru0208/anticipation-theory/actions/workflows/test.yml)

> **Independent research fork** of [Theory of Anticipation (ToA)](https://github.com/akalouis/anticipation-theory) by Jeeheon (Lloyd) Oh.
> All original code and theory belong to the original author (MIT License).

## What is This?

Many games have a hidden problem: **the boring, safe strategy wins**, and **the exciting, risky strategy loses**. Players must choose between having fun and winning.

This research uses [Anticipation Theory](https://github.com/akalouis/anticipation-theory) — a mathematical framework for measuring "fun" — to detect and fix this problem.

**Key result:** By adjusting a game's numbers (damage, hit rates, costs), we can make the exciting play *also* the winning play, eliminating the trade-off entirely.

### Try It

**[▶ Play the Design Lab](https://taru0208.github.io/toa-research/design-lab/)** — same combat game, two sets of numbers. Feel the difference yourself.

**[How It Works](https://taru0208.github.io/toa-research/about/)** — plain-language explanation of the theory.

---

## Key Finding: Eliminating the Choice Paradox

The **Choice Paradox Gap (CPG)** measures how much "playing for fun" diverges from "playing to win."

| Metric | Standard Game | Optimized Game | Change |
|--------|-------------|----------------|--------|
| CPG | 0.346 | 0.000 | **100% eliminated** |
| Policy Impact | 0.147 | 0.366 | 2.5× more agency |
| GDS (fun-optimal) | 0.534 | 0.591 | +10.7% |
| Fun = Winning? | No | **Yes** | Paradox eliminated |

**Design principle:** make risky actions have higher expected value than safe actions. Verified across combat, resource, and economic game structures (276 tests).

---

## Quick Start (Python)

```bash
cd python
python3 -m pytest tests/ -v              # Run all 276 tests
python3 experiments/agency_model.py --generalize  # CPG analysis
```

---

## What's In This Fork

### Python Port
Complete rewrite of the C++ engine in Python, covering all 8 original game models plus 8 new ones.

### Research Phases

| Phase | Topic | Key Finding |
|-------|-------|-------------|
| **0** | Quality review | Fixed Rage Arena guard exploit, verified all models |
| **1** | Agency integration | Policy Impact (PI) as correct agency metric; CPG elimination via parameter optimization |
| **2** | Perceptual weighting | wGDS(α) — most games are DECAYING (A₁ dominant); ENL 3-5 sufficient |
| **3** | Composite Fun Score | CFS = wGDS × (1 + PI/GDS) × (1 - CPG); CPG is dominant factor (r=0.921) |

### All Game Models

<details>
<summary>16 game models implemented</summary>

| Model | File | GDS | Notes |
|-------|------|-----|-------|
| CoinToss | `coin_toss.py` | 0.500 | Theoretical A₁ maximum |
| RPS | `rps.py` | 0.471 | |
| HpGame | `hpgame.py` | 0.430 | Baseline combat |
| HpGame_Rage | `hpgame_rage.py` | 0.544 | +26.5% from rage mechanics |
| GoldGame | `goldgame.py` | — | Economic competition |
| GoldGame_Critical | `goldgame_critical.py` | — | With steal mechanics |
| LaneGame | `lanegame.py` | 0.540 | MOBA model, 65.9% depth |
| TwoTurnGame | `two_turn_game.py` | — | Parameter optimization |
| CoinDuel | `coin_duel.py` | 0.404 | Resource + coin flipping |
| DraftWars | `draft_wars.py` | 0.377 | Sequential draft, 62% depth |
| ChainReaction | `chain_reaction.py` | 0.252 | Territory control |
| FFABattle | `ffa_battle.py` | 0.429 | N-player FFA |
| AsymmetricCombat | `asymmetric_combat.py` | 34.49 | Ultra-high GDS (HP=20) |
| Education | `education_model.py` | — | Quiz engagement |
| Gambling | `gambling_model.py` | — | 6 casino games |
| Trading | `trading_model.py` | — | 6 financial instruments |

</details>

<details>
<summary>Extended experiments (full table)</summary>

| Experiment | Key Finding |
|------------|-------------|
| Genetic Algorithm | Symmetric game with GDS 0.979 (+77.8% over hand-designed) |
| GA + Accumulation | GDS 1.429 — highest-scoring game found |
| Optimal Game Analysis | Kill probability ≈ 50% maximizes engagement |
| Unbound Conjecture v2 | Linear GDS growth with depth across 6 game classes |
| Player Choice Paradox | Nash play reduces GDS by 5-7% |
| Perspective Desire Proof | Naive and perspective formulations are equivalent |
| Superlinear Growth | GDS grows as T^1.35; each A_k ~ T^(k-1) |
| Convergence Test | Unbound iff independent trials |
| Comeback Paradox | Artificial comebacks *decrease* GDS by 6% |
| Multiplayer Dynamics | FFA: GDS/P(win) increases with N; focus fire halves GDS |
| Education Model | Goldilocks zone P≈50-80%; longer quizzes less engaging |
| Gambling Mechanics | Gambling GDS 2-5× lower than games; payout asymmetry is the real killer |
| Trading/Investment | Day trading GDS 0.877 (2× HpGame); stop options +132% |

</details>

---

## Research Blog

Writeups and interactive demos: **[taru0208.github.io/toa-research](https://taru0208.github.io/toa-research/)**

## Credits

- **Original theory**: [Jeeheon (Lloyd) Oh](https://github.com/akalouis) — [Paper](https://doi.org/10.5281/zenodo.15826917) · [Article](https://medium.com/@aka.louis/can-you-mathematically-measure-fun-you-could-not-until-now-01168128d428)
- **Original demos**: [HpGame](https://akalouis.github.io/anticipation-theory/Html/hpgame.html) · [Optimized](https://akalouis.github.io/anticipation-theory/Html/hpgame_rage_optimized.html)
