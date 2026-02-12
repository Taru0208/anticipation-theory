# Theory of Anticipation — Extended Research

> **This is an independent research fork** of the [Theory of Anticipation (ToA)](https://github.com/akalouis/anticipation-theory) by Jeeheon (Lloyd) Oh ([Farseer Co., Ltd.](https://github.com/akalouis)).
>
> **Original work:**
> - Repository: [akalouis/anticipation-theory](https://github.com/akalouis/anticipation-theory) (MIT License)
> - Paper: [Formulation of Engagement as Local Anticipation](https://doi.org/10.5281/zenodo.15826917)
> - Article: [Can You Mathematically Measure Fun?](https://medium.com/@aka.louis/can-you-mathematically-measure-fun-you-could-not-until-now-01168128d428)
>
> This repository extends the original with a Python port, additional experiments, and exploration of the theory's open conjectures. All original code and theory belong to the original author.

-----

## What is the Theory of Anticipation?

**Theory of Anticipation (ToA)** is a mathematical framework for quantifying engagement, or "fun." Created primarily as a game design tool, it offers an objective and computable measure of a player's engagement.

-----

## Overview

The Theory of Anticipation provides a formal language to describe and optimize for "fun." It aims to move aspects of game design from pure intuition to a domain that can be analytically explored and computationally optimized. By modeling a game as a system of states and probabilities, it calculates the engagement a player feels moment-to-moment.

### Inputs

1.  **Game Model**: The game is represented as a probabilistic graph:

      * **States ($s$)**: All possible, discrete game situations.
      * **Transitions ($P(s \\to s')$**: The probability of moving from one state to another.

2.  **Intrinsic Desire ($D\_{local}$)**: The root "seed" desire from which all derived anticipation values are generated. To enable objective analysis, you simply define the ultimate goal (e.g., set the "win condition" state to `1` and all other states to `0`). All other desire and anticipation values for the entire game are then computed automatically from this input.

### Outputs

  * **Global Desire ($D\_{global}$)**: The computed desirability of a given state.
  * **Anticipation ($A\_1$)**: The immediate, quantifiable "fun" of any given state.
  * **Nested Anticipation ($A\_2, A\_3, ...$)**: Higher-order components of anticipation for any given state, capturing deeper strategic fun.
  * **Game Design Score**: A single, objective score for the entire game design.

### Core Foundation

At its heart, the theory measures the uncertainty of meaningfulness of future branches. The Local Anticipation ($A\_1$) for a given state ($s$) is calculated as:

$$A_1(s) = \sqrt{\sum_{s'} P(s \to s') \cdot (D(s') - \mu)^2}$$

This formula is the core of the theory, from which all other metrics are derived through algorithmic expansion. Detailed implementation specifics, such as the concept of "Perspective Desire" or efficient dynamic programming practices, are fully documented within the source code.

-----

## Empirical Proof: Playable Demos 🎮

Below are two browser-playable versions of a simple game. The second version has been optimized by a computer algorithm, guided only by ToA's formulas, to maximize its "Game Design Score."

  * **Play the Baseline Game: [HpGame](https://akalouis.github.io/anticipation-theory/Html/hpgame.html)**
  * **Play the Optimized Game: [HpGame\_Rage\_Optimized](https://akalouis.github.io/anticipation-theory/Html/hpgame_rage_optimized.html)**

You will likely find the second game more engaging. This is a powerful and intuitive empirical proof that the Theory of Anticipation works.

-----

## Included Experiments

The following experiments are implemented as demonstrations and test cases for the theory.

  * `rock_paper_scissors`
  * `cointoss`
  * `hpgame` (The baseline playable demo)
  * `hpgame_interactive`
  * `hpgame_rage`
  * `hpgame_rage_optimize_critchance`
  * `hpgame_rage_compare_mechanics`
  * `hpgame_rage_optimized` (The optimized playable demo)

**Note**: Please refer to the source code for the most current and complete list of experiments.

-----

## Research Blog

Detailed writeups of each experiment and discovery: **[taru0208.github.io/toa-research](https://taru0208.github.io/toa-research/)**

-----

## Python Port & Extended Research

This fork includes a complete Python port of the C++ analysis engine, enabling rapid experimentation on any platform.

### Getting Started (Python)

```bash
cd python
python3 -m pytest tests/ -v      # Run all 35 tests
python3 experiments/moba_model.py # Run MOBA analysis
```

### Ported Components

| Component | File | Description |
|-----------|------|-------------|
| Core Engine | `python/toa/engine.py` | Full `analyze()` function — DP, nested anticipation, GDS |
| Monte Carlo | `python/toa/simulate.py` | Simulation-based GDS estimation |
| CoinToss | `python/toa/games/coin_toss.py` | A₁ = 0.500 (theoretical maximum) |
| RPS | `python/toa/games/rps.py` | A₁ = 0.471 |
| HpGame | `python/toa/games/hpgame.py` | GDS = 0.430 |
| HpGame\_Rage | `python/toa/games/hpgame_rage.py` | GDS = 0.544 (+26.5%) |
| GoldGame | `python/toa/games/goldgame.py` | Economic competition |
| GoldGame\_Critical | `python/toa/games/goldgame_critical.py` | With steal mechanics |
| LaneGame | `python/toa/games/lanegame.py` | MOBA laning model |
| TwoTurnGame | `python/toa/games/two_turn_game.py` | Parameter optimization |

### Extended Experiments

| Experiment | Key Finding |
|------------|-------------|
| **Genetic Algorithm Optimizer** | Discovered a symmetric game with GDS = 0.979 (77.8% better than hand-designed HpGame\_Rage) |
| **GA + Accumulation Mechanics** | GDS = 1.429 — resource accumulation creates the highest-scoring game found |
| **Optimal Game Analysis** | Optimal kill probability ≈ 50%; "moderate lethality" maximizes engagement |
| **Unbound Conjecture v2** | 6 game classes all show linear GDS growth with depth; Best-of-N exceeds GDS 1.0 at depth 17 |
| **Player Choice Paradox** | Nash equilibrium play reduces GDS by 5–7% — optimal strategy is less fun |
| **MOBA Lane Model** | GDS = 0.540 with highest strategic depth ratio (65.9%) of all tested games |
| **Perspective Desire Proof** | Mathematically proved that naive and perspective formulations are equivalent |
| **Narrative Structure** | Rage mechanics amplify narrative arc variance 10× |
| **GameFlow Bridge** | Maps GameFlow (2005) qualitative elements to computable ToA metrics |

-----

## Examples

The framework allows you to define a complete game model with minimal code. The engine will then compute the anticipation and evaluate your design. Below is the full example required to implement and analyze a new experiment.

```cpp
module;
export module hpgame_v1;
import "pch.h";
import game;

export namespace hpgame
{
	struct State
	{
		unsigned char hp1; // player1 hp
		unsigned char hp2; // player2 hp
		auto operator <=> (const State&) const = default;
	};
	struct Transition { float probability; State to; };

	struct Game
	{
		typedef game::EmptyConfig Config;
		typedef State State;

		static State initial_state() { return State{ 5, 5 }; }
		static bool is_terminal_state(const State& s) { return s.hp1 <= 0 || s.hp2 <= 0; }
		static std::vector<Transition> get_transitions(Config, const State& state)
		{
			std::vector<Transition> result;
			if (is_terminal_state(state)) return result;

			// win, draw, loss version
			result.push_back({ 1.0f / 3.0f, State{ state.hp1, (unsigned char)(state.hp2 - 1) }, });
			result.push_back({ 1.0f / 3.0f, State{ (unsigned char)(state.hp1 - 1), (unsigned char)(state.hp2 - 1) }, });
			result.push_back({ 1.0f / 3.0f, State{ (unsigned char)(state.hp1 - 1), state.hp2 }, });

			game::sanitize_transitions(result);
			return result;
		}
		static float compute_intrinsic_desire(const State& state)
		{
			// Player 1 wins if their HP is > 0 and Player 2's HP is 0
			return state.hp1 != 0 && state.hp2 == 0 ? 1.0f : 0.0f;
		}
		static std::string tostr(const State& s) { return "HP1:" + std::to_string(s.hp1) + " HP2:" + std::to_string(s.hp2); }
	};
}
```
