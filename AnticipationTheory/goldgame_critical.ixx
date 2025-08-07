export module goldgame_critical;
import <vector>;
import <string>;
import <cmath>;
import <algorithm>;

const unsigned char MAX_TURNS = 5;

export namespace goldgame_critical
{
	enum class EventRewardType : unsigned char
	{
		Linear,
		Geometric
	};

	struct Config
	{
		EventRewardType reward = EventRewardType::Linear;
		float success_chance = 0.5f;
		unsigned int base_reward = 100;
		float reward_growth = 0.0f;
		float critical_chance = 0.00f;
		float steal_percentage = 0.0f;

		// Geometric reward system
		float geometric_multiplier = 1.2f;    // +20% on success
		float geometric_penalty = 1.0f / 1.2f; // /1.2 on failure
	};

	struct State
	{
		unsigned int player1_gold;
		unsigned int player2_gold;
		unsigned char turn;
		bool has_won = false;  // P1 won the game
		auto operator <=> (const State&) const = default;
	};

	struct Transition
	{
		float probability;
		State to;
	};

	struct Game
	{
		using Config = goldgame_critical::Config;
		using State = goldgame_critical::State;

		static State initial_state()
		{
			return State{ 1000, 1000, 1, false };
		}

		static bool is_terminal_state(const State& s)
		{
			return s.turn > MAX_TURNS;
		}

		static std::vector<Transition> get_transitions(const Config& config, const State& state)
		{
			if (is_terminal_state(state)) return {};

			std::vector<Transition> result;

			// If this is the last turn, create win/lose transitions based on gold ratio
			if (state.turn == MAX_TURNS) {
				unsigned int total_gold = state.player1_gold + state.player2_gold;
				if (total_gold == 0) {
					State both_lose = state;
					both_lose.turn++;
					both_lose.has_won = false;
					result.push_back({ 1.0f, both_lose });
					return result;
				}

				// Extreme ratio: cube the advantage
				float p1_ratio = static_cast<float>(state.player1_gold) / static_cast<float>(total_gold);
				float p1_cubed = p1_ratio * p1_ratio * p1_ratio;
				float p2_cubed = (1.0f - p1_ratio) * (1.0f - p1_ratio) * (1.0f - p1_ratio);

				float p1_win_prob = p1_cubed / (p1_cubed + p2_cubed);
				float p2_win_prob = 1.0f - p1_win_prob;

				State p1_wins = state;
				p1_wins.turn++;
				p1_wins.has_won = true;

				State p2_wins = state;
				p2_wins.turn++;
				p2_wins.has_won = false;

				result.push_back({ p1_win_prob, p1_wins });
				result.push_back({ p2_win_prob, p2_wins });

				return result;
			}

			// Regular gameplay: gold events
			const float miss = 1.0f - config.success_chance;
			const float normal_hit = config.success_chance * (1.0f - config.critical_chance);
			const float critical_hit = config.success_chance * config.critical_chance;

			unsigned int base_reward = config.base_reward + static_cast<unsigned int>(config.reward_growth * state.turn * config.base_reward);

			std::vector<float> probs = { miss, normal_hit, critical_hit };

			for (int p1_result = 0; p1_result <= 2; p1_result++) {
				for (int p2_result = 0; p2_result <= 2; p2_result++) {
					State next_state = state;
					next_state.turn++;

					// Apply rewards based on reward type
					if (config.reward == EventRewardType::Linear) {
						// Linear: fixed gold amounts
						if (p1_result == 1) {
							next_state.player1_gold += base_reward;
						}
						else if (p1_result == 2) {
							next_state.player1_gold += base_reward;
							unsigned int stolen = static_cast<unsigned int>(next_state.player2_gold * config.steal_percentage);
							next_state.player1_gold += stolen;
							next_state.player2_gold -= stolen;
						}

						if (p2_result == 1) {
							next_state.player2_gold += base_reward;
						}
						else if (p2_result == 2) {
							next_state.player2_gold += base_reward;
							unsigned int stolen = static_cast<unsigned int>(next_state.player1_gold * config.steal_percentage);
							next_state.player2_gold += stolen;
							next_state.player1_gold -= stolen;
						}
					}
					else if (config.reward == EventRewardType::Geometric) {
						// Geometric: base + percentage, or penalty
						if (p1_result == 0) {
							// Miss: lose percentage of current gold
							next_state.player1_gold = static_cast<unsigned int>(next_state.player1_gold * config.geometric_penalty);
						}
						else if (p1_result == 1) {
							// Normal hit: base + percentage
							unsigned int bonus = static_cast<unsigned int>(next_state.player1_gold * (config.geometric_multiplier - 1.0f));
							next_state.player1_gold += base_reward + bonus;
						}
						else if (p1_result == 2) {
							// Critical: base + percentage + steal
							unsigned int bonus = static_cast<unsigned int>(next_state.player1_gold * (config.geometric_multiplier - 1.0f));
							next_state.player1_gold += base_reward + bonus;
							unsigned int stolen = static_cast<unsigned int>(next_state.player2_gold * config.steal_percentage);
							next_state.player1_gold += stolen;
							next_state.player2_gold -= stolen;
						}

						if (p2_result == 0) {
							// Miss: lose percentage of current gold
							next_state.player2_gold = static_cast<unsigned int>(next_state.player2_gold * config.geometric_penalty);
						}
						else if (p2_result == 1) {
							// Normal hit: base + percentage
							unsigned int bonus = static_cast<unsigned int>(next_state.player2_gold * (config.geometric_multiplier - 1.0f));
							next_state.player2_gold += base_reward + bonus;
						}
						else if (p2_result == 2) {
							// Critical: base + percentage + steal
							unsigned int bonus = static_cast<unsigned int>(next_state.player2_gold * (config.geometric_multiplier - 1.0f));
							next_state.player2_gold += base_reward + bonus;
							unsigned int stolen = static_cast<unsigned int>(next_state.player1_gold * config.steal_percentage);
							next_state.player2_gold += stolen;
							next_state.player1_gold -= stolen;
						}
					}

					result.push_back({ probs[p1_result] * probs[p2_result], next_state });
				}
			}

			return result;
		}

		static float compute_intrinsic_desire(const State& state)
		{
			if (!is_terminal_state(state)) return 0.0f;
			return state.has_won ? 1.0f : 0.0f;
		}

		static std::string tostr(const State& s)
		{
			std::string result = "T" + std::to_string(s.turn) +
				" P1:" + std::to_string(s.player1_gold) +
				" P2:" + std::to_string(s.player2_gold);

			if (is_terminal_state(s)) {
				result += s.has_won ? " [P1 WINS]" : " [P2 WINS]";
			}

			return result;
		}
	};
}