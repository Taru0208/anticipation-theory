export module goldgame_clean;
import <vector>;
import <string>;
import <cmath>;
const unsigned char MAX_TURNS = 10;
export namespace goldgame_clean
{
	struct Config
	{
		// note: start_gold:1000 success:*1.2 fail:/1.2 chance:0.68 -> GDS 0.370465

		float success_chance = 0.68f;
		unsigned int base_reward = 100;
		float geometric_multiplier = 1.2f;    // +20% on success
		float geometric_penalty = 1.0f / 1.2f; // /1.2 on failure
	};
	struct State
	{
		unsigned int player1_gold;
		unsigned int player2_gold;
		unsigned char turn;
		auto operator <=> (const State&) const = default;
	};
	struct Transition
	{
		float probability;
		State to;
	};
	struct Game
	{
		using Config = goldgame_clean::Config;
		using State = goldgame_clean::State;
		static State initial_state()
		{
			return State{ 1000, 1000, 0, };
		}
		static bool is_terminal_state(const State& s)
		{
			return MAX_TURNS <= s.turn;
		}
		static std::vector<Transition> get_transitions(const Config& config, const State& state)
		{
			if (is_terminal_state(state)) return {};
			std::vector<Transition> result;
			// Regular gameplay: simple hit/miss
			const float hit = config.success_chance;
			const float miss = 1.0f - hit;

			// 4 combinations: hit/miss for each player
			for (int p1_result = 0; p1_result <= 1; p1_result++) {
				for (int p2_result = 0; p2_result <= 1; p2_result++) {
					State next_state = state;
					next_state.turn++;

					// Geometric rewards: +base then multiply on hit, multiply penalty on miss
					if (p1_result == 1) {
						next_state.player1_gold *= config.geometric_multiplier;
					}
					else {
						next_state.player1_gold *= config.geometric_penalty;
					}
					if (p2_result == 1) {
						next_state.player2_gold *= config.geometric_multiplier;
					}
					else {
						next_state.player2_gold *= config.geometric_penalty;
					}

					float prob = (p1_result ? hit : miss) * (p2_result ? hit : miss);
					result.push_back({ prob, next_state });
				}
			}
			return result;
		}
		static float compute_intrinsic_desire(const State& state)
		{
			if (!is_terminal_state(state)) return 0.0f;
			return (state.player1_gold > state.player2_gold) ? 1.0f : 0.0f;
		}
		static std::string tostr(const State& s)
		{
			std::string result = "T" + std::to_string(s.turn) +
				" P1:" + std::to_string(s.player1_gold) +
				" P2:" + std::to_string(s.player2_gold);

			return result;
		}
	};
}