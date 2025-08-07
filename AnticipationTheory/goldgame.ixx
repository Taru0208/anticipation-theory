export module goldgame;
import <vector>;
import <string>;
import <cmath>;

const unsigned char MAX_TURNS = 20; // Maximum number of turns in the game

export namespace goldgame
{
	enum class EventRewardType : unsigned char
	{
		Linear,
		Geometric
	};

	enum class WinValueType : unsigned char
	{
		Constant,
		Linear,
		Exponential
	};

	struct Config
	{
		EventRewardType reward = EventRewardType::Linear;
		WinValueType win_value = WinValueType::Constant;

		// Linear reward system
		unsigned int linear_base_reward = 100;     // Base reward amount

		// Geometric reward system  
		float geometric_multiplier = 1.2f;
		float geometric_penalty_mult = 1.0f / 1.2f;

		Config() = default;
		Config(EventRewardType reward_type) : reward(reward_type) {}

		bool is_valid() const {
			return (reward == EventRewardType::Linear || reward == EventRewardType::Geometric) &&
				(linear_base_reward > 0 || geometric_multiplier > 1.0f);
		}
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
		using Config = goldgame::Config;
		using State = goldgame::State;

		static State initial_state()
		{
			return State{ 1000, 1000, 0 };
		}

		static bool is_terminal_state(const State& s)
		{
			return s.turn >= MAX_TURNS; // Keep original interface - actual logic uses config in get_transitions
		}

		static std::vector<Transition> get_transitions(const Config& config, const State& state)
		{
			std::vector<Transition> result;
			if (is_terminal_state(state)) return result;

			const unsigned char next_turn = state.turn + 1;

			auto add_outcome = [&](float prob, unsigned int p1_gold, unsigned int p2_gold) {
				result.push_back({ prob, {p1_gold, p2_gold, next_turn} });
				};

			switch (config.reward)
			{
			case EventRewardType::Linear:
			{
				const float success = 0.5f;
				const float failure = 1.0f - success;
				const unsigned int reward = config.linear_base_reward;

				add_outcome(success * success, state.player1_gold + reward, state.player2_gold + reward);
				add_outcome(success * failure, state.player1_gold + reward, state.player2_gold);
				add_outcome(failure * success, state.player1_gold, state.player2_gold + reward);
				add_outcome(failure * failure, state.player1_gold, state.player2_gold);
				break;
			}
			case EventRewardType::Geometric:
			{
				const float success = 0.65f;
				const float failure = 1.0f - success;
				const float multiplier = config.geometric_multiplier;
				const float penalty = config.geometric_penalty_mult;

				add_outcome(success * success,
					static_cast<unsigned int>(state.player1_gold * multiplier),
					static_cast<unsigned int>(state.player2_gold * multiplier));

				add_outcome(success * failure,
					static_cast<unsigned int>(state.player1_gold * multiplier),
					static_cast<unsigned int>(state.player2_gold * penalty));

				add_outcome(failure * success,
					static_cast<unsigned int>(state.player1_gold * penalty),
					static_cast<unsigned int>(state.player2_gold * multiplier));

				add_outcome(failure * failure,
					static_cast<unsigned int>(state.player1_gold * penalty),
					static_cast<unsigned int>(state.player2_gold * penalty));
				break;
			}
			}

			return result;
		}

		static float compute_intrinsic_desire(const State& state)
		{
			if (!is_terminal_state(state)) return 0.0f;
			if (state.player1_gold > state.player2_gold)
				return 1.0f; // Player 1 wins
			return 0.0f;
		}

		static std::string tostr(const State& s)
		{
			return "Turn:" + std::to_string(s.turn) +
				" P1_Gold:" + std::to_string(s.player1_gold) +
				" P2_Gold:" + std::to_string(s.player2_gold);
		}
	};
}