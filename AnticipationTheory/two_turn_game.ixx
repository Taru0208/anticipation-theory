module;
export module two_turn_game;
import "pch.h";
import game;
using namespace game;

export namespace two_turn_game
{
	struct Config
	{
		float probability1 = 0.50f;
		float probability2 = 0.50f;
		float probability3 = 0.50f;

		Config()
		{
		}
	};
	struct State
	{
		unsigned char turn;
		unsigned int node_idx_in_turn;
		auto operator <=> (const State&) const = default;
	};

	struct Transition { float probability; State to; };

	struct Game
	{
		typedef Config Config;
		typedef State State;

		static State initial_state() { return State{ 0, 0 }; }

		static bool is_terminal_state(const State& s) { return s.turn >= 2; }

		static std::vector<Transition> get_transitions(const Config& config, const State& state)
		{
			std::vector<Transition> result;

			if (state.turn == 0 && state.node_idx_in_turn == 0)
			{
				// turn0_ni0 -> turn1 intermediates
				result.push_back({ config.probability1, State{ 1, 0 } });
				result.push_back({ 1 - config.probability1, State{ 1, 1 } });
			}
			else if (state.turn == 1)
			{
				if (state.node_idx_in_turn == 0)
				{
					// turn1_ni0 -> turn2 terminals (30% win, 70% loss)
					result.push_back({ config.probability2, State{ 2, 0 } });
					result.push_back({ 1 - config.probability2, State{ 2, 1 } });
				}
				else if (state.node_idx_in_turn == 1)
				{
					// turn1_ni1 -> turn2 terminals (80% win, 20% loss)  
					result.push_back({ config.probability3, State{ 2, 0 } });
					result.push_back({ 1 - config.probability3, State{ 2, 1 } });
				}
			}

			sanitize_transitions(result);
			return result;
		}

		static float compute_intrinsic_desire(const State& state)
		{
			return state.turn == 2 && state.node_idx_in_turn == 0 ? 1.0f : 0.0f;
		}

		static std::string tostr(const State& s)
		{
			return "Turn:" + std::to_string(s.turn) +
				" Node:" + std::to_string(s.node_idx_in_turn);
		}
	};
}

export namespace game
{
	template<typename State> void two_turn_dump_most_fun_moments_a12(
		const std::vector<State>& states,
		std::map<State, StateNode>& stateNodes,
		std::map<State, StateNode>& stateNodes2)
	{
		std::vector<State> states_sorted_by_a12 = states;
		std::sort(states_sorted_by_a12.begin(), states_sorted_by_a12.end(),
			[&](const State& a, const State& b)
			{
				return (stateNodes[a].a + stateNodes2[a].a) > (stateNodes[b].a + stateNodes2[b].a);
			});

		printf("Most engaging moments(sorted by A1+A2)\n");
		printf("State\t\t\tA1\tA2\tSUM\n");
		for (const auto& state : states_sorted_by_a12)
		{
			auto& node = stateNodes[state];
			auto& node2 = stateNodes2[state];
			if (node.a > 0.0f || node2.a > 0.0f)
				printf("turn%d_ni%d\t\t%.3f\t%.3f\t%.3f\n",
					state.turn, state.node_idx_in_turn,
					node.a, node2.a, node.a + node2.a);
		}
	}
}