module;
export module game;

import "pch.h";

export namespace game
{
	constexpr unsigned long MAX_ANTICIPATION_NEST_LEVEL = 20;

	struct EmptyConfig {};
	struct StateNode
	{
		//float d_local = 0.0f;	// intrinsic desire
		float d_global = 0.0f;	// propagated desire
		float a[MAX_ANTICIPATION_NEST_LEVEL] = { 0.0f }; // anticipation components, measured engagement of given state

		float sum_A() const
		{
			float sum = 0.0f;
			for (unsigned long i = 0; i < MAX_ANTICIPATION_NEST_LEVEL; ++i)
				sum += a[i];
			return sum;
		}
	};

	template<typename game_t> void traverseR(const typename game_t::State& s, const typename game_t::Config& config, std::function<void(const typename game_t::State& s)> onState)
	{
		using State = typename game_t::State;
		using Config = typename game_t::Config;

		std::set<State> visited;
		std::function<void(const State&)> recurse;
		recurse = [&](const State& s)
			{
				if (visited.find(s) != visited.end()) return;
				visited.insert(s).second;
				for (const auto& ts : game_t::get_transitions(config, s))
					recurse(ts.to);
				onState(s);
			};
		recurse(s);
	};
	template<typename game_t> std::vector<typename game_t::State> serializeR(const typename game_t::State& start_state, const typename game_t::Config& config)
	{
		using State = typename game_t::State;
		std::vector<State> states_serialized;

		traverseR<game_t>(
			start_state,
			config,
			[&](const State& s)
			{
				states_serialized.push_back(s);
			});

		return states_serialized;
	};
	template<typename State> struct GameAnalysis
	{
		std::vector<State> states;
		std::vector<State> states_R;
		std::map<State, StateNode> stateNodes;
		double gds_components[MAX_ANTICIPATION_NEST_LEVEL] = { 0.0 }; // game design score components, A1~5
		double game_design_score;
	};

	template<typename game_t> GameAnalysis<typename game_t::State> analyze(
		typename game_t::State start_state,
		std::function<float(const typename game_t::State&)> compute_intrinsic_desire,
		const typename game_t::Config& config = typename game_t::Config{},
		unsigned long A_nest_level = MAX_ANTICIPATION_NEST_LEVEL
	)
	{
		using State = typename game_t::State;
		using Config = typename game_t::Config;

		if (MAX_ANTICIPATION_NEST_LEVEL < A_nest_level) throw std::runtime_error("A_component_idx exceeds MAX_ANTICIPATION_NEST_LEVEL");

		struct PropagationNode
		{
			float d_local = 0.0f; // intrinsic desire
			float d_global = 0.0f; // propagated desire
			float a = 0.0f;

			struct StepNode
			{
				float a = 0.0f;
				float reach_probability = 0.0f;
			};

			std::map<size_t, StepNode> steps; // propagated anticipation for game design score, step -> current_total_anticipation

		};
		std::vector<State> states;
		std::vector<State> states_R;
		std::map<State, PropagationNode> propNodes;
		std::map<State, StateNode> resultNodes;
		double game_design_scores[MAX_ANTICIPATION_NEST_LEVEL] = { 0.0 };

		states_R = serializeR<game_t>(start_state, config);
		std::reverse_copy(states_R.begin(), states_R.end(), std::back_inserter(states));

		auto buildNodes = [&]()
			{
				for (const auto& state : states)
					propNodes[state] = PropagationNode{};
			};
		auto seedD = [&](unsigned long A_component_idx)
			{
				for (const auto& state : states)
				{
					auto& node = propNodes[state];
					if (A_component_idx == 0)
					{
						node.d_local = compute_intrinsic_desire(state);
					}
					else
					{
						node.d_local = resultNodes[state].a[A_component_idx - 1]; // previous A component as new desire
					}
					//if (state.hp1 != 0 && state.hp2 == 0)		// player1 alive && player2 dead	
					//	node.d_local = 1.0f;					// player1 wins
				}
			};
		auto propagateD = [&]()
			{
				for (const auto& state : states_R)
				{
					auto& node = propNodes[state];
					if (node.d_global) throw std::runtime_error("Global D already seeded");
					node.d_global = node.d_local;

					for (const auto& ts : game_t::get_transitions(config, state))
					{
						auto& tnode = propNodes[ts.to];
						node.d_global += tnode.d_global * ts.probability;
					}
				}
			};
		auto compute_A = [&]()
			{
				for (const auto& state : states_R)
				{
					auto& node = propNodes[state];

					const auto transitions = game_t::get_transitions(config, state);
					const size_t transition_count = transitions.size();
					if (transition_count == 0)
					{
						node.a = 0.0f;
						continue;
					}

					auto compute_A = [&]()
						{
							// Note:
							//
							// 1. This calculation is conceptually a weighted standard deviation. However, its implementation
							//    differs from the "Local Anticipation Formula" described in our paper. The key difference
							//    is the use of D_perspective, which was not part of that paper's formula.
							//
							// 2. We define `D_perspective` ("perspective desire") as the difference in desire (D)
							//    between a subsequent state (s2) and an initial state (s1):
							//    D_perspective = D(s2) - D(s1)
							//
							// 3. Using the raw desire value (D) would yield a measurement from an external, objective
							//    viewpoint. To accurately model the anticipation experienced from within a specific state,
							//    we calculate this change from the state's point of view. This subtraction is what
							//    "perspectivizes" the value.

							float sum_pd = 0.0f;
							for (const auto& ts : game_t::get_transitions(config, state))
							{
								auto& tnode = propNodes[ts.to];
								float perspective_desire = tnode.d_global - node.d_global;
								sum_pd += ts.probability * perspective_desire;  // sum_pe += P ¡¿ D
							}

							float avg_pd = sum_pd;

							float weighted_variance = 0.0f;
							for (const auto& ts : game_t::get_transitions(config, state))
							{
								float perspective_desire = propNodes[ts.to].d_global - node.d_global;
								float diff = perspective_desire - avg_pd; // diff = D - avg(PD)
								float diff_sq = diff * diff;

								weighted_variance += ts.probability * diff_sq;
							}

							// No need to divide by transition_count since we're using weighted sum
							return std::sqrt(weighted_variance);
						};
					node.a = compute_A();
				}
			};
		auto compute_gamedesign_score = [&](unsigned long A_component_idx)
			{
				auto propagate_reach_p = [&]()
					{
						propNodes[start_state].steps[0].reach_probability = 1.0f;
						for (const auto& state : states)
						{
							auto& node = propNodes[state];
							for (const auto& ts : game_t::get_transitions(config, state))
							{
								auto& tnode = propNodes[ts.to];
								for (auto& [si, step] : node.steps)
									tnode.steps[si + 1].reach_probability += step.reach_probability * ts.probability;
							}
						}
					};
				auto propagate_A = [&]()
					{
						propNodes[start_state].steps[0].a = propNodes[start_state].a;
						for (const auto& state : states)
						{
							auto& node = propNodes[state];
							for (const auto& ts : game_t::get_transitions(config, state))
							{
								auto& tnode = propNodes[ts.to];
								for (auto& [si, step] : node.steps)
								{
									// Propagate current accumulated A and add target state's A once
									tnode.steps[si + 1].a += step.a * ts.probability + tnode.a * step.reach_probability * ts.probability;
								}
							}
						}
					};

				double total_end_probability = 0.0;
				auto sum = [&]()
					{
						for (const auto& state : states)
						{
							auto& node = propNodes[state];
							if (game_t::is_terminal_state(state))
							{
								for (const auto& [si, step] : node.steps)
								{
									total_end_probability += step.reach_probability;
									game_design_scores[A_component_idx] += step.a / si;
								}
								//game_design_score += node.a_propagated * node.reach_probability 
							}
						}

						// almost same
						bool almost_same = std::abs(total_end_probability - 1.0f) < 0.0001f;
						if (!almost_same)
							throw std::runtime_error("Bug found in probability propagation");
					};

				propagate_reach_p();
				propagate_A();
				sum();
			};
		auto reflect_to_result = [&](unsigned long A_component_idx)
			{
				for (const auto& state : states)
				{
					auto& node = propNodes[state];
					auto& resultNode = resultNodes[state];
					resultNode.a[A_component_idx] = node.a;
					if (A_component_idx == 0) resultNode.d_global = node.d_global;
					//resultNode.d_local = node.d_local; // not needed
				}
			};

		for (unsigned long component_idx = 0; component_idx < A_nest_level; component_idx++)
		{
			buildNodes();
			seedD(component_idx);
			propagateD();
			compute_A();
			compute_gamedesign_score(component_idx);
			reflect_to_result(component_idx);
		}

		GameAnalysis<State> result{};
		result.states = std::move(states);
		result.states_R = std::move(states_R);
		result.stateNodes = std::move(resultNodes);
		//result.game_design_score = game_design_score;
		for (unsigned long i = 0; i < A_nest_level; i++)
			result.gds_components[i] = game_design_scores[i];
		for (unsigned long i = 0; i < A_nest_level; i++)
			result.game_design_score += game_design_scores[i];
		return result;
	}

	template<typename game_t> inline GameAnalysis<typename game_t::State> analyze()
	{
		using State = typename game_t::State;
		using Config = typename game_t::Config;

		return analyze<game_t>(game_t::initial_state(), game_t::compute_intrinsic_desire, Config());
	}

	template<typename game_t> typename game_t::State run(
		typename game_t::State start_state,
		const std::function<size_t(const typename game_t::State&)> onChoice,
		const typename game_t::Config& config)
	{
		using State = typename game_t::State;

		State current_state = start_state;
		while (!game_t::is_terminal_state(current_state))
		{
			size_t choice;
			choice = onChoice(current_state);
			current_state = game_t::get_transitions(config, current_state)[choice].to;
		}

		return current_state;
	}

	template<typename game_t>
	double compute_gamedesign_score_simulation(const std::function<float(const typename game_t::State&)>& state_to_anticipation, const typename game_t::Config& config)
	{
		using State = typename game_t::State;

		size_t game_count = 10'000;
		double total_anticipation = 0.0;

		for (size_t gi = 0; gi < game_count; gi++)
		{
			std::random_device rd;
			std::mt19937 rng(rd());

			double total_anticipation_for_thisgame = 0.0;
			double avg_anticipation_for_thisgame = 0.0;
			size_t total_turns = 0;

			State current_state = run<game_t>(
				game_t::initial_state(),
				[&](const State& s)
				{
					float anticipation = state_to_anticipation(s);
					total_anticipation_for_thisgame += anticipation;
					total_turns++;

					auto transitions = game_t::get_transitions(config, s);
					std::vector<float> probabilities;
					for (const auto& ts : transitions)
						probabilities.push_back(ts.probability);

					std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
					return dist(rng);
				}, config);

			// Add final state anticipation
			total_anticipation_for_thisgame += state_to_anticipation(current_state);
			//total_turns++;

			avg_anticipation_for_thisgame = total_anticipation_for_thisgame / total_turns;
			total_anticipation += avg_anticipation_for_thisgame;
		}

		double score = total_anticipation / game_count;
		return score;
	}

	template<typename game_t>
	double compute_gamedesign_score_simulation(std::map<typename game_t::State, StateNode>& stateNodes, typename game_t::Config config)
	{
		using State = typename game_t::State;
		return compute_gamedesign_score_simulation<game_t>([&stateNodes](const State& s) { return stateNodes[s].a; }, config);
	}

	// dump most engaging moments
	template<typename game_t>
	void dump_most_engaging_states(GameAnalysis<typename game_t::State>& analysis)
	{
		std::vector<typename game_t::State> states_sorted_by_a = analysis.states;
		std::sort(states_sorted_by_a.begin(), states_sorted_by_a.end(),
			[&](const typename game_t::State& a, const typename game_t::State& b)
			{
				return analysis.stateNodes[a].sum_A() > analysis.stateNodes[b].sum_A();
			});

		// Calculate max state string length
		size_t max_state_length = 5;
		for (const auto& state : states_sorted_by_a) {
			size_t len = game_t::tostr(state).length();
			if (len > max_state_length) {
				max_state_length = len;
			}
		}

		// Header
		printf("%-*s\tD_global", static_cast<int>(max_state_length), "State");
		for (int i = 0; i < MAX_ANTICIPATION_NEST_LEVEL; i++) {
			printf("\tA%d", i+1);
		}
		printf("\tsum(A)\n");

		// Separator
		for (size_t i = 0; i < max_state_length; i++) printf("-");
		printf("\t--------");
		for (int i = 0; i < MAX_ANTICIPATION_NEST_LEVEL; i++) {
			printf("\t----");
		}
		printf("\t------\n");

		// Data
		for (const auto& state : states_sorted_by_a)
		{
			auto& node = analysis.stateNodes[state];
			if (node.sum_A() > 0.0f) {
				printf("%-*s\t%.2f\t",
					static_cast<int>(max_state_length),
					game_t::tostr(state).c_str(),
					node.d_global);

				for (int i = 0; i < MAX_ANTICIPATION_NEST_LEVEL; i++) {
					printf("\t%.2f", node.a[i]);
				}

				printf("\t%.2f\n", node.sum_A());
			}

			if (&state - &states_sorted_by_a[0] >= 25) break;
		}
	}

	template<typename transition_t>
	void sanitize_transitions(std::vector<transition_t>& transitions)
	{
		float total = 0.0f;
		for (const auto& ts : transitions) total += ts.probability;
		//if (std::abs(total - 1.0f) > 0.001f) throw std::runtime_error("Invalid probabilities in transitions");
		for (auto& ts : transitions) ts.probability /= total;
	}
}
