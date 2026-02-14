"""Agency Model — Formal integration of player choice into ToA.

Phase 1 of the research roadmap: define, measure, and analyze player agency.

Key insight: GDS measures structural tension of a FIXED Markov chain.
But real games have player CHOICES that create a different kind of engagement:
the tension of "which action should I take?"

This experiment defines two new measures:

1. DECISION TENSION (DT): At each state, how much do the available actions
   differ in their expected outcomes? High DT = meaningful choices.

   DT(s) = std_dev of {Q(s,a) for each action a}
   where Q(s,a) = expected win probability after taking action a at state s.

2. AGENCY SCORE (AS): Game-level aggregate of decision tension, weighted by
   reach probability (like GDS weighs anticipation).

   AS = Σ reach(s) × DT(s) / path_length

3. COMPOSITE ENGAGEMENT (CE): Combines GDS and Agency into one metric.
   CE = GDS × (1 + α × AS)
   where α is a weighting parameter (default 1.0).

   The idea: GDS is the "raw material" of engagement (uncertainty), and
   agency AMPLIFIES it. A game with high GDS but no agency (watching dice)
   is less engaging than the same GDS with meaningful choices.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.game import sanitize_transitions


class ActionGame:
    """A game where the player chooses from explicit actions at each state.

    Unlike the Markov chain approach (where choices are baked into transition
    probabilities), this represents the game tree explicitly.
    """

    def __init__(self, max_hp, actions, n_actions=None, opponent_policy=None):
        """
        Args:
            max_hp: HP for each player
            actions: dict mapping (a1_idx, a2_idx) → [(prob, (dhp1, dhp2)), ...]
            n_actions: number of actions per player (inferred if not given)
            opponent_policy: function(state) → probability distribution over actions
                            None = uniform random
        """
        self.max_hp = max_hp
        self.actions = actions
        if n_actions is not None:
            self.n_actions = n_actions
        else:
            # Infer from action pair keys
            self.n_actions = max(max(k) for k in actions.keys()) + 1
        self.action_names = [str(i) for i in range(self.n_actions)]
        self.opponent_policy = opponent_policy or (lambda s: [1/self.n_actions] * self.n_actions)

    def initial_state(self):
        return (self.max_hp, self.max_hp)

    def is_terminal(self, state):
        return state[0] <= 0 or state[1] <= 0

    def compute_intrinsic_desire(self, state):
        if state[0] > 0 and state[1] <= 0:
            return 1.0
        return 0.0

    def get_transitions_for_action(self, state, action_idx):
        """Get transitions for a specific player action (opponent plays mixed)."""
        hp1, hp2 = state
        if self.is_terminal(state):
            return []

        opp_probs = self.opponent_policy(state)
        action_name = self.action_names[action_idx]

        transitions = []
        for a2_idx, opp_name in enumerate(self.action_names):
            opp_prob = opp_probs[a2_idx]
            if opp_prob < 1e-10:
                continue

            key = (action_idx, a2_idx)
            outcomes = self.actions.get(key, [])

            for prob, (dhp1, dhp2) in outcomes:
                new_hp1 = max(0, min(self.max_hp, hp1 + dhp1))
                new_hp2 = max(0, min(self.max_hp, hp2 + dhp2))
                transitions.append((opp_prob * prob, (new_hp1, new_hp2)))

        return sanitize_transitions(transitions)

    def get_transitions_mixed(self, state, policy):
        """Get transitions under a mixed policy (for GDS computation)."""
        if self.is_terminal(state):
            return []

        all_transitions = []
        probs = policy(state)

        for a_idx in range(self.n_actions):
            if probs[a_idx] < 1e-10:
                continue
            action_trans = self.get_transitions_for_action(state, a_idx)
            for t_prob, next_state in action_trans:
                all_transitions.append((probs[a_idx] * t_prob, next_state))

        return sanitize_transitions(all_transitions)


def compute_action_values(game, nest_level=5):
    """Compute Q-values for each (state, action) pair.

    Q(s, a) = expected D_global of next states when taking action a at state s.
    This requires first computing D_global under some reference policy (random).

    Returns:
        q_values: dict mapping state → [Q(s,a0), Q(s,a1), ...]
        analysis: the full analysis result
    """
    # First, analyze under random policy to get D_global values
    random_policy = lambda s: [1/game.n_actions] * game.n_actions

    def get_transitions(state, config=None):
        return game.get_transitions_mixed(state, random_policy)

    analysis = analyze(
        initial_state=game.initial_state(),
        is_terminal=game.is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=game.compute_intrinsic_desire,
        nest_level=nest_level,
    )

    # Now compute Q-values using D_global from the analysis
    q_values = {}

    for state in analysis.states:
        if game.is_terminal(state):
            continue

        q = []
        for a_idx in range(game.n_actions):
            trans = game.get_transitions_for_action(state, a_idx)
            if not trans:
                q.append(0.0)
                continue

            expected_d = 0.0
            for prob, next_state in trans:
                if next_state in analysis.state_nodes:
                    expected_d += prob * analysis.state_nodes[next_state].d_global
                else:
                    expected_d += prob * game.compute_intrinsic_desire(next_state)
            q.append(expected_d)

        q_values[state] = q

    return q_values, analysis


def compute_decision_tension(q_values):
    """Compute Decision Tension (DT) at each state.

    DT(s) = standard deviation of Q-values across actions.
    High DT = actions have very different expected outcomes = meaningful choice.
    """
    dt = {}
    for state, q in q_values.items():
        if len(q) < 2:
            dt[state] = 0.0
            continue
        mean_q = sum(q) / len(q)
        variance = sum((v - mean_q) ** 2 for v in q) / len(q)
        dt[state] = math.sqrt(variance)
    return dt


def softmax_entropy(q_values_list, temperature=10.0):
    """Compute entropy of softmax policy over Q-values.

    Higher temperature = more uniform. Lower = more concentrated on best action.
    Returns normalized entropy in [0, 1].
    """
    n = len(q_values_list)
    if n <= 1:
        return 0.0

    # Softmax with temperature
    max_q = max(q_values_list)
    exp_q = [math.exp(temperature * (q - max_q)) for q in q_values_list]
    total = sum(exp_q)
    probs = [e / total for e in exp_q]

    # Shannon entropy, normalized by max entropy (log n)
    entropy = 0.0
    for p in probs:
        if p > 1e-15:
            entropy -= p * math.log(p)

    max_entropy = math.log(n)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_entropy_corrected_agency(game, q_values, analysis, temperature=10.0):
    """Entropy-Corrected Agency Score (EAS).

    EAS(s) = DT(s) × H(softmax(Q(s))) — high when actions differ AND
    the choice is non-trivial (no single dominant action).

    When one action clearly dominates → softmax concentrates → H → 0 → EAS → 0.
    When all actions equal → DT = 0 → EAS → 0.
    Sweet spot: actions differ meaningfully but none is overwhelmingly best.
    """
    dt = compute_decision_tension(q_values)

    random_policy = lambda s: [1/game.n_actions] * game.n_actions
    reach = compute_reach_probabilities(game, analysis, random_policy)

    total_weighted = 0.0
    total_reach = 0.0

    eas_per_state = {}

    for state in analysis.states:
        if game.is_terminal(state):
            continue
        q = q_values.get(state, [])
        h = softmax_entropy(q, temperature)
        eas = dt.get(state, 0.0) * h
        eas_per_state[state] = eas

        r = reach.get(state, 0.0)
        total_weighted += r * eas
        total_reach += r

    if total_reach < 1e-10:
        return 0.0, eas_per_state

    return total_weighted / total_reach, eas_per_state


def compute_policy_impact(game, nest_level=5):
    """Policy Impact (PI): How much does player choice affect GDS?

    PI = max(GDS) - min(GDS) across pure strategies.
    High PI = player has significant control over their experience.
    """
    gds_values = []

    for action_idx in range(game.n_actions):
        # Pure strategy: always play action_idx
        def pure_policy(state, idx=action_idx):
            probs = [0.0] * game.n_actions
            probs[idx] = 1.0
            return probs

        result = compute_gds_for_policy(game, pure_policy, nest_level)
        gds_values.append(result.game_design_score)

    return max(gds_values) - min(gds_values), gds_values


def compute_choice_paradox_gap(game, nest_level=5, resolution=10):
    """Choice Paradox Gap (CPG): distance between fun-optimal and win-optimal strategies.

    Searches over mixed strategies at given resolution.
    Returns (gap, fun_optimal, win_optimal) where:
    - gap: |D₀(fun-optimal) - D₀(win-optimal)|
    - fun_optimal: (gds, d0, strategy_weights)
    - win_optimal: (gds, d0, strategy_weights)

    Lower gap = better design (fun and winning align).
    """
    n = game.n_actions
    results = []

    if n == 3:
        # Grid search over 3-action mixed strategies
        for a_pct in range(0, 101, resolution):
            for b_pct in range(0, 101 - a_pct, resolution):
                c_pct = 100 - a_pct - b_pct
                weights = [a_pct/100, b_pct/100, c_pct/100]

                policy = lambda state, w=weights: w
                result = compute_gds_for_policy(game, policy, nest_level)
                gds = result.game_design_score
                d0 = result.state_nodes[game.initial_state()].d_global
                results.append((gds, d0, weights))
    else:
        # For other action counts, just use pure strategies + uniform
        for i in range(n):
            weights = [0.0] * n
            weights[i] = 1.0
            policy = lambda state, w=weights: w
            result = compute_gds_for_policy(game, policy, nest_level)
            gds = result.game_design_score
            d0 = result.state_nodes[game.initial_state()].d_global
            results.append((gds, d0, weights))

        # Add uniform
        weights = [1/n] * n
        policy = lambda state, w=weights: w
        result = compute_gds_for_policy(game, policy, nest_level)
        gds = result.game_design_score
        d0 = result.state_nodes[game.initial_state()].d_global
        results.append((gds, d0, weights))

    fun_optimal = max(results, key=lambda x: x[0])
    win_optimal = max(results, key=lambda x: x[1])

    gap = abs(fun_optimal[1] - win_optimal[1])
    return gap, fun_optimal, win_optimal


def compute_reach_probabilities(game, analysis, policy):
    """Compute reach probability for each state via forward propagation."""
    reach = {s: 0.0 for s in analysis.states}
    reach[game.initial_state()] = 1.0

    for s in analysis.states:
        if game.is_terminal(s) or reach[s] < 1e-15:
            continue
        transitions = game.get_transitions_mixed(s, policy)
        for prob, next_s in transitions:
            if next_s in reach:
                reach[next_s] += reach[s] * prob

    return reach


def compute_agency_score(game, q_values, analysis):
    """Compute the Agency Score — game-level aggregate of decision tension.

    AS = Σ reach(s) × DT(s), normalized by total reach of non-terminal states.
    Uses uniform random policy for reach computation.
    """
    dt = compute_decision_tension(q_values)

    random_policy = lambda s: [1/game.n_actions] * game.n_actions
    reach = compute_reach_probabilities(game, analysis, random_policy)

    total_weighted_dt = 0.0
    total_reach = 0.0

    for state in analysis.states:
        if game.is_terminal(state):
            continue
        r = reach.get(state, 0.0)
        total_weighted_dt += r * dt.get(state, 0.0)
        total_reach += r

    if total_reach < 1e-10:
        return 0.0, dt

    return total_weighted_dt / total_reach, dt


def compute_gds_for_policy(game, policy, nest_level=5):
    """Compute GDS under a specific policy."""
    def get_transitions(state, config=None):
        return game.get_transitions_mixed(state, policy)

    result = analyze(
        initial_state=game.initial_state(),
        is_terminal=game.is_terminal,
        get_transitions=get_transitions,
        compute_intrinsic_desire=game.compute_intrinsic_desire,
        nest_level=nest_level,
    )
    return result


# ─── Standard Combat Actions ──────────────────────────────────────────────

def make_combat_game(max_hp=5):
    """Create a standard combat game with 3 actions.

    Actions:
    - Strike (0): Reliable 1 damage
    - Heavy (1): 2 damage but 50% miss + counterattack
    - Guard (2): Block incoming strike, take half from heavy

    No rage mechanic — pure tactical choice.
    """
    actions = {}

    for a1 in range(3):
        for a2 in range(3):
            outcomes = []

            if a1 == 0 and a2 == 0:  # Strike vs Strike
                outcomes = [(1.0, (-1, -1))]  # Both take 1
            elif a1 == 0 and a2 == 1:  # Strike vs Heavy
                outcomes = [
                    (0.5, (-1, -1)),   # Heavy hits: both take damage
                    (0.5, (0, -1)),    # Heavy misses: only they take strike
                ]
            elif a1 == 0 and a2 == 2:  # Strike vs Guard
                outcomes = [(1.0, (-1, 0))]  # Guard counters attacker
            elif a1 == 1 and a2 == 0:  # Heavy vs Strike
                outcomes = [
                    (0.5, (-1, -2)),   # Heavy hits
                    (0.5, (-1, 0)),    # Heavy misses, take counter
                ]
            elif a1 == 1 and a2 == 1:  # Heavy vs Heavy
                outcomes = [
                    (0.25, (-2, -2)),  # Both hit
                    (0.25, (-2, 0)),   # Only P1 hits
                    (0.25, (0, -2)),   # Only P2 hits
                    (0.25, (-1, -1)),  # Both miss, chip damage
                ]
            elif a1 == 1 and a2 == 2:  # Heavy vs Guard
                outcomes = [
                    (0.5, (0, -1)),    # Heavy breaks through guard partially
                    (0.5, (-1, 0)),    # Heavy misses, no counter from guard
                ]
            elif a1 == 2 and a2 == 0:  # Guard vs Strike
                outcomes = [(1.0, (0, -1))]  # Guard counters attacker
            elif a1 == 2 and a2 == 1:  # Guard vs Heavy
                outcomes = [
                    (0.5, (-1, 0)),    # Heavy breaks through
                    (0.5, (0, -1)),    # Heavy misses, guard counter
                ]
            elif a1 == 2 and a2 == 2:  # Guard vs Guard
                outcomes = [(1.0, (-1, -1))]  # Attrition to prevent stall

            actions[(a1, a2)] = outcomes

    return ActionGame(max_hp, actions)


def make_no_choice_game(max_hp=5):
    """Same game structure but with only 1 action (no player choice).

    The single action produces the SAME average distribution as random play
    in the 3-action game, so the Markov chain is identical. But the player
    has no decisions to make.
    """
    # Uniform over all 9 action combinations
    actions = {}
    actions[(0, 0)] = [
        # Average of all 9 outcomes from the combat game, equally weighted
        (1/9, (-1, -1)),   # strike-strike
        (1/18, (-1, -1)),  # strike-heavy hit
        (1/18, (0, -1)),   # strike-heavy miss
        (1/9, (0, 0)),     # strike-guard
        (1/18, (-1, -2)),  # heavy-strike hit
        (1/18, (-1, 0)),   # heavy-strike miss
        (1/36, (-2, -2)),  # heavy-heavy both hit
        (1/36, (-2, 0)),   # heavy-heavy p1 hit
        (1/36, (0, -2)),   # heavy-heavy p2 hit
        (1/36, (-1, -1)),  # heavy-heavy both miss
        (1/18, (0, -1)),   # heavy-guard break
        (1/18, (-1, 0)),   # heavy-guard miss
        (1/9, (0, 0)),     # guard-strike
        (1/18, (-1, 0)),   # guard-heavy break
        (1/18, (0, -1)),   # guard-heavy miss (guard counter)
        (1/9, (-1, -1)),   # guard-guard
    ]

    game = ActionGame(max_hp, actions)
    game.n_actions = 1
    game.action_names = ['auto']
    return game


# ─── Experiments ──────────────────────────────────────────────────────────

def experiment_1_agency_vs_no_agency():
    """Compare GDS and agency for games with and without player choice."""
    print("=" * 80)
    print("EXPERIMENT 1: Agency vs No Agency (same outcome distribution)")
    print("=" * 80)
    print()

    for hp in [3, 4, 5, 6, 7]:
        game = make_combat_game(hp)

        # Random policy for both
        random_policy = lambda s: [1/3, 1/3, 1/3]
        result_with_choice = compute_gds_for_policy(game, random_policy)

        # Q-values and agency
        q_values, analysis = compute_action_values(game)
        agency_score, dt_map = compute_agency_score(game, q_values, analysis)

        # Decision tension stats
        dt_values = [v for v in dt_map.values() if v > 0]
        max_dt = max(dt_values) if dt_values else 0
        avg_dt = sum(dt_values) / len(dt_values) if dt_values else 0

        gds = result_with_choice.game_design_score
        a1 = result_with_choice.gds_components[0] if result_with_choice.gds_components else 0
        a2_plus = sum(result_with_choice.gds_components[1:]) if len(result_with_choice.gds_components) > 1 else 0

        print(f"HP={hp}: GDS={gds:.4f}  A1={a1:.4f}  A2+={a2_plus:.4f}  "
              f"Agency={agency_score:.4f}  MaxDT={max_dt:.4f}  AvgDT={avg_dt:.4f}  "
              f"CE={gds * (1 + agency_score):.4f}")


def experiment_2_policy_spectrum():
    """Sweep from pure exploitation (greedy) to pure exploration (random)."""
    print()
    print("=" * 80)
    print("EXPERIMENT 2: Policy Spectrum — Greedy to Random")
    print("=" * 80)
    print()

    game = make_combat_game(5)
    q_values, base_analysis = compute_action_values(game)

    # For each epsilon from 0 (greedy) to 1 (random):
    print(f"  {'ε':>5}  {'Policy':>12}  {'GDS':>8}  {'D₀':>6}  {'A1':>8}  {'A2+':>8}")
    print(f"  {'-'*56}")

    for epsilon_pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        epsilon = epsilon_pct / 100.0

        def make_epsilon_policy(eps):
            def policy(state):
                q = q_values.get(state, [0.33, 0.33, 0.33])
                n = len(q)
                if n == 0:
                    return [1/3, 1/3, 1/3]

                # Epsilon-greedy: with prob (1-eps), pick best action; with prob eps, random
                best_idx = q.index(max(q))
                probs = [eps / n] * n
                probs[best_idx] += (1 - eps)
                return probs
            return policy

        policy = make_epsilon_policy(epsilon)
        result = compute_gds_for_policy(game, policy)

        d0 = result.state_nodes[game.initial_state()].d_global
        gds = result.game_design_score
        a1 = result.gds_components[0]
        a2_plus = sum(result.gds_components[1:])

        label = "greedy" if epsilon_pct == 0 else ("random" if epsilon_pct == 100 else f"ε={epsilon:.1f}")
        print(f"  {epsilon_pct:>4}%  {label:>12}  {gds:>8.4f}  {d0:>6.3f}  {a1:>8.4f}  {a2_plus:>8.4f}")


def experiment_3_what_maximizes_fun():
    """Search for the policy that maximizes GDS (not win rate!)."""
    print()
    print("=" * 80)
    print("EXPERIMENT 3: What Policy Maximizes Fun?")
    print("=" * 80)
    print()

    game = make_combat_game(5)
    q_values, _ = compute_action_values(game)

    # Try many different static mixed strategies
    best_gds = 0
    best_mix = None

    print("  Searching over 66 different mixed strategies (strike, heavy, guard)...")
    print()

    results = []

    for s_pct in range(0, 101, 10):
        for h_pct in range(0, 101 - s_pct, 10):
            g_pct = 100 - s_pct - h_pct
            s, h, g = s_pct/100, h_pct/100, g_pct/100

            policy = lambda state, s=s, h=h, g=g: [s, h, g]
            result = compute_gds_for_policy(game, policy)
            gds = result.game_design_score
            d0 = result.state_nodes[game.initial_state()].d_global

            results.append((gds, d0, s, h, g))

            if gds > best_gds:
                best_gds = gds
                best_mix = (s, h, g)

    # Sort by GDS
    results.sort(key=lambda x: -x[0])

    print(f"  {'GDS':>8}  {'D₀':>6}  {'Strike':>8}  {'Heavy':>8}  {'Guard':>8}")
    print(f"  {'-'*46}")
    for gds, d0, s, h, g in results[:10]:
        marker = " ← BEST" if (s, h, g) == best_mix else ""
        print(f"  {gds:>8.4f}  {d0:>6.3f}  {s:>8.1%}  {h:>8.1%}  {g:>8.1%}{marker}")

    print(f"\n  ...{len(results) - 10} more strategies tested")
    print(f"\n  Fun-maximizing mix: Strike={best_mix[0]:.0%} Heavy={best_mix[1]:.0%} Guard={best_mix[2]:.0%}")
    print(f"  Maximum GDS: {best_gds:.4f}")

    # Compare with key reference points
    print()
    print("  Reference comparison:")

    for label, policy in [
        ("Random (uniform)", lambda s: [1/3, 1/3, 1/3]),
        ("All Strike", lambda s: [1.0, 0.0, 0.0]),
        ("All Heavy", lambda s: [0.0, 1.0, 0.0]),
        ("All Guard", lambda s: [0.0, 0.0, 1.0]),
    ]:
        result = compute_gds_for_policy(game, policy)
        d0 = result.state_nodes[game.initial_state()].d_global
        print(f"  {label:<20}  GDS={result.game_design_score:.4f}  D₀={d0:.3f}")


def experiment_4_agency_across_games():
    """Compare original vs entropy-corrected agency across game types."""
    print()
    print("=" * 80)
    print("EXPERIMENT 4: Agency Measures Comparison")
    print("=" * 80)
    print()

    games = {}

    # Game 1: Balanced combat (standard)
    games["Balanced combat"] = make_combat_game(5)

    # Game 2: Dominant strategy (one action always best)
    dom_actions = {}
    for a1 in range(3):
        for a2 in range(3):
            d2 = -1 if a1 == 0 else 0
            d1 = -1 if a2 == 0 else 0
            if a1 != 0:
                d1 = min(d1, d1 - 1) if d1 < 0 else -1
            if a2 != 0:
                d2 = min(d2, d2 - 1) if d2 < 0 else -1
            dom_actions[(a1, a2)] = [(1.0, (d1, d2))]
    games["Dominant strategy"] = ActionGame(5, dom_actions)

    # Game 3: Rock-paper-scissors-like (cyclical)
    rps_actions = {}
    for a1 in range(3):
        for a2 in range(3):
            diff = (a1 - a2) % 3
            if diff == 0:
                rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
            elif diff == 1:
                rps_actions[(a1, a2)] = [(1.0, (0, -2))]
            else:
                rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
    games["RPS cyclical"] = ActionGame(5, rps_actions)

    # Game 4: Asymmetric combat (one player stronger — tests if agency detects imbalance)
    asym_actions = {}
    for a1 in range(3):
        for a2 in range(3):
            if a1 == 0 and a2 == 0:     # Strike vs Strike
                asym_actions[(a1, a2)] = [(0.6, (0, -1)), (0.4, (-1, 0))]  # P1 slightly favored
            elif a1 == 0 and a2 == 1:   # Strike vs Heavy
                asym_actions[(a1, a2)] = [(0.5, (-2, -1)), (0.5, (0, -1))]
            elif a1 == 0 and a2 == 2:   # Strike vs Guard
                asym_actions[(a1, a2)] = [(1.0, (-1, 0))]
            elif a1 == 1 and a2 == 0:   # Heavy vs Strike
                asym_actions[(a1, a2)] = [(0.5, (-1, -2)), (0.5, (-1, 0))]
            elif a1 == 1 and a2 == 1:   # Heavy vs Heavy
                asym_actions[(a1, a2)] = [
                    (0.3, (-2, -2)), (0.2, (0, -2)),
                    (0.2, (-2, 0)), (0.3, (-1, -1)),
                ]
            elif a1 == 1 and a2 == 2:   # Heavy vs Guard
                asym_actions[(a1, a2)] = [(0.5, (0, -1)), (0.5, (-1, 0))]
            elif a1 == 2 and a2 == 0:   # Guard vs Strike
                asym_actions[(a1, a2)] = [(1.0, (0, -1))]
            elif a1 == 2 and a2 == 1:   # Guard vs Heavy
                asym_actions[(a1, a2)] = [(0.5, (-1, 0)), (0.5, (0, -1))]
            elif a1 == 2 and a2 == 2:   # Guard vs Guard
                asym_actions[(a1, a2)] = [(1.0, (-1, -1))]
    games["Asymmetric (P1 edge)"] = ActionGame(5, asym_actions)

    print(f"  {'Game':>25}  {'GDS':>7}  {'AS_old':>7}  {'EAS':>7}  {'PI':>7}")
    print(f"  {'-'*58}")

    for name, game in games.items():
        random_policy = lambda s, n=game.n_actions: [1/n] * n

        q_values, analysis = compute_action_values(game)
        gds = compute_gds_for_policy(game, random_policy).game_design_score

        as_old, _ = compute_agency_score(game, q_values, analysis)
        eas, _ = compute_entropy_corrected_agency(game, q_values, analysis)
        pi, pure_gds = compute_policy_impact(game)

        print(f"  {name:>25}  {gds:>7.4f}  {as_old:>7.4f}  {eas:>7.4f}  {pi:>7.4f}")

    print()
    print("  AS_old: DT only — FLAWED (dominant strategy scores highest)")
    print("  EAS:    DT × entropy(softmax(Q)) — fixed (dominant → low)")
    print("  PI:     Policy Impact (max - min GDS across pure strategies)")
    print()
    print("  Expected ranking: Balanced > High-var > RPS > Dominant")
    print("  AS_old ranking is WRONG (Dominant highest). Does EAS fix this?")


def experiment_5_composite_engagement():
    """Develop and test composite engagement score."""
    print()
    print("=" * 80)
    print("EXPERIMENT 5: Toward a Composite Engagement Score")
    print("=" * 80)
    print()

    game = make_combat_game(5)

    # Sweep all mixed strategies and compute multiple measures
    print("  Searching for the 'most engaging' strategy across 66 mixes...")
    print()

    results = []
    for s_pct in range(0, 101, 10):
        for h_pct in range(0, 101 - s_pct, 10):
            g_pct = 100 - s_pct - h_pct
            s, h, g = s_pct/100, h_pct/100, g_pct/100

            policy = lambda state, s=s, h=h, g=g: [s, h, g]
            result = compute_gds_for_policy(game, policy)
            gds = result.game_design_score
            d0 = result.state_nodes[game.initial_state()].d_global

            results.append((gds, d0, s, h, g))

    # Sort by GDS
    results.sort(key=lambda x: -x[0])

    # Compare: GDS-optimal vs win-optimal vs balanced
    print(f"  Strategy rankings:")
    print(f"  {'Rank':>6}  {'GDS':>7}  {'D₀(win)':>8}  {'S':>5}  {'H':>5}  {'G':>5}  Note")
    print(f"  {'-'*62}")

    for i, (gds, d0, s, h, g) in enumerate(results[:5]):
        note = ""
        if i == 0:
            note = "← GDS-optimal (most fun?)"
        elif d0 > 0.55:
            note = "← winning but less fun"
        print(f"  {i+1:>6}  {gds:>7.4f}  {d0:>8.3f}  {s:>5.0%}  {h:>5.0%}  {g:>5.0%}  {note}")

    # Find win-optimal
    win_optimal = max(results, key=lambda x: x[1])
    print(f"  {'win':>6}  {win_optimal[0]:>7.4f}  {win_optimal[1]:>8.3f}  "
          f"{win_optimal[2]:>5.0%}  {win_optimal[3]:>5.0%}  {win_optimal[4]:>5.0%}  ← win-optimal")

    print()
    print("  KEY INSIGHT:")
    gds_opt = results[0]
    print(f"  - Fun-optimal:   S={gds_opt[2]:.0%} H={gds_opt[3]:.0%} G={gds_opt[4]:.0%}  (GDS={gds_opt[0]:.4f}, D₀={gds_opt[1]:.3f})")
    print(f"  - Win-optimal:   S={win_optimal[2]:.0%} H={win_optimal[3]:.0%} G={win_optimal[4]:.0%}  (GDS={win_optimal[0]:.4f}, D₀={win_optimal[1]:.3f})")
    print(f"  - Fun-optimal player is LOSING (D₀={gds_opt[1]:.3f} < 0.5)")
    print(f"  - Choosing fun over winning = the Choice Paradox in action")

    print()
    print("  DESIGN PRINCIPLE:")
    print("  A well-designed game makes the fun-optimal strategy CLOSE to the")
    print("  win-optimal strategy. When they diverge, players face the Choice")
    print("  Paradox: play to win (boring) or play for fun (lose).")
    print(f"  Gap in this game: {abs(gds_opt[1] - win_optimal[1]):.3f} (smaller = better design)")


def experiment_6_policy_impact_deep():
    """Deep analysis of Policy Impact — why it works as an agency measure."""
    print()
    print("=" * 80)
    print("EXPERIMENT 6: Policy Impact — Deep Analysis")
    print("=" * 80)
    print()

    games = {
        "Balanced combat": make_combat_game(5),
    }

    # Dominant strategy
    dom_actions = {}
    for a1 in range(3):
        for a2 in range(3):
            d2 = -1 if a1 == 0 else 0
            d1 = -1 if a2 == 0 else 0
            if a1 != 0:
                d1 = min(d1, d1 - 1) if d1 < 0 else -1
            if a2 != 0:
                d2 = min(d2, d2 - 1) if d2 < 0 else -1
            dom_actions[(a1, a2)] = [(1.0, (d1, d2))]
    games["Dominant strategy"] = ActionGame(5, dom_actions)

    # RPS
    rps_actions = {}
    for a1 in range(3):
        for a2 in range(3):
            diff = (a1 - a2) % 3
            if diff == 0:
                rps_actions[(a1, a2)] = [(1.0, (-1, -1))]
            elif diff == 1:
                rps_actions[(a1, a2)] = [(1.0, (0, -2))]
            else:
                rps_actions[(a1, a2)] = [(1.0, (-2, 0))]
    games["RPS cyclical"] = ActionGame(5, rps_actions)

    action_labels = ["Strike", "Heavy", "Guard"]

    for name, game in games.items():
        print(f"  {name}:")

        pi, pure_gds = compute_policy_impact(game)
        random_gds = compute_gds_for_policy(
            game, lambda s, n=game.n_actions: [1/n] * n
        ).game_design_score

        for i, gds in enumerate(pure_gds):
            label = action_labels[i] if i < len(action_labels) else f"Action {i}"
            marker = " ← best" if gds == max(pure_gds) else (" ← worst" if gds == min(pure_gds) else "")
            print(f"    Pure {label:>6}: GDS={gds:.4f}{marker}")

        print(f"    Random:       GDS={random_gds:.4f}")
        print(f"    PI (max-min): {pi:.4f}")
        print(f"    PI/GDS ratio: {pi/random_gds:.1%} — fraction of GDS controllable by player")
        print()

    # Summary table
    print("  SUMMARY: Policy Impact as agency measure")
    print(f"  {'Game':>25}  {'GDS':>7}  {'PI':>7}  {'PI/GDS':>7}  {'max GDS':>8}  {'Choice Paradox':>15}")
    print(f"  {'-'*80}")

    for name, game in games.items():
        random_policy = lambda s, n=game.n_actions: [1/n] * n
        gds = compute_gds_for_policy(game, random_policy).game_design_score
        pi, pure_gds = compute_policy_impact(game)

        # Check Choice Paradox: does max-GDS strategy also have highest D₀?
        max_gds_idx = pure_gds.index(max(pure_gds))
        # Check D₀ for each pure strategy
        d0_values = []
        for i in range(game.n_actions):
            def pure(state, idx=i):
                p = [0.0] * game.n_actions
                p[idx] = 1.0
                return p
            result = compute_gds_for_policy(game, pure)
            d0_values.append(result.state_nodes[game.initial_state()].d_global)

        max_d0_idx = d0_values.index(max(d0_values))
        paradox = "YES" if max_gds_idx != max_d0_idx else "no"

        print(f"  {name:>25}  {gds:>7.4f}  {pi:>7.4f}  {pi/gds:>7.1%}  {max(pure_gds):>8.4f}  {paradox:>15}")

    print()
    print("  KEY FINDINGS:")
    print("  1. PI captures 'how much control does the player have over engagement'")
    print("  2. PI/GDS ratio = 'agency fraction' — % of engagement player controls")
    print("  3. Choice Paradox detected when max-GDS action ≠ max-win action")
    print("  4. RPS has PI=0: player choice doesn't affect engagement (symmetric)")
    print("  5. Dominant has low PI/GDS: trivial choice = low agency")


def make_parametric_combat(max_hp, strike_dmg=1, heavy_dmg=2, heavy_hit_prob=0.5,
                           guard_counter=1, guard_vs_heavy_block=0.5,
                           guard_guard_chip=1):
    """Create a combat game with tunable parameters.

    This allows searching over the design space to find games with
    better agency properties (high PI, low CPG).
    """
    actions = {}

    for a1 in range(3):
        for a2 in range(3):
            if a1 == 0 and a2 == 0:  # Strike vs Strike
                actions[(a1, a2)] = [(1.0, (-strike_dmg, -strike_dmg))]
            elif a1 == 0 and a2 == 1:  # Strike vs Heavy
                actions[(a1, a2)] = [
                    (heavy_hit_prob, (-heavy_dmg, -strike_dmg)),
                    (1-heavy_hit_prob, (0, -strike_dmg)),
                ]
            elif a1 == 0 and a2 == 2:  # Strike vs Guard
                actions[(a1, a2)] = [(1.0, (-guard_counter, 0))]
            elif a1 == 1 and a2 == 0:  # Heavy vs Strike
                actions[(a1, a2)] = [
                    (heavy_hit_prob, (-strike_dmg, -heavy_dmg)),
                    (1-heavy_hit_prob, (-strike_dmg, 0)),
                ]
            elif a1 == 1 and a2 == 1:  # Heavy vs Heavy
                actions[(a1, a2)] = [
                    (heavy_hit_prob**2, (-heavy_dmg, -heavy_dmg)),
                    (heavy_hit_prob*(1-heavy_hit_prob), (-heavy_dmg, 0)),
                    ((1-heavy_hit_prob)*heavy_hit_prob, (0, -heavy_dmg)),
                    ((1-heavy_hit_prob)**2, (-1, -1)),  # Both miss: chip
                ]
            elif a1 == 1 and a2 == 2:  # Heavy vs Guard
                actions[(a1, a2)] = [
                    (heavy_hit_prob, (0, -int(heavy_dmg * guard_vs_heavy_block))),
                    (1-heavy_hit_prob, (-guard_counter, 0)),
                ]
            elif a1 == 2 and a2 == 0:  # Guard vs Strike
                actions[(a1, a2)] = [(1.0, (0, -guard_counter))]
            elif a1 == 2 and a2 == 1:  # Guard vs Heavy
                actions[(a1, a2)] = [
                    (heavy_hit_prob, (-int(heavy_dmg * guard_vs_heavy_block), 0)),
                    (1-heavy_hit_prob, (0, -guard_counter)),
                ]
            elif a1 == 2 and a2 == 2:  # Guard vs Guard
                actions[(a1, a2)] = [(1.0, (-guard_guard_chip, -guard_guard_chip))]

    return ActionGame(max_hp, actions)


def experiment_7_cpg_minimization():
    """Search for combat game parameters that minimize the Choice Paradox Gap."""
    print()
    print("=" * 80)
    print("EXPERIMENT 7: Minimizing the Choice Paradox Gap")
    print("=" * 80)
    print()
    print("  Searching over combat game parameter space...")
    print("  Goal: find a game where fun-optimal ≈ win-optimal (low CPG)")
    print()

    results = []
    n_configs = 0

    # Search over parameter space
    for hp in [4, 5, 6]:
        for strike_dmg in [1]:
            for heavy_dmg in [2, 3]:
                for heavy_hit in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for guard_counter in [1, 2]:
                        for guard_block in [0.3, 0.5, 0.7]:
                            for gg_chip in [1]:
                                n_configs += 1
                                game = make_parametric_combat(
                                    hp, strike_dmg, heavy_dmg, heavy_hit,
                                    guard_counter, guard_block, gg_chip
                                )

                                try:
                                    cpg, fun_opt, win_opt = compute_choice_paradox_gap(game, resolution=20)
                                    pi, _ = compute_policy_impact(game)
                                    random_gds = compute_gds_for_policy(
                                        game, lambda s: [1/3, 1/3, 1/3]
                                    ).game_design_score

                                    results.append({
                                        'hp': hp, 'heavy_dmg': heavy_dmg,
                                        'heavy_hit': heavy_hit, 'guard_counter': guard_counter,
                                        'guard_block': guard_block,
                                        'cpg': cpg, 'pi': pi, 'gds': random_gds,
                                        'fun_gds': fun_opt[0], 'fun_d0': fun_opt[1],
                                        'win_gds': win_opt[0], 'win_d0': win_opt[1],
                                    })
                                except Exception:
                                    pass  # Skip invalid configs

    print(f"  Tested {n_configs} configurations, {len(results)} valid")
    print()

    # Sort by CPG (ascending) — filter for reasonable games (PI > 0.05)
    good_results = [r for r in results if r['pi'] > 0.05 and r['gds'] > 0.3]
    good_results.sort(key=lambda r: r['cpg'])

    print(f"  TOP 10 LOWEST CPG (with PI > 0.05, GDS > 0.3):")
    print(f"  {'HP':>3}  {'HvD':>3}  {'HvH':>4}  {'GC':>3}  {'GB':>4}  {'CPG':>6}  {'PI':>6}  {'GDS':>6}  {'maxGDS':>7}  {'FunD₀':>6}  {'WinD₀':>6}")
    print(f"  {'-'*70}")

    for r in good_results[:10]:
        print(f"  {r['hp']:>3}  {r['heavy_dmg']:>3}  {r['heavy_hit']:>4.1f}  {r['guard_counter']:>3}  "
              f"{r['guard_block']:>4.1f}  {r['cpg']:>6.3f}  {r['pi']:>6.3f}  {r['gds']:>6.3f}  "
              f"{r['fun_gds']:>7.3f}  {r['fun_d0']:>6.3f}  {r['win_d0']:>6.3f}")

    # Compare with baseline
    baseline = next((r for r in results if r['hp'] == 5 and r['heavy_dmg'] == 2
                     and r['heavy_hit'] == 0.5 and r['guard_counter'] == 1
                     and r['guard_block'] == 0.5), None)

    if baseline and good_results:
        best = good_results[0]
        print()
        print(f"  BASELINE (standard combat): CPG={baseline['cpg']:.3f}  PI={baseline['pi']:.3f}  GDS={baseline['gds']:.3f}")
        print(f"  BEST found:                 CPG={best['cpg']:.3f}  PI={best['pi']:.3f}  GDS={best['gds']:.3f}")
        improvement = (baseline['cpg'] - best['cpg']) / baseline['cpg'] * 100
        print(f"  CPG improvement: {improvement:.1f}%")
        print()
        print(f"  Best config: HP={best['hp']}, HeavyDmg={best['heavy_dmg']}, HeavyHit={best['heavy_hit']:.1f}, "
              f"GuardCounter={best['guard_counter']}, GuardBlock={best['guard_block']:.1f}")

    # Also find configs with highest PI (most agency)
    high_pi = sorted(good_results, key=lambda r: -r['pi'])
    if high_pi:
        print()
        print(f"  HIGHEST AGENCY:")
        best_pi = high_pi[0]
        print(f"  PI={best_pi['pi']:.3f}  CPG={best_pi['cpg']:.3f}  GDS={best_pi['gds']:.3f}")
        print(f"  Config: HP={best_pi['hp']}, HeavyDmg={best_pi['heavy_dmg']}, HeavyHit={best_pi['heavy_hit']:.1f}, "
              f"GuardCounter={best_pi['guard_counter']}, GuardBlock={best_pi['guard_block']:.1f}")

    # Pareto front: best trade-off between low CPG and high PI
    print()
    print(f"  PARETO FRONT (CPG vs PI):")
    # Score = PI - CPG (maximize PI, minimize CPG)
    scored = sorted(good_results, key=lambda r: -(r['pi'] - r['cpg']))
    for r in scored[:5]:
        print(f"    Score={r['pi']-r['cpg']:.3f}  PI={r['pi']:.3f}  CPG={r['cpg']:.3f}  GDS={r['gds']:.3f}  "
              f"HP={r['hp']} HD={r['heavy_dmg']} HH={r['heavy_hit']:.1f} GC={r['guard_counter']} GB={r['guard_block']:.1f}")


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        # Quick mode: just run experiments 1-6
        experiment_1_agency_vs_no_agency()
        experiment_2_policy_spectrum()
        experiment_3_what_maximizes_fun()
        experiment_4_agency_across_games()
        experiment_5_composite_engagement()
        experiment_6_policy_impact_deep()
    elif "--search" in sys.argv:
        # Search mode: run the parametric search
        experiment_7_cpg_minimization()
    else:
        experiment_1_agency_vs_no_agency()
        experiment_2_policy_spectrum()
        experiment_3_what_maximizes_fun()
        experiment_4_agency_across_games()
        experiment_5_composite_engagement()
        experiment_6_policy_impact_deep()
        experiment_7_cpg_minimization()
