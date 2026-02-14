"""Tests for Asymmetric Combat ultra-high GDS game."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toa.engine import analyze
from toa.games.asymmetric_combat import AsymmetricCombat


class TestAsymmetricCombat(unittest.TestCase):

    def _analyze(self, max_hp=10, nest_level=5):
        config = AsymmetricCombat.Config(max_hp=max_hp)
        return analyze(
            initial_state=AsymmetricCombat.initial_state(config),
            is_terminal=AsymmetricCombat.is_terminal,
            get_transitions=AsymmetricCombat.get_transitions,
            compute_intrinsic_desire=AsymmetricCombat.compute_intrinsic_desire,
            config=config,
            nest_level=nest_level,
        )

    def test_basic_analysis(self):
        result = self._analyze(max_hp=5, nest_level=5)
        self.assertGreater(result.game_design_score, 0)
        self.assertTrue(len(result.states) > 10)

    def test_fairness(self):
        """Game should be perfectly fair (D₀ = 0.5)."""
        for hp in [3, 5, 7, 10]:
            config = AsymmetricCombat.Config(max_hp=hp)
            result = self._analyze(max_hp=hp)
            init = AsymmetricCombat.initial_state(config)
            d0 = result.state_nodes[init].d_global
            self.assertAlmostEqual(d0, 0.5, places=3,
                                   msg=f"HP={hp}: D₀={d0}, expected 0.5")

    def test_gds_grows_with_hp(self):
        """GDS should grow with higher HP (more depth)."""
        gds_values = []
        for hp in [3, 5, 7, 10]:
            result = self._analyze(max_hp=hp, nest_level=5)
            gds_values.append(result.game_design_score)

        for i in range(1, len(gds_values)):
            self.assertGreater(gds_values[i], gds_values[i - 1],
                               f"GDS should grow: HP progression yielded {gds_values}")

    def test_gds_grows_with_nest(self):
        """Higher nest level should reveal more anticipation."""
        gds_5 = self._analyze(max_hp=7, nest_level=5).game_design_score
        gds_7 = self._analyze(max_hp=7, nest_level=7).game_design_score
        self.assertGreater(gds_7, gds_5)

    def test_beats_hpgame(self):
        """At HP=5, should beat standard HpGame GDS (~0.43)."""
        result = self._analyze(max_hp=5, nest_level=5)
        self.assertGreater(result.game_design_score, 0.43)

    def test_beats_hpgame_rage(self):
        """At HP=7, should beat HpGame_Rage GDS (~0.55)."""
        result = self._analyze(max_hp=7, nest_level=5)
        self.assertGreater(result.game_design_score, 0.55)

    def test_ultra_high_gds(self):
        """At HP=10 nest=10, GDS should exceed 2.0."""
        result = self._analyze(max_hp=10, nest_level=10)
        self.assertGreater(result.game_design_score, 2.0,
                           f"Expected GDS > 2.0, got {result.game_design_score}")

    def test_no_boring_states(self):
        """All non-terminal states should have meaningful A₁."""
        result = self._analyze(max_hp=5, nest_level=5)
        non_terminal = [s for s in result.states if not AsymmetricCombat.is_terminal(s)]
        for s in non_terminal:
            a1 = result.state_nodes[s].a[0]
            self.assertGreater(a1, 0.05,
                               f"State {s} has boring A₁={a1}")

    def test_all_outcomes_asymmetric(self):
        """Every outcome should affect players differently."""
        config = AsymmetricCombat.Config(max_hp=5)
        init = AsymmetricCombat.initial_state(config)
        transitions = AsymmetricCombat.get_transitions(init, config)
        # At full HP with these outcomes, most transitions should lead to asymmetric states
        asymmetric_count = sum(1 for _, s in transitions if s[0] != s[1])
        self.assertGreater(asymmetric_count / len(transitions), 0.5)

    def test_superlinear_growth(self):
        """GDS growth should be superlinear with HP."""
        gds_5 = self._analyze(max_hp=5, nest_level=5).game_design_score
        gds_10 = self._analyze(max_hp=10, nest_level=5).game_design_score
        # If linear, gds_10 = 2 * gds_5. Superlinear means > 2x.
        ratio = gds_10 / gds_5
        self.assertGreater(ratio, 2.0,
                           f"Growth ratio {ratio:.2f} should be > 2.0 (superlinear)")

    def test_state_count(self):
        """State count should be (max_hp + 1)^2 - 1 or similar."""
        for hp in [3, 5, 7]:
            result = self._analyze(max_hp=hp, nest_level=3)
            # States include all (h1, h2) where h1, h2 in [0, max_hp]
            # minus (0, 0) if unreachable, plus terminal states
            self.assertGreater(len(result.states), hp * hp)


if __name__ == "__main__":
    unittest.main()
