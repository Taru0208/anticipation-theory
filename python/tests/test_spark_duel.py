"""Tests for SparkDuel v3 and v4 game models."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from toa.engine import analyze
from toa.games.spark_duel_v3 import SparkDuelV3
from toa.games.spark_duel_v4 import SparkDuelV4


class TestSparkDuelV3:
    """Tests for v3 symmetric sequential combat."""

    def test_initial_state(self):
        state = SparkDuelV3.initial_state()
        assert state == (7, 7, 0, 0, 0)

    def test_initial_state_custom_hp(self):
        cfg = SparkDuelV3.Config(max_hp=5)
        state = SparkDuelV3.initial_state(cfg)
        assert state == (5, 5, 0, 0, 0)

    def test_terminal_states(self):
        assert SparkDuelV3.is_terminal((0, 5, 0, 0, 0))
        assert SparkDuelV3.is_terminal((5, 0, 0, 0, 0))
        assert SparkDuelV3.is_terminal((0, 0, 0, 0, 0))
        assert not SparkDuelV3.is_terminal((3, 3, 0, 0, 0))

    def test_transitions_from_terminal(self):
        assert SparkDuelV3.get_transitions((0, 5, 0, 0, 0)) == []

    def test_transitions_probabilities_sum_to_one(self):
        cfg = SparkDuelV3.Config(max_hp=5)
        state = SparkDuelV3.initial_state(cfg)
        trans = SparkDuelV3.get_transitions(state, cfg)
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_phase0_applies_chip_damage(self):
        """Phase 0 should apply chip damage before attack resolution."""
        cfg = SparkDuelV3.Config(max_hp=5, chip_damage=1)
        state = (5, 5, 0, 0, 0)
        trans = SparkDuelV3.get_transitions(state, cfg)
        # All resulting HP values should be <= 4 (chip applied)
        for _, next_state in trans:
            hp1, hp2 = next_state[0], next_state[1]
            assert hp1 <= 4  # P1 took chip damage
            # hp2 could be lower from attacks too

    def test_phase1_no_chip(self):
        """Phase 1 should NOT apply chip damage (already done in phase 0)."""
        cfg = SparkDuelV3.Config(max_hp=5, chip_damage=1)
        state = (4, 4, 0, 0, 1)  # phase 1, already chipped
        trans = SparkDuelV3.get_transitions(state, cfg)
        # P1 (defender) HP should be <= 4, not 3 (no extra chip)
        for _, next_state in trans:
            hp2 = next_state[1]  # P2 is attacker in phase 1
            assert hp2 <= 4  # No extra chip to P2

    def test_cooldown_limits_blast(self):
        """Blast should be unavailable during cooldown."""
        attacks = SparkDuelV3._available_attacks(0)
        assert SparkDuelV3.BLAST in attacks
        attacks_cd = SparkDuelV3._available_attacks(1)
        assert SparkDuelV3.BLAST not in attacks_cd
        assert SparkDuelV3.ZAP in attacks_cd

    def test_symmetry(self):
        """D0 should be close to 0.5 for HP=5."""
        cfg = SparkDuelV3.Config(max_hp=5)
        initial = SparkDuelV3.initial_state(cfg)
        result = analyze(
            initial_state=initial,
            is_terminal=SparkDuelV3.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV3.get_transitions(s, cfg),
            compute_intrinsic_desire=SparkDuelV3.compute_intrinsic_desire,
            nest_level=3,
        )
        d0 = result.state_nodes[initial].d_global
        assert 0.48 < d0 < 0.55, f"D0={d0} too far from 0.5"

    def test_gds_reasonable(self):
        """GDS should be in a reasonable range."""
        cfg = SparkDuelV3.Config(max_hp=5)
        initial = SparkDuelV3.initial_state(cfg)
        result = analyze(
            initial_state=initial,
            is_terminal=SparkDuelV3.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV3.get_transitions(s, cfg),
            compute_intrinsic_desire=SparkDuelV3.compute_intrinsic_desire,
            nest_level=5,
        )
        gds = result.game_design_score
        assert 0.40 < gds < 0.70, f"GDS={gds} out of expected range"

    def test_chip_death_terminal(self):
        """If chip damage kills both at HP=1, game should end."""
        cfg = SparkDuelV3.Config(max_hp=5, chip_damage=1)
        state = (1, 1, 0, 0, 0)
        trans = SparkDuelV3.get_transitions(state, cfg)
        assert len(trans) == 1
        _, next_state = trans[0]
        assert SparkDuelV3.is_terminal(next_state)


class TestSparkDuelV4:
    """Tests for v4 with dodge counter-attack."""

    def test_initial_state(self):
        state = SparkDuelV4.initial_state()
        assert state == (7, 7, 0, 0, 0)

    def test_terminal_states(self):
        assert SparkDuelV4.is_terminal((0, 5, 0, 0, 0))
        assert not SparkDuelV4.is_terminal((3, 3, 0, 0, 0))

    def test_dodge_counter_damage(self):
        """Successful dodge should deal counter damage to attacker."""
        cfg = SparkDuelV4.Config(dodge_counter=1)
        # Blast vs Dodge: one outcome is (hit+dodge_success, 0 to def, 1 to atk)
        outcomes = SparkDuelV4._resolve_attack(SparkDuelV4.BLAST, SparkDuelV4.DODGE, cfg)
        # Find the dodge success outcome
        dodge_success = [o for o in outcomes if len(o) == 3 and o[2] > 0]
        assert len(dodge_success) >= 1, "Should have at least one counter-attack outcome"
        assert dodge_success[0][2] == 1, "Counter damage should be 1"

    def test_brace_no_counter(self):
        """Brace should never deal counter damage."""
        cfg = SparkDuelV4.Config(dodge_counter=2)
        outcomes_blast = SparkDuelV4._resolve_attack(SparkDuelV4.BLAST, SparkDuelV4.BRACE, cfg)
        outcomes_zap = SparkDuelV4._resolve_attack(SparkDuelV4.ZAP, SparkDuelV4.BRACE, cfg)
        for outcome in outcomes_blast + outcomes_zap:
            assert outcome[2] == 0, "Brace should never deal counter damage"

    def test_transitions_probabilities_sum_to_one(self):
        cfg = SparkDuelV4.Config(max_hp=5)
        state = SparkDuelV4.initial_state(cfg)
        trans = SparkDuelV4.get_transitions(state, cfg)
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-10

    def test_v4_more_states_than_v3(self):
        """v4 should have more states due to counter damage creating more HP combos."""
        cfg3 = SparkDuelV3.Config(max_hp=5)
        cfg4 = SparkDuelV4.Config(max_hp=5, dodge_counter=1)

        r3 = analyze(
            initial_state=SparkDuelV3.initial_state(cfg3),
            is_terminal=SparkDuelV3.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV3.get_transitions(s, cfg3),
            compute_intrinsic_desire=SparkDuelV3.compute_intrinsic_desire,
            nest_level=3,
        )
        r4 = analyze(
            initial_state=SparkDuelV4.initial_state(cfg4),
            is_terminal=SparkDuelV4.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV4.get_transitions(s, cfg4),
            compute_intrinsic_desire=SparkDuelV4.compute_intrinsic_desire,
            nest_level=3,
        )
        assert len(r4.state_nodes) >= len(r3.state_nodes)

    def test_recommended_config_gds(self):
        """Recommended config (HP=5, counter=1, dodge=30%) should have good GDS."""
        cfg = SparkDuelV4.Config(max_hp=5, dodge_counter=1, dodge_chance=0.30)
        initial = SparkDuelV4.initial_state(cfg)
        result = analyze(
            initial_state=initial,
            is_terminal=SparkDuelV4.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV4.get_transitions(s, cfg),
            compute_intrinsic_desire=SparkDuelV4.compute_intrinsic_desire,
            nest_level=5,
        )
        gds = result.game_design_score
        d0 = result.state_nodes[initial].d_global
        assert 0.45 < gds < 0.70, f"GDS={gds}"
        assert 0.48 < d0 < 0.55, f"D0={d0}"

    def test_zap_dodge_counter(self):
        """Zap vs Dodge should also produce counter damage."""
        cfg = SparkDuelV4.Config(dodge_counter=1)
        outcomes = SparkDuelV4._resolve_attack(SparkDuelV4.ZAP, SparkDuelV4.DODGE, cfg)
        counter_outcomes = [o for o in outcomes if o[2] > 0]
        assert len(counter_outcomes) == 1, "Zap dodged should counter"
        assert counter_outcomes[0][2] == 1

    def test_blast_miss_no_counter(self):
        """When Blast misses entirely, no counter damage (nothing to dodge)."""
        cfg = SparkDuelV4.Config(dodge_counter=1)
        outcomes = SparkDuelV4._resolve_attack(SparkDuelV4.BLAST, SparkDuelV4.DODGE, cfg)
        # Miss outcome: (1-hit_rate, 0, 0)
        miss = [o for o in outcomes if o[0] == 1.0 - cfg.blast_hit_rate]
        assert len(miss) == 1
        assert miss[0][1] == 0 and miss[0][2] == 0

    def test_zap2_blast4_config(self):
        """Zap=2, Blast=4, HP=7 config — the prototype configuration."""
        cfg = SparkDuelV4.Config(max_hp=7, zap_damage=2, blast_damage=4, dodge_counter=1, dodge_chance=0.30)
        initial = SparkDuelV4.initial_state(cfg)
        result = analyze(
            initial_state=initial,
            is_terminal=SparkDuelV4.is_terminal,
            get_transitions=lambda s, c=None: SparkDuelV4.get_transitions(s, cfg),
            compute_intrinsic_desire=SparkDuelV4.compute_intrinsic_desire,
            nest_level=5,
        )
        gds = result.game_design_score
        d0 = result.state_nodes[initial].d_global
        assert 0.40 < gds < 0.65, f"GDS={gds}"
        assert 0.50 < d0 < 0.60, f"D0={d0}"

    def test_zap2_brace_leaves_damage(self):
        """With Zap=2, Brace should reduce to 1 damage (not 0)."""
        cfg = SparkDuelV4.Config(zap_damage=2, brace_reduction=1)
        outcomes = SparkDuelV4._resolve_attack(SparkDuelV4.ZAP, SparkDuelV4.BRACE, cfg)
        assert len(outcomes) == 1
        assert outcomes[0][1] == 1, f"Zap(2) - Brace(1) should be 1, got {outcomes[0][1]}"
