import unittest

import numpy as np

from rby1_assembly.rj45_comparison_providers import (
    CLASSICAL_CANDIDATES,
    ClassicalForceSearchProvider,
    FixedConstantProvider,
    ImmutablePlaybackProvider,
    ZeroProvider,
    make_fair_provider,
)


class ComparisonProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = np.zeros(79, dtype=np.float64)
        self.obs[3] = 1.0

    def test_registry_excludes_oracle_and_hidden_input_mappings(self) -> None:
        with self.assertRaises(ValueError):
            make_fair_provider("oracle_gt_reference")
        provider = ZeroProvider()
        with self.assertRaises(TypeError):
            provider.act({"observation79": self.obs, "case_id": "tx-positive"}, provider.reset())

    def test_zero_and_constant_are_bounded_and_copy_outputs(self) -> None:
        zero = ZeroProvider()
        action, state = zero.act(self.obs, zero.reset())
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.ticks, 1)

        constant = FixedConstantProvider([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        first, state = constant.act(self.obs, constant.reset())
        first[0] = 1.0
        second, _ = constant.act(self.obs, state)
        self.assertAlmostEqual(second[0], 0.1)
        with self.assertRaises(ValueError):
            FixedConstantProvider([1.01, 0, 0, 0, 0, 0])

    def test_playback_is_immutable_and_holds_final_action(self) -> None:
        source = [[0.1, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]]
        provider = ImmutablePlaybackProvider(source)
        source[0][0] = 0.9
        state = provider.reset()
        values = []
        for _ in range(3):
            action, state = provider.act(self.obs, state)
            values.append(action[0])
        self.assertEqual(values, [0.1, 0.2, 0.2])

    def test_classical_uses_eight_newton_force_scale_and_opposes_lateral_force(self) -> None:
        provider = ClassicalForceSearchProvider(CLASSICAL_CANDIDATES["A"])
        obs = self.obs.copy()
        obs[37] = 1.0  # normalized actor force -> 8 N
        state = provider.reset()
        action, state = provider.act(obs, state)
        self.assertLess(action[0], 0.0)
        self.assertAlmostEqual(action[0], -0.1, places=7)  # 0.15 mm / 1.5 mm
        self.assertEqual(state.initial_direction_xz, (-1.0, 0.0))

    def test_classical_twenty_newton_stop_and_vector_saturation(self) -> None:
        provider = ClassicalForceSearchProvider(CLASSICAL_CANDIDATES["B"])
        stop = self.obs.copy()
        stop[37] = 2.5
        action, _ = provider.act(stop, provider.reset())
        np.testing.assert_array_equal(action, np.zeros(6))

        saturated = self.obs.copy()
        saturated[0] = saturated[2] = 100.0
        action, _ = provider.act(saturated, provider.reset())
        self.assertLessEqual(float(np.linalg.norm(action[:3])), 1.0 + 1e-12)
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_only_exact_sealed_candidates_are_accepted(self) -> None:
        from dataclasses import replace

        with self.assertRaises(ValueError):
            ClassicalForceSearchProvider(replace(CLASSICAL_CANDIDATES["A"], force_trigger_n=7.0))
        with self.assertRaises(ValueError):
            make_fair_provider("classical_force_search", classical_candidate_id="D")


if __name__ == "__main__":
    unittest.main()
