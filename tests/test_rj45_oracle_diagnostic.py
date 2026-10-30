from __future__ import annotations

import math
from pathlib import Path
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rby1_assembly.rj45_oracle_diagnostic import (  # noqa: E402
    ORACLE_ACTOR,
    STANDARD_ACTOR,
    build_oracle_state,
    compose_actor_critic_observations,
    make_rj45_oracle_diagnostic_env_type,
    observation_space_spec,
)


def _identity(count: int, *, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype).repeat(count, 1)


def _z_rotation(angle: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]],
        dtype=torch.float64,
    )


def _oracle_inputs(count: int = 1) -> dict[str, torch.Tensor]:
    zeros = torch.zeros((count, 3), dtype=torch.float64)
    return {
        "socket_position_world": zeros.clone(),
        "socket_quaternion_world_wxyz": _identity(count),
        "socket_linear_velocity_world": zeros.clone(),
        "socket_angular_velocity_world": zeros.clone(),
        "plug_position_world": zeros.clone(),
        "plug_quaternion_world_wxyz": _identity(count),
        "plug_linear_velocity_world": zeros.clone(),
        "plug_angular_velocity_world": zeros.clone(),
        "socket_entry_position_body": zeros.clone(),
        "socket_entry_quaternion_body_wxyz": _identity(count),
        "plug_tip_position_body": zeros.clone(),
        "plug_tip_quaternion_body_wxyz": _identity(count),
    }


class OracleStateMathTests(unittest.TestCase):
    def test_moving_socket_and_offset_tip_include_all_lever_arm_terms(self) -> None:
        values = _oracle_inputs()
        values["socket_entry_position_body"][0] = torch.tensor([1.0, 0.0, 0.0])
        values["plug_position_world"][0] = torch.tensor([2.0, 0.0, 0.0])
        values["plug_tip_position_body"][0] = torch.tensor([0.0, 1.0, 0.0])
        values["socket_angular_velocity_world"][0, 2] = 2.0
        values["plug_angular_velocity_world"][0, 2] = 3.0

        oracle = build_oracle_state(**values)

        torch.testing.assert_close(
            oracle[0, :3], torch.tensor([20.0, 20.0, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            oracle[0, 7:10], torch.tensor([-4.0, -16.0, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            oracle[0, 10:], torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
        )

    def test_pose_and_twist_are_expressed_in_rotated_entry_frame_wxyz(self) -> None:
        values = _oracle_inputs()
        values["socket_quaternion_world_wxyz"] = _z_rotation(torch.pi / 2.0)
        values["plug_position_world"][0] = torch.tensor([1.0, 0.0, 0.0])
        values["plug_linear_velocity_world"][0] = torch.tensor([1.0, 0.0, 0.0])

        oracle = build_oracle_state(**values)

        torch.testing.assert_close(
            oracle[0, :3], torch.tensor([0.0, -20.0, 0.0], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        expected_quaternion = torch.tensor(
            [2.0**-0.5, 0.0, 0.0, -(2.0**-0.5)], dtype=torch.float64
        )
        torch.testing.assert_close(oracle[0, 3:7], expected_quaternion)
        torch.testing.assert_close(
            oracle[0, 7:10], torch.tensor([0.0, -4.0, 0.0], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_rejects_nonfinite_wrong_shape_and_nonunit_quaternion(self) -> None:
        values = _oracle_inputs()
        values["plug_position_world"][0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "plug_position_world must be finite"):
            build_oracle_state(**values)

        values = _oracle_inputs()
        values["plug_tip_position_body"] = torch.zeros((1, 2), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "plug_tip_position_body must have shape"):
            build_oracle_state(**values)

        values = _oracle_inputs()
        values["socket_quaternion_world_wxyz"] *= 2.0
        with self.assertRaisesRegex(ValueError, "must be unit length"):
            build_oracle_state(**values)

    def test_rejects_invalid_scales_and_mixed_tensor_contracts(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(scale=invalid):
                values = _oracle_inputs()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    build_oracle_state(
                        **values,
                        scales={
                            "relative_position_m": invalid,
                            "relative_linear_velocity_m_s": 0.25,
                            "relative_angular_velocity_rad_s": 2.0,
                        },
                    )

        values = _oracle_inputs()
        values["plug_position_world"] = values["plug_position_world"].float()
        with self.assertRaisesRegex(TypeError, "one common dtype"):
            build_oracle_state(**values)

        values = _oracle_inputs()
        values["plug_position_world"] = torch.zeros((1, 3), dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "floating-point dtype"):
            build_oracle_state(**values)

        values = _oracle_inputs()
        values["plug_position_world"] = torch.zeros(
            (1, 3), dtype=torch.float64, device="meta"
        )
        with self.assertRaisesRegex(ValueError, "one common device"):
            build_oracle_state(**values)


class ObservationContractTests(unittest.TestCase):
    def test_standard_actor_has_no_oracle_fields_and_critic_is_identical(self) -> None:
        base = torch.arange(2 * 79, dtype=torch.float32).reshape(2, 79)
        oracle = torch.arange(2 * 13, dtype=torch.float32).reshape(2, 13) + 500.0
        valid = torch.tensor([True, False])

        standard = compose_actor_critic_observations(base, oracle, valid, STANDARD_ACTOR)
        privileged = compose_actor_critic_observations(base, oracle, valid, ORACLE_ACTOR)

        self.assertEqual(standard["policy"].shape, (2, 79))
        self.assertEqual(privileged["policy"].shape, (2, 92))
        torch.testing.assert_close(standard["policy"][0], base[0])
        torch.testing.assert_close(standard["policy"][1], torch.zeros(79))
        torch.testing.assert_close(standard["critic"], privileged["critic"])
        torch.testing.assert_close(privileged["policy"], privileged["critic"])
        torch.testing.assert_close(standard["critic"][1], torch.zeros(92))
        self.assertEqual(observation_space_spec(STANDARD_ACTOR), {"policy": 79, "critic": 92})
        self.assertEqual(observation_space_spec(ORACLE_ACTOR), {"policy": 92, "critic": 92})

    def test_observation_composition_rejects_shape_dtype_and_nan(self) -> None:
        base = torch.zeros((2, 79))
        oracle = torch.zeros((2, 13))
        valid = torch.ones(2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, r"shape \(num_envs, 79\)"):
            compose_actor_critic_observations(base[:, :-1], oracle, valid, STANDARD_ACTOR)
        with self.assertRaisesRegex(ValueError, "valid_mask must be boolean"):
            compose_actor_critic_observations(base, oracle, valid.float(), STANDARD_ACTOR)
        base[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compose_actor_critic_observations(base, oracle, valid, STANDARD_ACTOR)

    def test_observation_composition_rejects_dtype_and_device_mismatch(self) -> None:
        base = torch.zeros((2, 79), dtype=torch.float32)
        oracle = torch.zeros((2, 13), dtype=torch.float64)
        valid = torch.ones(2, dtype=torch.bool)
        with self.assertRaisesRegex(TypeError, "same dtype"):
            compose_actor_critic_observations(base, oracle, valid, STANDARD_ACTOR)

        with self.assertRaisesRegex(TypeError, "floating-point dtypes"):
            compose_actor_critic_observations(
                torch.zeros((2, 79), dtype=torch.int64),
                torch.zeros((2, 13), dtype=torch.int64),
                valid,
                STANDARD_ACTOR,
            )

        oracle_meta = torch.zeros((2, 13), dtype=torch.float32, device="meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            compose_actor_critic_observations(base, oracle_meta, valid, STANDARD_ACTOR)

        valid_meta = torch.ones(2, dtype=torch.bool, device="meta")
        with self.assertRaisesRegex(ValueError, "valid_mask must use"):
            compose_actor_critic_observations(
                base, torch.zeros((2, 13)), valid_meta, STANDARD_ACTOR
            )


class _DummyTrainingEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self._training_initialized = True
        identity = _identity(2, dtype=torch.float32)
        zeros = torch.zeros((2, 3), dtype=torch.float32)
        velocity = torch.zeros((2, 3), dtype=torch.float32)
        self.socket = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=zeros.clone(),
                root_quat_w=identity.clone(),
                root_link_lin_vel_w=velocity.clone(),
                root_link_ang_vel_w=velocity.clone(),
            )
        )
        self.plug = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=zeros.clone(),
                root_quat_w=identity.clone(),
                root_link_lin_vel_w=velocity.clone(),
                root_link_ang_vel_w=velocity.clone(),
            )
        )
        self._base_policy = torch.arange(2 * 79, dtype=torch.float32).reshape(2, 79)
        self.extras = {"policy_observation_valid": torch.tensor([True, False])}
        self.reset_calls = 0

    def _frame_tensors(self, name: str):
        if name not in {"socket_entry", "plug_tip"}:
            raise KeyError(name)
        return torch.zeros((2, 3)), _identity(2, dtype=torch.float32)

    def _get_observations(self):
        return {"policy": self._base_policy.clone()}

    def _reset_idx(self, env_ids):
        self.reset_calls += 1
        return env_ids.clone()


class OracleEnvironmentAdapterTests(unittest.TestCase):
    def test_factory_reads_state_without_mutating_physics_or_reset_contract(self) -> None:
        env_type = make_rj45_oracle_diagnostic_env_type(
            _DummyTrainingEnv, actor_mode=STANDARD_ACTOR
        )
        env = env_type()
        socket_before = env.socket.data.root_pos_w.clone()
        plug_before = env.plug.data.root_pos_w.clone()

        observations = env._get_observations()
        reset_result = env._reset_idx(torch.tensor([1]))

        self.assertEqual(observations["policy"].shape, (2, 79))
        self.assertEqual(observations["critic"].shape, (2, 92))
        torch.testing.assert_close(env.socket.data.root_pos_w, socket_before)
        torch.testing.assert_close(env.plug.data.root_pos_w, plug_before)
        torch.testing.assert_close(reset_result, torch.tensor([1]))
        self.assertEqual(env.reset_calls, 1)

    def test_factory_preserves_preinitialization_observation_hook(self) -> None:
        env_type = make_rj45_oracle_diagnostic_env_type(
            _DummyTrainingEnv, actor_mode=ORACLE_ACTOR
        )
        env = env_type()
        env._training_initialized = False

        observations = env._get_observations()

        self.assertEqual(set(observations), {"policy"})
        self.assertEqual(observations["policy"].shape, (2, 79))


if __name__ == "__main__":
    unittest.main()
