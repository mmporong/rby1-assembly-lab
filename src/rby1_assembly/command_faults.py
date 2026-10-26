"""Explicit command-only negative tests for recorded single-environment runs."""

import math


SCENARIOS = ("right_gripper_open", "insertion_lateral_disturbance")


class CommandFaultExperiment:
    """Alter commands; leave physical states, sensors and evaluation untouched."""

    def __init__(self, env, scenario: str, parameters: dict, phase_codes: dict):
        if env.num_envs != 1 or scenario not in SCENARIOS:
            raise ValueError("command-fault demonstrations require one environment and a known scenario")
        required = (
            ("trigger_min_plug_height_m",) if scenario == "right_gripper_open" else
            ("trigger_insertion_depth_m", "max_position_error_command_m", "ramp_duration_s")
        )
        for key in required:
            value = parameters.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number")
        if scenario == "insertion_lateral_disturbance" and parameters["max_position_error_command_m"] > .008:
            raise ValueError("lateral position-error command is limited to 8 mm")
        self.env = env
        self.scenario = scenario
        self.parameters = dict(parameters)
        self.phase_codes = phase_codes
        self.events = []
        self.triggered = False
        self.active = False
        self.command_m = 0.0
        self._pulse_start_s = None
        self._original = {}

    def install(self):
        if self._original:
            raise RuntimeError("command fault is already installed")
        name = "_pre_physics_step" if self.scenario == "right_gripper_open" else "_desired_ee_twists"
        self._original[name] = getattr(self.env, name)
        setattr(self.env, name, self._pre_physics_step if name == "_pre_physics_step" else self._desired_ee_twists)

    def close(self):
        for name, method in self._original.items():
            setattr(self.env, name, method)
        self._original.clear()

    def _event(self, event: str, **values):
        self.events.append({
            "event": event,
            "issued_at_simulation_time_s": float(self.env._sim_time_s()),
            "control_step_to_be_executed": int(self.env.common_step_counter) + 1,
            **values,
        })

    def _pre_physics_step(self, actions):
        self._original["_pre_physics_step"](actions)
        phase = int(self.env._phase[0].item())
        height = float((self.env.plug.data.root_pos_w[0, 2] - self.env.scene.env_origins[0, 2]).item())
        if (
            not self.triggered
            and phase in (self.phase_codes["align"], self.phase_codes["insert"])
            and bool(self.env._grip_proven[0].all().item())
            and height >= self.parameters["trigger_min_plug_height_m"]
        ):
            self.triggered = self.active = True
            self._event("right_gripper_open_command_started", plug_height_m=height)
        if self.active:
            joint_ids = self.env._gripper_joint_ids[2:]
            magnitude = self.env.cfg.gripper_open_magnitude_m
            self.env._joint_targets[:, joint_ids[0]] = -magnitude
            self.env._joint_targets[:, joint_ids[1]] = magnitude

    def _desired_ee_twists(self):
        import torch

        left, right = self._original["_desired_ee_twists"]()
        phase = int(self.env._phase[0].item())
        if phase != self.phase_codes["insert"]:
            if self.active:
                self._event("lateral_command_stopped", phase_code=phase)
            self.active = False
            self.command_m = 0.0
            self._pulse_start_s = None
            return left, right
        relative_position, _ = self.env._relative_object_pose()
        depth = float(relative_position[0, 1].item())
        now = float(self.env._sim_time_s())
        if not self.active and depth >= self.parameters["trigger_insertion_depth_m"]:
            self.triggered = self.active = True
            self._pulse_start_s = now
            self._event("lateral_command_started", insertion_depth_m=depth)
        if self.active:
            assert self._pulse_start_s is not None
            fraction = min(1.0, (now - self._pulse_start_s + self.env.step_dt) / self.parameters["ramp_duration_s"])
            self.command_m = self.parameters["max_position_error_command_m"] * fraction
            # First rotation-matrix column: socket-local +X expressed in world.
            w, x, y, z = self.env.socket.data.root_quat_w.unbind(dim=1)
            direction = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)), dim=1)
            right = right.clone()
            right[:, :3] += self.command_m * direction
        return left, right

    def current_state(self):
        return {
            "scenario": self.scenario, "active": self.active,
            "lateral_position_error_command_m": self.command_m,
            "right_gripper_open_command": self.scenario == "right_gripper_open" and self.active,
        }

    def report(self):
        return {
            "scope": "scripted command-fault test, not a learned-policy failure or natural-failure frequency estimate",
            "scenario": self.scenario, "parameters": self.parameters,
            "triggered": self.triggered, "events": list(self.events),
            "modified_channel": "right gripper joint position target" if self.scenario == "right_gripper_open" else "right EE socket-local +X position-error command before DLS",
            "physical_state_force_sensor_or_termination_injection": False,
        }
