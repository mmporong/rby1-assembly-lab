#!/usr/bin/env python3
"""Isaac Sim smoke test proving force transfer through the physical cable."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-utc")
    return parser.parse_args()


def _deadline_ok(value: str | None) -> bool:
    if not value:
        return True
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(timezone.utc) <= deadline.astimezone(timezone.utc)


def _run(args: argparse.Namespace, simulation_app) -> int:
    import numpy as np
    import torch
    import omni.usd
    import omni.physics.tensors as tensors
    from pxr import Gf, UsdGeom, UsdPhysics
    from isaaclab.sim import SimulationCfg, SimulationContext

    from rj45_physical_cable import (
        PhysicalCableConfig,
        assert_view_paths,
        author_physical_cable,
        straight_boundaries,
    )

    if not _deadline_ok(args.deadline_utc):
        raise RuntimeError("deadline UTC has already passed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    sim = SimulationContext(SimulationCfg(dt=1.0 / 240.0, device=args.device, gravity=(0.0, 0.0, 0.0)))
    handles = []
    for name, y in (("A", -0.5), ("B", 0.5)):
        plug_path = f"/World/Smoke/{name}/Plug"
        plug = UsdGeom.Cube.Define(stage, plug_path)
        plug.CreateSizeAttr(0.02)
        plug.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(plug.GetPrim())
        UsdPhysics.CollisionAPI.Apply(plug.GetPrim())
        UsdPhysics.MassAPI.Apply(plug.GetPrim()).CreateMassAttr(0.004)
        config = PhysicalCableConfig(prim_path=f"/World/Smoke/{name}/Cable", plug_prim_path=plug_path, table_top_z_m=-10.0)
        anchor = (0.0, y, 1.0)
        boundaries = straight_boundaries(anchor, (1.0, 0.0, 0.0), config)
        handles.append(
            author_physical_cable(
                stage, plug_path, anchor, (1.0, 0.0, 0.0), config, boundaries=boundaries
            )
        )
    sim.reset()
    view = tensors.create_simulation_view("torch")
    view.set_subspace_roots("/")
    view.reset()
    plug_paths = [h.plug_path for h in handles]
    tail_paths = [handles[1].link_paths[-1]]
    plugs = view.create_rigid_body_view(plug_paths)
    assert_view_paths(plugs, plug_paths)
    tail_b = view.create_rigid_body_view(tail_paths)
    assert_view_paths(tail_b, tail_paths)
    for handle in handles:
        handle.bind_simulation(simulation_view=view)
    initial = plugs.get_transforms().clone()
    records = []
    force = torch.tensor([[0.02, 0.0, 0.0]], device=args.device, dtype=torch.float32)
    indices = torch.tensor([0], device=args.device, dtype=torch.int32)
    for step in range(240):
        if step < 72:
            tail_b.apply_forces(force, indices, is_global=True)
        sim.step(render=False)
        if step % 12 == 0 or step == 239:
            xforms, velocities = plugs.get_transforms(), plugs.get_velocities()
            records.append({"step": step, "time_s": (step + 1) / 240.0,
                            "plug_x_m": xforms[:, 0].detach().cpu().tolist(),
                            "plug_vx_m_s": velocities[:, 0].detach().cpu().tolist()})
    final, final_velocity = plugs.get_transforms(), plugs.get_velocities()
    delta_x = float((final[1, 0] - initial[1, 0]) - (final[0, 0] - initial[0, 0]))
    delta_vx = float(final_velocity[1, 0] - final_velocity[0, 0])
    measurements = [handle.measure() for handle in handles]
    finite = bool(torch.isfinite(final).all() and torch.isfinite(final_velocity).all())
    anchor_error = max(m["max_joint_anchor_gap_m"] for m in measurements)
    passed = finite and delta_x > 0.001 and delta_vx > 0.01 and anchor_error < 0.005
    report = {"test": "physical_cable_force_coupling_not_task_success", "passed": passed,
              "physics_coupling_model_intent": True, "physics_coupling_verified_by_smoke": passed,
              "thresholds": {"plug_delta_x_m": 0.001, "plug_delta_vx_m_s": 0.01, "max_joint_anchor_error_m": 0.005},
              "results": {"plug_delta_x_m": delta_x, "plug_delta_vx_m_s": delta_vx,
                          "max_joint_anchor_error_m": anchor_error, "finite": finite},
              "force_probe": {"tail": "B", "global_force_n": [0.02, 0.0, 0.0], "duration_s": 0.3},
              "limitations": ["no real robot", "no training", "not an insertion-success measurement"]}
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


def main() -> int:
    args = _arguments()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "device": args.device})
    simulation_app = launcher.app
    try:
        result = _run(args, simulation_app)
    finally:
        simulation_app.close()
    return result


if __name__ == "__main__":
    sys.exit(main())
