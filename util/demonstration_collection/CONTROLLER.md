# Reusable Panda controller

`panda_controller.py` is the standalone control implementation to share in simulation. It contains `PandaCartesianController`, Pink kinematics, the bounded quintic planner, trajectory sampling and an explicit simulation PD torque helper. It imports no ROS, Quest, GUI or recording modules. Dependencies are NumPy, SciPy, Pinocchio, pin-pink, qpsolvers and quadprog.

`joint_teleop.py` only re-exports the old names for existing callers. Teleoperation now instantiates `PandaCartesianController`; collection uses that same instance and calls `reset()` after its external reset sequence. ROS message publication remains in `teleoperate.py`. Existing CLI commands and limits are unchanged.

## VLA / simulation example

```python
from pathlib import Path
from panda_controller import PandaCartesianController

controller = PandaCartesianController(
    Path('robot.urdf').read_text(),
    horizon=0.35, speed=2.0, acceleration=5.0, jerk=50.0,
)

# On EACH NEW policy action, exactly once:
# metres in base frame; rotation vector in base frame, radians; closed=0 or 1.
controller.set_delta_action([dx, dy, dz, rx, ry, rz, closed], measured_q)

# At the planner frequency (nominally 50 Hz), using simulation time in seconds:
segment = controller.step(measured_q, measured_dq, sim_time)

# At each actuator update, without replanning or reapplying the action:
tau = controller.compute_torques(
    measured_q, measured_dq, sim_time,
    effort_limits=joint_effort_limits,  # seven positive finite limits, N m
    feedforward=gravity_compensation,  # seven torques; omit if compensated elsewhere
)
# Apply tau through your simulator's effort interface.
# Send controller.gripper_closed to a separate gripper actuator adapter.
```

The adapter computes FK at action receipt, adds the translation delta, and left-multiplies the measured rotation by the delta rotation. The resulting absolute target is retained between actions. Delta values must already be decoded from the policy's normalization. Action frequency, action-chunk reference conventions and training-label convention must agree with this interface; a chunk encoded relative to one observation must not be interpreted as successive measured-relative actions without conversion.

For already processed Cartesian targets, use `set_target(T_base_tcp)` followed by `step(...)`. The existing `update(target, q, dq, enabled, time)` entry point supports teleoperation without applying its smoothing or scaling a second time. `enabled=False` generates a bounded stop; a subsequent enabled update resumes toward the target. Input freshness and episode lifecycle belong to the caller. Call `reset()` after an external reposition or simulation reset, then set a new target. Initial motion planning requires a stationary arm.

Use one persistent instance per robot. Planning preserves the current segment's commanded position, velocity and acceleration across replacements. Calls must use a nondecreasing clock; actuator samples must not precede the active segment. `sample(time)` returns desired `(q, dq, ddq)`; it never changes the target or replans. Beyond the horizon it holds the final reference with zero derivatives. This implementation is CPU-based, not a batched GPU Isaac Lab controller.

## ROS adapter versus simulation adapter

On the real robot, `trajectory_points(segment)` supplies eight time/position/velocity/acceleration samples to the existing ROS trajectory controller. Hardware torque control stays in ROS. The new Python torque helper is NOT called by teleoperation or collection.

In simulation, `compute_torques` evaluates:

```
clip(kp * (q_desired - q) + kd * (dq_desired - dq) + feedforward,
     -effort_limits, effort_limits)
```

Its default gains match the inspected collection configuration. Pass per-session gains explicitly when they differ. This helper models PD feedback, not all of Franka's firmware: actuator filtering, rate limits, transport delay, gravity compensation, friction and payload dynamics require simulator configuration/validation. Disable duplicate implicit position drives when applying explicit PD torques. No inertia or gravity computation is performed here. The URDF must contain the intended kinematic TCP; separately verify the dynamics of the simulated robot and attached payload.

ROS starts the published trajectory on receipt, whereas direct simulation sampling uses the planner clock. Do not assume timing equivalence: validate trajectory replacement, latency and measured response with a recorded episode. The caller controls the planner and actuator frequencies (nominal hardware configuration: 50 Hz planning, 1000 Hz joint control); this class does not run threads or timers.

## Files to share and validation

Share `panda_controller.py`, the session's `robot.urdf`, and `session.json`. Read trajectory limits from `settings` and ROS gains from `controller_parameters` (ROS ParameterValue dictionaries, e.g. `gains.panda_joint1.p.double_value`). The defaults do not replace those snapshots. Gripper control is an independent actuator adapter, not a second use of the arm's PD helper.

For validation also share `control.jsonl`, `sent_command.jsonl`, and `robot_state.jsonl`. Verify delta decoding reconstructs recorded targets, then compare commanded joint trajectories and simulated/measured joint responses. The helper does not guarantee identical contact dynamics or provide collision checking.

Offline regressions from the repository root:

```bash
PYTHONPATH=tests/quest_tracking_alternative python -m unittest discover \
  -s util/demonstration_collection -p 'test_*.py'
```
