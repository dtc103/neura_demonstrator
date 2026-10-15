# Offline controller review — 2026-10-13

The supplied controller list has only Cartesian impedance commanding the arm.
The joint trajectory/impedance/gravity/torque-feedback controllers are inactive;
state/pose/twist broadcasters do not command arm torque.

## Confirmed in the local source/configuration

- Teleoperation uses a fixed Quest-world-to-base rotation by default; resetting
  references originally did not change it. New optional `--frame-alignment tool`
  captures the rotation between controller and TCP axes at thumbstick reset.
- Teleoperation target speed is capped at 0.1 m/s and 0.5 rad/s. Target filtering
  adds a 0.15 s first-order time constant. At a 0.5 translation ratio, a 0.4 m/s
  hand motion asks for 0.2 m/s, twice the target cap. Thus lag can accumulate
  even if robot tracking were perfect. At 0.5 rad/s, a 90-degree target rotation
  requires at least 3.14 s, excluding acceleration/filter/tracking delay.
- The installed bringup YAML configures Cartesian stiffness 400 N/m, rotational
  stiffness 30 Nm/rad, nullspace stiffness 2, joint position/velocity and output
  torque filter parameters 0.5. Verify actual live parameters before changing
  anything: runtime changes may differ from these files.
- CRISP computes Cartesian torques as J.T * (K*error - D*J*dq), plus projected
  nullspace, Coriolis, and joint-limit repulsion terms. With no explicit task
  damping override it uses 2*sqrt(K). This formula does not by itself establish
  critical damping for the full configuration-dependent robot dynamics.
- The EMA helper is (1-alpha)*current + alpha*previous. q/dq filters call it
  in that order. The output torque call swaps the arguments, so its parameter
  has the opposite direction. At the configured 0.5 the numerical result is
  identical; this observation is NOT an explanation of the current failure.
- Hardware torque control runs in another thread, exchanging the latest state
  and torque through mutexes with ros2_control. The callback calls libfranka
  with rate limiting enabled and kMaxCutoffFrequency. Do not assume a default
  100 Hz libfranka torque filter is applied by this wrapper.
- on_activate initializes pose and nullspace references from current measurements.
  The source inspected does not reset tau_previous in on_activate (it initializes
  it during configuration). Reactivation merits investigation, but this does not
  prove the cause of oscillation growing long after activation.

Sources read: installed franka_bringup/config/controllers.yaml; CRISP
src/cartesian_controller.cpp, src/cartesian_controller.yaml,
include/crisp_controllers/utils/fiters.hpp; franka_hardware/src/real/robot.cpp,
under /home/jan/install/pixi_panda_ros2.

## Evidence and limits

The RT run successfully enabled FIFO scheduling. Logged oscillation still ends in
joint_velocity_violation. A fixed final target continued to produce growing wrist
joint motion. Posture-dependent dynamics, feedback delay, gains, filtering, model
configuration, and torque-loop timing require investigation. A successful run with
rough hand movement does not establish stability throughout the workspace.

No live gains were changed, no installed driver/controller sources were modified,
and no hardware motion tests were performed. Mapping unit tests only establish the
coordinate math. The next useful read-only capture is a live parameter dump of the
Cartesian controller, plus the hardware/model configuration. A subsequent controlled
validation should use a known baseline controller and stationary targets before
reintroducing Quest motion. Do not loosen hardware velocity protections.

## Runtime dump follow-up

Read /home/jan/robot_logs/cartesian_controller_parameters.yaml. Confirms 400/30
stiffness, automatic task/nullspace damping (-1 sentinel), filter q/dq/output=.5,
local Jacobian, Coriolis enabled, gravity compensation disabled, operational-space
mode disabled, 1000 Hz update. Fresh-launch status correctly has JTC active and
Cartesian inactive. controller_manager dump is empty and provides no rate evidence.

Additional concrete model issue: saved captures/automatic/panda.urdf has no
inertial elements. The currently installed checkout's franka_description/robots
Xacro sources also have no inertial/mass entries. Building the saved URDF with
Pinocchio and reducing the finger joints exactly as CRISP does gives ZERO arm
masses, zero mass-matrix norm, and zero Coriolis torque at the final recorded
configuration. Thus the enabled URDF-based Coriolis computation cannot compensate
actual robot Coriolis effects with this model. This is a confirmed limitation of
the inspected model, NOT proof that it caused the observed instability. Need the
live robot_description to conclusively establish the runtime model. Do not enable
dynamic nullspace projection or operational-space inertia compensation on this
massless model, and do not invent inertia values. A robot dynamic model should
come from verified inertial data or Franka's model interface.

Attempted a posture-dependent M^-1 D damping analysis; it could not be performed
because this URDF mass matrix is singular. No gain recommendations were derived
from that invalid model. -1 damping is a sentinel: effective Cartesian damping
is 40 translational and approximately 10.954 rotational; nullspace damping is
approximately 2.828 at unit joint weights. These are calculations of current
behavior, not proposed tuned values.
