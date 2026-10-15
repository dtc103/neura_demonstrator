"""Robot-independent relative pose mapping and edge-triggered controls."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(position, quaternion):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    pose[:3, 3] = position
    if not np.isfinite(pose).all():
        raise ValueError("Non-finite pose")
    return pose


class Mapper:
    def __init__(self, basis, scale=.5, speed=.1, angular_speed=.5, radius=.35, smoothing=0., require_reset=False, translation_only=False, frame_alignment="world", precision_scale=.5):
        self.normal_scale = scale
        self.precision_scale = precision_scale
        self.precision = False
        self.rotation_scale = 1.
        self.frame_alignment = frame_alignment
        self.smoothing = smoothing
        self.require_reset = require_reset
        self.ready = not require_reset
        self.translation_only = translation_only
        self.reset_armed = False
        self.filtered = None
        self.basis = np.asarray(basis)
        self.scale, self.speed, self.angular_speed, self.radius = scale, speed, angular_speed, radius
        self.target = self.origin = None
        self.engaged = False
        self.armed = False
        self.trigger_armed = False
        self.epoch = None
        self.status = "Waiting for tracking"

    def suspend(self, measured):
        if self.engaged and measured is not None:
            self.target = measured.copy()
        self.engaged = self.armed = self.trigger_armed = False
        self.status = "Hold: release grip and trigger to re-arm"

    def update(self, frame, measured, dt):
        if self.target is None:
            self.target = measured.copy()
            self.origin = measured[:3, 3].copy()
        pose = None if frame is None else frame['controllers']['right']
        epoch = None if frame is None else (frame['source'], frame['epoch'])
        if epoch != self.epoch:
            self.suspend(measured)
            self.epoch = epoch
            self.reset_armed = False
            if self.require_reset:
                self.ready = False
        if pose is None or 'squeeze' not in pose:
            self.suspend(measured)
            reason = 'Quest frame missing/stale' if frame is None else ('Right controller not tracked' if pose is None else 'Grip data missing')
            self.status = reason + '; release grip and trigger to re-arm'
            return self.target.copy(), None
        button = pose.get('thumbstick', False)
        if not button:
            self.reset_armed = True
        elif self.reset_armed:
            self.suspend(measured)
            self.reset_armed = False
            self.target = measured.copy()
            self.origin = measured[:3, 3].copy()
            if self.frame_alignment == "tool":
                controller = pose_matrix(pose['position'], pose['orientation'])
                # Frozen world-to-base rotation: controller local axes at reset
                # become TCP local axes at reset. Never follow the live hand axes.
                self.basis = measured[:3, :3] @ controller[:3, :3].T
            self.ready = True
            self.status = 'Reference reset; release grip before re-engaging'
            return self.target.copy(), None
        if not self.ready:
            self.status = 'Waiting: release then click RIGHT thumbstick to set reference'
            return self.target.copy(), None
        precision = pose.get('secondary', False)
        if precision != self.precision:
            self.precision = precision
            self.scale = self.precision_scale if precision else self.normal_scale
            self.rotation_scale = self.precision_scale if precision else 1.
            if self.engaged:
                # Rebase both edges so press and release cannot jump the target.
                self.controller_ref = pose_matrix(pose['position'], pose['orientation'])
                self.robot_ref = self.target.copy()
                self.filtered = self.target.copy()
                self.rotation_controller = self.controller_ref[:3, :3].copy()
                self.rotation_target = self.robot_ref[:3, :3].copy()
        if pose['trigger'] < .2:
            self.trigger_armed = True
        gripper_click = None
        if pose['trigger'] > .8 and self.trigger_armed:
            gripper_click = True
            self.trigger_armed = False
        if pose['squeeze'] < .2:
            if self.engaged:
                # Cancel pending target catch-up instead of finishing a distant target.
                self.target = measured.copy()
            self.engaged = False
            self.armed = True
            self.status = 'Hold: grip to move'
        elif pose['squeeze'] > .8 and self.armed and not self.engaged:
            self.controller_ref = pose_matrix(pose['position'], pose['orientation'])
            self.robot_ref = measured.copy()
            self.filtered = self.robot_ref.copy()
            self.target = measured.copy()
            self.rotation_controller = self.controller_ref[:3, :3].copy()
            self.rotation_target = self.robot_ref[:3, :3].copy()
            self.engaged = True
            self.status = 'Clutched'
        if self.engaged:
            controller = pose_matrix(pose['position'], pose['orientation'])
            desired_p = self.robot_ref[:3, 3] + self.scale * self.basis @ (
                controller[:3, 3] - self.controller_ref[:3, 3])
            offset = desired_p - self.origin
            if np.linalg.norm(offset) > self.radius:
                self.suspend(measured)
                self.status = 'Workspace radius exceeded; release grip and re-engage'
                return self.target.copy(), None
            # Scale successive rotation increments, rather than an absolute
            # axis-angle offset that wraps at 180 degrees in precision mode.
            increment = self.basis @ controller[:3, :3] @ self.rotation_controller.T @ self.basis.T
            rv = Rotation.from_matrix(increment).as_rotvec()
            self.rotation_target = Rotation.from_rotvec(self.rotation_scale * rv).as_matrix() @ self.rotation_target
            self.rotation_controller = controller[:3, :3].copy()
            desired_r = self.rotation_target
            if self.translation_only:
                desired_r = self.robot_ref[:3, :3]
            dt = min(max(dt, 0), .05)  # Never catch up a paused process in one step.
            # Filter the requested pose, not the measured robot feedback. Held
            # targets remain constant; do not chase encoder noise when unclutched.
            alpha = 1. if self.smoothing <= 0 else -np.expm1(-dt / self.smoothing)
            self.filtered[:3, 3] += alpha * (desired_p - self.filtered[:3, 3])
            rv = Rotation.from_matrix(desired_r @ self.filtered[:3, :3].T).as_rotvec()
            self.filtered[:3, :3] = Rotation.from_rotvec(alpha * rv).as_matrix() @ self.filtered[:3, :3]
            desired_p, desired_r = self.filtered[:3, 3], self.filtered[:3, :3]
            delta = desired_p - self.target[:3, 3]
            distance = np.linalg.norm(delta)
            self.target[:3, 3] += delta * min(1., self.speed * dt / max(distance, 1e-12))
            rotvec = Rotation.from_matrix(desired_r @ self.target[:3, :3].T).as_rotvec()
            angle = np.linalg.norm(rotvec)
            self.target[:3, :3] = Rotation.from_rotvec(
                rotvec * min(1., self.angular_speed * self.rotation_scale * dt / max(angle, 1e-12))).as_matrix() @ self.target[:3, :3]
        return self.target.copy(), gripper_click
