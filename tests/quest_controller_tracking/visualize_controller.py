"""Render a cuboid that follows a Quest controller's orientation with Pygame."""

import argparse
from itertools import product
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from track_controllers import format_pose, load_reader

# Distinct dimensions make rotations easier to see. These are display units,
# not a physical controller model. Keep position fixed to isolate orientation.
VERTICES = np.array(list(product((-1., 1.), repeat=3))) * [1.1, .5, .32]
FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
FACE_COLORS = ((175, 65, 75), (240, 90, 100), (50, 145, 95),
               (75, 215, 140), (55, 100, 170), (85, 155, 245))
AXIS_COLORS = ((250, 105, 115), (100, 230, 160), (110, 175, 255))


def selected_hand(buttons, previous, current):
    """Select on rising button edges; a held button does not undo Tab switching."""
    for button, hand in (("X", "left"), ("A", "right")):
        if buttons.get(button, False) and not previous.get(button, False):
            current = hand
    return current


def camera_basis(frame):
    eye = np.array([3.5, -4.5, 3.0] if frame == "ros" else [3.5, 3.0, 4.5])
    up = np.array([0., 0., 1.] if frame == "ros" else [0., 1., 0.])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    return eye, np.column_stack((right, vertical, forward))


def project(points, size, frame="ros"):
    eye, basis = camera_basis(frame)
    camera = (np.asarray(points) - eye) @ basis
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Geometry lies behind the viewing camera")
    width, height = size
    focal = min(width, height) * 1.4
    pixels = camera[:, :2] * focal / camera[:, 2:3]
    pixels[:, 0] += width / 2
    pixels[:, 1] = height * .48 - pixels[:, 1]
    return pixels, camera[:, 2]


def valid_rotation(transform):
    if transform is None:
        return None
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=.05) or not np.isclose(np.linalg.det(rotation), 1, atol=.05):
        return None
    return rotation


def render_scene(pygame, screen, rotation, frame, font):
    size = screen.get_size()
    screen.fill((19, 24, 35))
    # Fixed tracking-frame grid, drawn behind the controller.
    up_axis = 2 if frame == "ros" else 1
    plane_axes = [axis for axis in range(3) if axis != up_axis]
    for axis in plane_axes:
        other = next(i for i in plane_axes if i != axis)
        for offset in np.arange(-3, 3.1, .5):
            line = np.zeros((2, 3))
            line[:, up_axis] = -.8
            line[:, axis] = [-3, 3]
            line[:, other] = offset
            points, _ = project(line, size, frame)
            pygame.draw.aaline(screen, (37, 46, 61), points[0], points[1])
    vertices = VERTICES @ rotation.T
    pixels, depths = project(vertices, size, frame)
    eye, basis = camera_basis(frame)
    light = np.array([-1., -2., 4.])
    light /= np.linalg.norm(light)
    ordered = sorted(range(len(FACES)), key=lambda i: np.mean(depths[list(FACES[i])]), reverse=True)
    for index in ordered:
        face = list(FACES[index])
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, eye - np.mean(vertices[face], axis=0)) <= 0:
            continue
        brightness = .45 + .55 * max(0, np.dot(normal, light))
        color = tuple(int(c * brightness) for c in FACE_COLORS[index])
        polygon = [tuple(p) for p in pixels[face]]
        pygame.draw.polygon(screen, color, polygon)
        pygame.draw.aalines(screen, (195, 210, 230), True, polygon)
    # Controller-local axes rotate with the cuboid.
    endpoints = np.vstack((np.zeros(3), rotation.T * 1.55))
    axes, _ = project(endpoints, size, frame)
    for axis, color in enumerate(AXIS_COLORS):
        pygame.draw.line(screen, color, axes[0], axes[axis + 1], 3)
        screen.blit(font.render("XYZ"[axis], True, color), axes[axis + 1] + [7, -7])
    # Small fixed frame reference, separate from the moving axes.
    origin = np.array([78., size[1] - 135.])
    for axis, color in enumerate(AXIS_COLORS):
        direction = basis[axis, :2] * [65, -65]
        end = origin + direction
        pygame.draw.line(screen, color, origin, end, 2)
        screen.blit(font.render("XYZ"[axis], True, color), end + [4, -7])
    screen.blit(font.render("Tracking frame", True, (155, 172, 194)), (18, size[1] - 100))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--frame", choices=("ros", "openxr"), default="ros")
    parser.add_argument("--ip", help="Quest IP for wireless ADB; omit for USB")
    parser.add_argument("--reader-source", type=Path)
    parser.add_argument("--fps", type=int, default=60, help="Display refresh rate, independent of tracking rate")
    parser.add_argument("--demo", action="store_true", help="Animate a synthetic orientation without accessing a headset")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        parser.error("Pygame is missing. Run `uv pip install --python .venv/bin/python pygame` in this test folder.")
    reader = None
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
        pygame.display.set_caption("Quest controller orientation")
        font = pygame.font.Font(None, 23)
        title = pygame.font.Font(None, 34)
        render_scene(pygame, screen, np.eye(3), args.frame, font)
        screen.blit(title.render("Connecting to Quest…" if not args.demo else "Demo mode", True, (230, 237, 248)), (24, 22))
        pygame.display.flip()
        if not args.demo:
            adb = shutil.which("adb")
            if adb is None:
                raise RuntimeError("ADB is missing; install it with `sudo apt install adb`.")
            subprocess.run([adb, "start-server"], check=True)
            reader = load_reader(args.reader_source)(ip_address=args.ip)
        clock = pygame.time.Clock()
        running, reference, latest = True, None, None
        previous_buttons = {}
        started = time.monotonic()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and latest is not None:
                        reference = latest.copy()
                    elif event.key == pygame.K_a:
                        reference = None
                    elif event.key == pygame.K_TAB:
                        args.hand = "left" if args.hand == "right" else "right"
                        reference = latest = None
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(640, event.w), max(480, event.h)), pygame.RESIZABLE)
            if not running:
                break
            trigger = 0.0
            if args.demo:
                from scipy.spatial.transform import Rotation
                t = time.monotonic() - started
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_euler("xyz", [.45 * math.sin(t), .3 * math.cos(t * .7), t * .4]).as_matrix()
            else:
                buttons = {name: reader.get_button_state(name) for name in ("X", "A")}
                hand = selected_hand(buttons, previous_buttons, args.hand)
                previous_buttons = buttons
                if hand != args.hand:
                    args.hand = hand
                    reference = latest = None
                getter = reader.get_hand_controller_transform_ros if args.frame == "ros" else reader.get_hand_controller_transform_openxr
                transform = getter(args.hand)
                trigger = reader.get_trigger_value(args.hand)
            latest = valid_rotation(transform)
            displayed = np.eye(3) if latest is None else latest if reference is None else reference.T @ latest
            render_scene(pygame, screen, displayed, args.frame, font)
            mode = "relative to zero reference" if reference is not None else f"{args.frame} tracking orientation"
            screen.blit(title.render(f"{args.hand.title()} controller  |  {mode}", True, (230, 237, 248)), (24, 22))
            if latest is None:
                message = "Waiting for valid tracking data…"
            else:
                displayed_pose = np.asarray(transform).copy()
                displayed_pose[:3, :3] = displayed
                message = format_pose(args.hand, displayed_pose, orientation=True)
            screen.blit(font.render(message, True, (177, 195, 218)), (24, 67))
            width, height = screen.get_size()
            trigger = min(1., max(0., trigger)) if math.isfinite(trigger) else 0.0
            pygame.draw.rect(screen, (39, 51, 68), (width - 240, height - 115, 210, 16), border_radius=5)
            pygame.draw.rect(screen, (90, 190, 235), (width - 240, height - 115, int(210 * trigger), 16), border_radius=5)
            screen.blit(font.render(f"Trigger {trigger:.3f}", True, (177, 195, 218)), (width - 240, height - 145))
            screen.blit(font.render("Controller X: left / A: right   Keyboard R: zero  A: tracking  Tab: switch  Esc: close", True, (177, 195, 218)), (24, height - 55))
            notice = "Demo: no headset connection" if args.demo else "Orientation only • position stays centred • input source/reference comes from the Quest app"
            screen.blit(font.render(notice, True, (126, 145, 170)), (24, height - 29))
            pygame.display.flip()
            clock.tick(args.fps)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, subprocess.CalledProcessError, pygame.error) as error:
        raise SystemExit(f"Viewer stopped: {error}") from error
    finally:
        if reader is not None:
            reader.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
