#!/usr/bin/env python3
"""Interactive collision-threshold editor. No readback API exists in this driver."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

SERVICE = '/panda_param_service_server/set_full_collision_behavior'
FIELDS = {f'{level}_{kind}_thresholds_{phase}': size
          for phase in ('nominal', 'acceleration')
          for kind, size in (('torque', 7), ('force', 6))
          for level in ('lower', 'upper')}


def defaults():
    # Snapshot of installed franka_hardware/src/real/robot.cpp::setDefaultParams.
    # These are source defaults, NOT a readback of the active robot settings.
    return {field: ([20., 20., 18., 18., 16., 14., 12.] if size == 7 else
                    [20., 20., 20., 25., 25., 25.]) for field, size in FIELDS.items()}


def parse_values(text, size):
    values = [float(x) for x in text.strip().strip('[]').replace(',', ' ').split()]
    if len(values) != size:
        raise ValueError(f'Enter exactly {size} numbers, separated by spaces or commas')
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Values must be finite and strictly positive')
    return values


def validate(values):
    if set(values) != set(FIELDS):
        raise ValueError('Configuration must contain all eight threshold arrays')
    for field, size in FIELDS.items():
        array = values[field]
        if not isinstance(array, list) or len(array) != size or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in array):
            raise ValueError(f'{field}: expected {size} positive finite numbers')
    for phase in ('nominal', 'acceleration'):
        for kind in ('torque', 'force'):
            lower, upper = (values[f'{level}_{kind}_thresholds_{phase}'] for level in ('lower', 'upper'))
            if any(low > high for low, high in zip(lower, upper)):
                raise ValueError(f'{kind}/{phase}: every upper threshold must be >= its lower threshold')


def load_baseline(path, service, use_defaults=False):
    if use_defaults or not path.exists():
        return defaults(), 'source default (active value unknown)'
    data = json.loads(path.read_text())
    if data.get('service') != service:
        raise ValueError('Saved settings belong to another service; select another --state-file or use --defaults')
    values = data['values']
    validate(values)
    return values, f'last applied {data["applied_utc"]} (not live readback)'


def edit(values, label, read=input, write=print):
    proposed = {k: list(v) for k, v in values.items()}
    write('Enter keeps a field. Enter q to cancel. Arrays accept spaces or commas.')
    write('Torque: joints 1–7, all Nm. Force: Fx Fy Fz [N], Mx My Mz [Nm].')
    write('Lower = contact detection; upper = collision stop. Both motion phases are supplied.')
    while True:
        for field, size in FIELDS.items():
            while True:
                text = read(f'Change {field}\n  {label}: {proposed[field]}\n  New values: ').strip()
                if text.lower() in ('q', 'quit'):
                    return None
                if not text:
                    break
                try:
                    proposed[field] = parse_values(text, size)
                    break
                except ValueError as error:
                    write(str(error))
        try:
            validate(proposed)
            return proposed
        except ValueError as error:
            write(f'{error}. Review the fields again, or q to cancel.')
            label = 'proposed value'


def apply_ros(values, service, timeout):
    """Send one complete request after checking fresh, stationary robot state."""
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from franka_msgs.msg import FrankaState
        from franka_msgs.srv import SetFullCollisionBehavior
    except ImportError as error:
        raise RuntimeError('Run this script with pixi run -e jazzy python in pixi_panda_ros2') from error
    validate(values)
    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node('configure_collision_thresholds', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    state = None
    received = 0.
    steady_since = None

    def receive(message):
        nonlocal state, received, steady_since
        now = time.monotonic()
        previous = received
        state, received = message, now
        valid = (message.robot_mode in (1, 2) and len(message.dq) == 7
                 and all(math.isfinite(v) and abs(v) < .01 for v in message.dq)
                 and not any(getattr(message.current_errors, k) for k in message.current_errors.get_fields_and_field_types()))
        steady_since = (steady_since if steady_since is not None and now-previous < .25 else now) if valid else None

    try:
        node.create_subscription(FrankaState, '/franka_robot_state_broadcaster/robot_state', receive,
                                 qos_profile_sensor_data)
        client = node.create_client(SetFullCollisionBehavior, service)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'Service unavailable: {service}')
        deadline = time.monotonic()+timeout
        while True:
            executor.spin_once(timeout_sec=.05)
            now = time.monotonic()
            if state is not None and now-received < .25 and steady_since is not None and now-steady_since >= .5:
                break
            if now >= deadline:
                raise RuntimeError('No fresh, stationary, error-free robot state. Stop teleoperation/replay first.')
        request = SetFullCollisionBehavior.Request()
        for field, array in values.items():
            setattr(request, field, [float(v) for v in array])
        future = client.call_async(request)
        deadline = time.monotonic()+timeout
        while not future.done():
            executor.spin_once(timeout_sec=.05)
            if time.monotonic() >= deadline:
                raise RuntimeError('Service response timed out: application status UNKNOWN. No automatic retry; cache unchanged.')
        response = future.result()
        if not response.success:
            raise RuntimeError(f'Driver rejected settings: {response.error}')
    finally:
        executor.shutdown()
        node.destroy_node()
        context.shutdown()


def save_applied(path, values, service):
    data = {'service': service, 'applied_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'successful ROS service response; not live readback', 'values': values}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', default=SERVICE)
    parser.add_argument('--state-file', type=Path, default=Path(__file__).with_name('collision_last_applied.json'))
    parser.add_argument('--defaults', action='store_true', help='Start editing from the source defaults instead of the cache')
    parser.add_argument('--dry-run', action='store_true', help='Edit and display the request without connecting to ROS or saving it')
    parser.add_argument('--timeout', type=float, default=10.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    args.state_file = args.state_file.expanduser()
    try:
        baseline, label = load_baseline(args.state_file, args.service, args.defaults)
        print('This driver cannot report active thresholds. Displayed values are a starting point only.\n'
              'Robot/driver restarts or other programs may have changed them since the last application.\n'
              'Stop teleoperation/replay before applying. The Panda driver must remain available.')
        values = edit(baseline, label)
        if values is None:
            print('Cancelled; no settings sent.')
            return 0
        print(f'\nComplete request to {args.service}:\n{json.dumps(values, indent=2)}')
        if args.dry_run:
            print('Dry run: no ROS connection, no changes, no cache update.')
            return 0
        if input('Apply this complete request to the robot? Type APPLY: ').strip() != 'APPLY':
            print('Cancelled; no settings sent.')
            return 0
        apply_ros(values, args.service, args.timeout)
        print('Driver reported success. Collision thresholds applied.')
        try:
            save_applied(args.state_file, values, args.service)
            print(f'Last-applied settings saved to {args.state_file}')
        except OSError as error:
            print(f'Settings applied, but could not save the local record: {error}')
            return 1
        return 0
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. If interrupted while awaiting a service response, application status is unknown.')
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Could not complete configuration: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
