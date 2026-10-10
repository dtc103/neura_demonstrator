"""Report negotiated USB links and RGB profiles without starting camera streams."""

from importlib.metadata import version

import pyrealsense2 as rs


def main():
    print(f"RealSense SDK: {version('pyrealsense2')}", flush=True)
    try:
        context = rs.context()
        devices = list(context.query_devices())
        if not devices:
            raise RuntimeError("No RealSense cameras detected")
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"\nCamera {serial}: {device.get_info(rs.camera_info.name)}", flush=True)
            for field, label in ((rs.camera_info.usb_type_descriptor, "Negotiated USB"),
                                 (rs.camera_info.firmware_version, "Firmware"),
                                 (rs.camera_info.physical_port, "Physical port")):
                if device.supports(field):
                    print(f"  {label}: {device.get_info(field)}", flush=True)
            modes = set()
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        video = profile.as_video_stream_profile()
                        modes.add((video.width(), video.height(), video.fps(), str(video.format())))
            print("  Advertised color profiles:", flush=True)
            for width, height, fps, format_name in sorted(modes, reverse=True):
                print(f"    {width}x{height} @ {fps} FPS, {format_name}", flush=True)
            if not modes:
                print("    NONE — no color stream advertised on this connection", flush=True)
            pipeline = rs.pipeline(context)
            wrapper = rs.pipeline_wrapper(pipeline)
            for width, height, fps in ((1280, 720, 30), (1280, 720, 15), (640, 480, 30)):
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                try:
                    supported = config.can_resolve(wrapper)
                    print(f"  BGR8 request {width}x{height} @ {fps}: {'resolves' if supported else 'DOES NOT RESOLVE'}", flush=True)
                except RuntimeError as error:
                    print(f"  BGR8 request {width}x{height} @ {fps}: {error}", flush=True)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, rs.format.bgr8, 0)
            try:
                auto = config.resolve(wrapper).get_stream(rs.stream.color).as_video_stream_profile()
                print(f"  SDK automatic BGR8 selection: {auto.width()}x{auto.height()} @ {auto.fps()} FPS", flush=True)
            except RuntimeError as error:
                print(f"  SDK automatic BGR8 selection failed: {error}", flush=True)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
