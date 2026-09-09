"""ZED Mini capture, drop-in compatible with RSCapture.

Franka pivot (2026-09-08): the wrist camera is a ZED Mini, not a RealSense. This mirrors
``RSCapture``'s tiny surface used by the env — ``read() -> (ok, bgr_image)`` and
``close()`` — so ``FrankaEnv`` needs no camera-logic change; it just constructs this
instead of ``RSCapture`` when ``config.CAMERA_TYPE == "zed"`` (see init_cameras).

Returns a BGR uint8 image the same way ``RSCapture.read()`` does (RealSense delivers
``rs.format.bgr8``; the ZED SDK delivers BGRA, so we drop the alpha channel). The env's
``get_im()`` then crops, resizes, and flips to RGB exactly as before.

The ``pyzed`` SDK is NOT installed at home, so the import is guarded and deferred to
__init__: this module imports fine offline; constructing ZEDCapture without pyzed raises
a clear error (only happens on the real rig, where pyzed is present).
"""

import numpy as np

try:  # import-guard: pyzed only exists on the robot PC
    import pyzed.sl as sl  # noqa: F401
    _PYZED_AVAILABLE = True
    _PYZED_IMPORT_ERROR = None
except Exception as e:  # ImportError on home box; keep module importable
    sl = None
    _PYZED_AVAILABLE = False
    _PYZED_IMPORT_ERROR = e


class ZEDCapture:
    """Minimal RSCapture-compatible wrapper around a single ZED (Mini) camera.

    Only the left (RGB) eye is used — HIL-SERL is RGB-only. Signature mirrors RSCapture:
    ``name`` + ``serial_number`` (kept for API parity; the env passes it from
    REALSENSE_CAMERAS-style config), plus dim/fps.
    """

    def __init__(self, name, serial_number=None, dim=(640, 480), fps=15, depth=False, exposure=None):
        if not _PYZED_AVAILABLE:
            raise ImportError(
                "pyzed (ZED SDK) is not installed — ZEDCapture can only run on the "
                f"robot PC. Original import error: {_PYZED_IMPORT_ERROR!r}"
            )
        self.name = name
        self.serial_number = serial_number
        self.depth = depth  # kept for parity; HIL-SERL uses RGB only

        self.cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720  # closest ZED-Mini res; we resize
        init.camera_fps = int(fps)
        init.depth_mode = sl.DEPTH_MODE.NONE  # RGB only
        init.coordinate_units = sl.UNIT.METER
        # TODO(rig): if multiple ZEDs are ever present, select by serial:
        #   if serial_number is not None:
        #       init.set_from_serial_number(int(serial_number))

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        # Optional manual exposure to match RSCapture's fixed-exposure behavior.
        if exposure is not None:
            # ZED exposure is 0-100 (%). RSCapture uses microseconds; not comparable,
            # so treat `exposure` here as the ZED percentage if a caller passes one.
            self.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(exposure))

        self._runtime = sl.RuntimeParameters()
        self._mat = sl.Mat()
        self._dim = dim  # (w, h) target; env resizes anyway, but keep for parity

    def read(self):
        """Return (ok, bgr_image) like RSCapture.read()."""
        if self.cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return False, None
        self.cam.retrieve_image(self._mat, sl.VIEW.LEFT)  # left eye, BGRA
        frame = self._mat.get_data()  # HxWx4 (BGRA), uint8
        bgr = np.ascontiguousarray(frame[:, :, :3])  # drop alpha -> BGR, matches RSCapture
        return True, bgr

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass
