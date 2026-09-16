#!/usr/bin/env python3
"""
flat_world_calibrate.py
=======================
Interactive tool to calibrate the flat-world extrinsic parameters of a camera:
  • Pitch  — downward tilt of the camera from horizontal (degrees)
  • Roll   — rotation around the optical axis (degrees, 0 = level)
  • Yaw    — rotation about the vertical axis, i.e. camera heading error
             relative to the vehicle's forward direction (degrees)
  • Height — camera optical centre above the ground plane (metres, entered in config)
  • X/Y offsets — camera position relative to the vehicle reference point (config)

Physical setup required (see INSTRUCTIONS.html in this directory):
  1. Place a ground mark at a known distance D directly ahead of the camera
     along the vehicle centreline.
  2. Optionally: stretch a tape measure horizontally at the same distance D,
     perpendicular to the vehicle, for roll calibration.

Yaw is derived automatically from the same tape measure used for roll (its
endpoints sit at known positions since it's centred on the vehicle centreline
at distance D) — no extra clicks or physical setup beyond steps 1-2 above.
It becomes available once both pitch and roll have been calibrated.

Usage
-----
    python flat_world_calibrate.py
    python flat_world_calibrate.py --config calibration_config.yaml
    python flat_world_calibrate.py --image /path/to/frame.jpg
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "calibration_config.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_intrinsics(yaml_path: str, target_width: int = None, target_height: int = None):
    """
    Load camera intrinsics (K, D).

    If target_width/target_height are given and the yaml specifies
    image_width/image_height (the resolution the intrinsics were actually
    calibrated at), K is scaled to match — mirroring
    road_centerline_node.cpp's loadAndScaleK(). Without this, fx/fy/cx/cy
    are silently wrong whenever the calibration photo's resolution differs
    from the intrinsics' calibration resolution — and unlike a small
    intrinsics error, a resolution mismatch (e.g. 640x480 intrinsics used
    against a 1920x1440 photo) produces results that are off by tens of
    degrees, not fractions of one.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    elif "K" in data:
        K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
    else:
        raise ValueError(f"Cannot find camera matrix in {yaml_path}")

    if "distortion_coefficients" in data:
        D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
    elif "D" in data:
        D = np.array(data["D"], dtype=np.float64)
    else:
        D = np.zeros(5, dtype=np.float64)

    if target_width and target_height:
        native_w = data.get("image_width", target_width)
        native_h = data.get("image_height", target_height)
        if native_w != target_width or native_h != target_height:
            sx = target_width / native_w
            sy = target_height / native_h
            K[0, 0] *= sx   # fx
            K[0, 2] *= sx   # cx
            K[1, 1] *= sy   # fy
            K[1, 2] *= sy   # cy
            print(f"[INFO] Scaled intrinsics from {native_w}x{native_h} "
                  f"(calibration resolution) to {target_width}x{target_height} "
                  f"(calibration photo resolution)")
        elif "image_width" not in data or "image_height" not in data:
            print(f"[WARN] {yaml_path} has no image_width/image_height — "
                  f"cannot verify intrinsics match the photo's resolution "
                  f"({target_width}x{target_height}). If they were "
                  f"calibrated at a different resolution, results will be "
                  f"badly wrong with no warning.")

    return K, D


# ── Calibration math ───────────────────────────────────────────────────────────

def compute_pitch(v_click: float, cy: float, fy: float,
                  height_m: float, mark_dist_m: float) -> float:
    """
    Derive camera pitch from a ground mark clicked in the image.

    A pixel at row v has a ray that is atan((v-cy)/fy) radians below the
    optical axis.  A ground mark at distance D below a camera of height h
    subtends an elevation angle of atan(h/D) below horizontal.
    Combining these gives the camera pitch.

    Returns pitch in degrees (positive = camera tilted downward).
    """
    angle_from_axis = math.atan((v_click - cy) / fy)   # below optical axis
    angle_to_ground = math.atan(height_m / mark_dist_m) # below horizontal
    return math.degrees(angle_to_ground - angle_from_axis)


def compute_roll(u1: float, v1: float, u2: float, v2: float) -> float:
    """
    Derive camera roll from two clicked endpoints of a horizontal tape measure.
    Without roll, both ends project to the same image row.  Any v difference
    is due to roll.

    u1,v1 = left tape end;  u2,v2 = right tape end.
    Returns roll in degrees (positive = clockwise, right side tilted down).
    """
    if abs(u2 - u1) < 1.0:
        return 0.0
    return math.degrees(math.atan2(v2 - v1, u2 - u1))


def _build_rotation(pitch_deg: float, roll_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """
    Build the combined rotation matrix: vehicle frame → camera frame.

    Vehicle frame: X = forward, Y = left, Z = up
    Camera frame:  X = right,  Y = down, Z = forward (OpenCV convention)

    Yaw is applied first, to the vehicle-frame point — it represents camera
    mounting error about the vehicle's vertical axis (positive = camera
    turned toward the vehicle's left, matching the convention that positive
    steering curvature = left turn in road_centerline_node). The base
    rotation then aligns the frames, pitch is applied around camera X (tilts
    Z downward), and roll around camera Z (tilts X downward on the right).
    """
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    y = math.radians(yaw_deg)

    # Yaw around the vehicle's vertical (Z-up) axis, applied to the point
    # before anything else — see docstring above for the sign convention.
    Ryaw = np.array([
        [ math.cos(y), math.sin(y), 0],
        [-math.sin(y), math.cos(y), 0],
        [0,             0,           1],
    ], dtype=np.float64)

    # Base alignment: vehicle → camera (no pitch/roll/yaw)
    R_base = np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0],
    ], dtype=np.float64)

    # Pitch around camera X
    Rx = np.array([
        [1,           0,            0],
        [0,  math.cos(p), -math.sin(p)],
        [0,  math.sin(p),  math.cos(p)],
    ], dtype=np.float64)

    # Roll around camera Z
    Rz = np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r),  math.cos(r), 0],
        [0,            0,           1],
    ], dtype=np.float64)

    return Rz @ Rx @ R_base @ Ryaw


def project_ground_point(X_fwd: float, Y_left: float,
                          K: np.ndarray, height_m: float,
                          pitch_deg: float, roll_deg: float,
                          x_offset: float = 0.0, y_offset: float = 0.0,
                          yaw_deg: float = 0.0):
    """
    Project a vehicle-frame ground point (X_fwd, Y_left, Z=0) to image pixel (u, v).
    Returns (u, v) or None if the point is behind the camera.
    """
    R = _build_rotation(pitch_deg, roll_deg, yaw_deg)

    # Point relative to camera in vehicle frame: subtract camera position
    p_rel = np.array([X_fwd - x_offset, Y_left - y_offset, -height_m])

    p_cam = R @ p_rel
    if p_cam[2] <= 0.1:
        return None

    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return u, v


def compute_pose_joint(u_p: float, v_p: float, u_l: float, v_l: float,
                       u_r: float, v_r: float, X_fwd: float, half_width: float,
                       K: np.ndarray, height_m: float,
                       x_offset: float = 0.0, y_offset: float = 0.0,
                       pitch_init: float = 0.0, roll_init: float = 0.0,
                       yaw_init: float = 0.0, pitch_span: float = 6.0,
                       roll_span: float = 20.0, yaw_span: float = 20.0):
    """
    Jointly solve pitch, roll, and yaw from all three reference clicks (the
    ground mark plus both tape endpoints — 3 known vehicle-frame points, 6
    observed pixel coordinates), minimizing total squared reprojection error.

    Pitch, roll, and yaw are NOT independent — each reference point's
    projected position depends on all three together, so solving them one
    at a time via separate closed forms (pitch from the mark, then roll from
    the tape, then yaw from the tape given that roll) compounds error: pitch
    computed while ignoring unknown roll/yaw is already biased once either
    is non-negligible, and that bias then propagates into roll and yaw too.
    This instead does a dependency-free coarse-to-fine grid search over all
    three angles jointly — exact (up to grid resolution), no small-angle
    assumptions, seeded from the cheap closed-form estimates for speed.

    Returns (pitch_deg, roll_deg, yaw_deg).
    """
    points = [(X_fwd, 0.0,         u_p, v_p),
              (X_fwd,  half_width, u_l, v_l),
              (X_fwd, -half_width, u_r, v_r)]

    def residual(pitch_deg, roll_deg, yaw_deg):
        total = 0.0
        for X, Y, uc, vc in points:
            p = project_ground_point(X, Y, K, height_m, pitch_deg, roll_deg,
                                     x_offset, y_offset, yaw_deg)
            if p is None:
                return float("inf")
            total += (p[0] - uc) ** 2 + (p[1] - vc) ** 2
        return total

    pc, rc, yc = pitch_init, roll_init, yaw_init
    ps, rs, ys = pitch_span, roll_span, yaw_span
    for _ in range(10):
        best = (pc, rc, yc, residual(pc, rc, yc))
        for dp in np.linspace(-ps, ps, 7):
            for dr in np.linspace(-rs, rs, 7):
                for dy in np.linspace(-ys, ys, 7):
                    e = residual(pc + dp, rc + dr, yc + dy)
                    if e < best[3]:
                        best = (pc + dp, rc + dr, yc + dy, e)
        pc, rc, yc, _ = best
        ps, rs, ys = ps / 3.0, rs / 3.0, ys / 3.0
    return pc, rc, yc


def build_verification_grid(K, height_m, pitch_deg, roll_deg,
                              x_offset, y_offset, img_shape, yaw_deg=0.0):
    """
    Return a list of (u, v, dist_m, lat_m) for a ground grid.
    Used to visually verify the calibration.

    dist is distance from the CAMERA (matching mark_distance_m's documented
    meaning and how pitch/roll/yaw are actually solved in _refine_pose), so
    it must be converted to vehicle-frame X_fwd the same way: + x_offset.
    Without this, the grid's own "mark" point doesn't line up with the
    actual clicked mark whenever x_offset != 0.
    """
    h, w = img_shape[:2]
    pts = []
    for dist in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for lat in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            r = project_ground_point(x_offset + dist, lat, K, height_m, pitch_deg,
                                     roll_deg, x_offset, y_offset, yaw_deg)
            if r is None:
                continue
            u, v = r
            if 0 <= u < w and 0 <= v < h:
                pts.append((int(round(u)), int(round(v)), dist, lat))
    return pts


# ── Interactive tool ───────────────────────────────────────────────────────────

class CalibrationTool:
    MODE_PITCH  = "pitch"
    MODE_ROLL_L = "roll_left"
    MODE_ROLL_R = "roll_right"

    def __init__(self, image: np.ndarray, K: np.ndarray, D: np.ndarray, cfg: dict):
        self.orig    = image.copy()
        self.K       = K
        self.D       = D
        self.cfg     = cfg

        self.height_m    = float(cfg.get("camera_height_m",  1.0))
        self.mark_dist_m = float(cfg.get("mark_distance_m",  3.0))
        self.tape_width_m = float(cfg.get("tape_width_m",    0.0))
        self.x_offset    = float(cfg.get("camera_x_offset_m", 0.0))
        self.y_offset    = float(cfg.get("camera_y_offset_m", 0.0))
        self.output_path = cfg.get("output_yaml", "flat_world_extrinsic.yaml")

        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        # Status bar height (see _draw) — the displayed window is
        # vstack([bar, canvas]), so this offset must be subtracted from click
        # y-coordinates to convert them back into image (canvas) coordinates.
        self.bar_h = max(56, round(0.13 * self.orig.shape[0]))

        self.undistort   = False
        self.show_grid   = False
        self.mode        = self.MODE_PITCH

        self.pitch_click = None   # (u, v)
        self.roll_l      = None   # (u, v)
        self.roll_r      = None   # (u, v)

        self.pitch_deg   = None
        self.roll_deg    = 0.0
        self.yaw_deg     = 0.0

        self.win = "Flat-World Camera Calibration"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def _refine_pose(self):
        """
        Jointly refine pitch, roll, and yaw once the ground mark and both
        tape endpoints have all been clicked. See compute_pose_joint for why
        this joint solve is needed instead of the separate closed forms
        alone — pitch, roll, and yaw each bias the others' naive per-click
        estimates once any of them is non-negligible. The tape is centred on
        the vehicle centreline at the same distance as the pitch mark, so
        its endpoints sit at known vehicle-frame positions
        (mark_dist_m, +/-tape_width_m/2, 0) — no separate physical setup or
        click is needed beyond what pitch/roll calibration already collect.

        Overwrites self.pitch_deg/roll_deg (previously set from the cheap
        closed forms, which stay in effect until tape data is available)
        and sets self.yaw_deg.
        """
        if self.pitch_click is None or not (self.roll_l and self.roll_r):
            return
        if self.tape_width_m <= 0.0:
            print("  [INFO] tape_width_m is 0 in config — cannot refine "
                  "roll/yaw without a known tape width.")
            return

        half = self.tape_width_m / 2.0
        # The mark/tape are mark_dist_m FROM THE CAMERA, so their vehicle-frame
        # forward position also includes the camera's own offset from the
        # vehicle reference point.
        X_fwd = self.x_offset + self.mark_dist_m

        pitch_seed = compute_pitch(self.pitch_click[1], self.cy, self.fy,
                                   self.height_m, self.mark_dist_m)
        roll_seed  = compute_roll(*self.roll_l, *self.roll_r)

        self.pitch_deg, self.roll_deg, self.yaw_deg = compute_pose_joint(
            self.pitch_click[0], self.pitch_click[1],
            self.roll_l[0], self.roll_l[1],
            self.roll_r[0], self.roll_r[1],
            X_fwd, half, self.K, self.height_m,
            self.x_offset, self.y_offset,
            pitch_init=pitch_seed, roll_init=roll_seed, yaw_init=0.0)

        print(f"  Refined pose → pitch={self.pitch_deg:.3f}°  "
              f"roll={self.roll_deg:.3f}°  yaw={self.yaw_deg:.3f}°")

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # The window shows vstack([bar, canvas]), so clicks arrive in that
        # combined coordinate space — convert back to image (canvas) space.
        y -= self.bar_h
        if y < 0:
            return   # click landed on the status bar, not the image
        if self.mode == self.MODE_PITCH:
            self.pitch_click = (x, y)
            self.pitch_deg = compute_pitch(y, self.cy, self.fy,
                                           self.height_m, self.mark_dist_m)
            print(f"  Pitch mark clicked at ({x}, {y})  →  pitch = {self.pitch_deg:.3f} °")
            self._refine_pose()
        elif self.mode == self.MODE_ROLL_L:
            self.roll_l = (x, y)
            print(f"  Left tape end: ({x}, {y})")
            self.mode = self.MODE_ROLL_R   # auto-advance
            print("  Now click the RIGHT tape end")
        elif self.mode == self.MODE_ROLL_R:
            self.roll_r = (x, y)
            print(f"  Right tape end: ({x}, {y})")
            if self.roll_l:
                self.roll_deg = compute_roll(*self.roll_l, *self.roll_r)
                print(f"  Roll = {self.roll_deg:.3f} °")
            self.mode = self.MODE_PITCH
            self._refine_pose()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _get_base(self) -> np.ndarray:
        if self.undistort:
            return cv2.undistort(self.orig, self.K, self.D)
        return self.orig.copy()

    def _draw(self) -> np.ndarray:
        canvas = self._get_base()
        h, w   = canvas.shape[:2]
        s      = h / 480.0
        lw     = max(1, round(2 * s))
        cr     = max(4, round(8 * s))
        fs     = max(0.5, 0.7 * s)
        th     = max(1, round(fs * 1.8))

        # ── Verification grid ──────────────────────────────────────────────────
        if self.show_grid and self.pitch_deg is not None:
            grid = build_verification_grid(self.K, self.height_m, self.pitch_deg,
                                           self.roll_deg, self.x_offset, self.y_offset,
                                           canvas.shape, self.yaw_deg)
            for u, v, dist, lat in grid:
                is_mark = abs(dist - self.mark_dist_m) < 0.1 and abs(lat) < 0.1
                color = (0, 255, 160) if is_mark else (180, 180, 100)
                rad = max(2, round(3 * s)) if not is_mark else max(4, round(6 * s))
                cv2.circle(canvas, (u, v), rad, color, -1, cv2.LINE_AA)
                if abs(lat) < 0.01:  # centreline
                    cv2.putText(canvas, f"{dist:.0f}m", (u + 4, v - 4),
                                cv2.FONT_HERSHEY_PLAIN, fs * 0.9, color, 1, cv2.LINE_AA)

        # ── Pitch mark ────────────────────────────────────────────────────────
        if self.pitch_click:
            u, v = self.pitch_click
            cv2.circle(canvas, (u, v), cr, (0, 255, 100), lw + 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, (u, v), (0, 255, 100),
                           cv2.MARKER_CROSS, max(18, round(24 * s)), lw, cv2.LINE_AA)

        # ── Roll tape ─────────────────────────────────────────────────────────
        if self.roll_l:
            cv2.circle(canvas, self.roll_l, max(5, round(7 * s)),
                       (0, 200, 255), lw + 1, cv2.LINE_AA)
            cv2.putText(canvas, "L",
                        (self.roll_l[0] + max(6, round(8 * s)), self.roll_l[1]),
                        cv2.FONT_HERSHEY_PLAIN, fs * 1.1, (0, 200, 255), th)
        if self.roll_r:
            cv2.circle(canvas, self.roll_r, max(5, round(7 * s)),
                       (0, 200, 255), lw + 1, cv2.LINE_AA)
            cv2.putText(canvas, "R",
                        (self.roll_r[0] + max(6, round(8 * s)), self.roll_r[1]),
                        cv2.FONT_HERSHEY_PLAIN, fs * 1.1, (0, 200, 255), th)
        if self.roll_l and self.roll_r:
            cv2.line(canvas, self.roll_l, self.roll_r,
                     (0, 200, 255), lw, cv2.LINE_AA)

        # ── Status bar ────────────────────────────────────────────────────────
        bar_h = self.bar_h
        bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (25, 28, 38)

        pitch_str = (f"Pitch: {self.pitch_deg:+.3f} deg"
                     if self.pitch_deg is not None else "Pitch: (click the ground mark)")
        roll_str  = f"Roll:  {self.roll_deg:+.3f} deg     Yaw: {self.yaw_deg:+.3f} deg"

        mode_color = {
            self.MODE_PITCH:  (100, 255, 140),
            self.MODE_ROLL_L: (100, 220, 255),
            self.MODE_ROLL_R: (100, 220, 255),
        }[self.mode]
        mode_text = {
            self.MODE_PITCH:  "MODE: PITCH  — click the ground mark",
            self.MODE_ROLL_L: "MODE: ROLL   — click LEFT  tape end",
            self.MODE_ROLL_R: "MODE: ROLL   — click RIGHT tape end",
        }[self.mode]

        bs = max(0.55, 0.85 * s)
        bt = max(1, round(bs * 1.5))
        cv2.putText(bar, pitch_str, (8, round(bar_h * 0.38)),
                    cv2.FONT_HERSHEY_PLAIN, bs * 1.1, (180, 255, 190), bt)
        cv2.putText(bar, roll_str,  (8, round(bar_h * 0.72)),
                    cv2.FONT_HERSHEY_PLAIN, bs * 1.1, (180, 220, 255), bt)
        cv2.putText(bar, mode_text, (w // 3, round(bar_h * 0.55)),
                    cv2.FONT_HERSHEY_PLAIN, bs, mode_color, bt)

        hint = ("p = pitch mode    r = roll mode    g = toggle grid    "
                "u = undistort    s = save    q = quit")
        cv2.putText(bar, hint, (8, bar_h - 6),
                    cv2.FONT_HERSHEY_PLAIN, max(0.4, bs * 0.75), (90, 90, 90), 1)

        return np.vstack([bar, canvas])

    # ── Save ───────────────────────────────────────────────────────────────────

    def _save(self):
        if self.pitch_deg is None:
            print("[WARN] Pitch not yet calibrated — click the ground mark first.")
            return

        result = {
            "camera_height_m":   round(self.height_m,    4),
            "camera_pitch_deg":  round(self.pitch_deg,   4),
            "camera_roll_deg":   round(self.roll_deg,    4),
            "camera_yaw_deg":    round(self.yaw_deg,     4),
            "camera_x_offset_m": round(self.x_offset,   4),
            "camera_y_offset_m": round(self.y_offset,   4),
        }

        with open(self.output_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)

        print(f"\nCalibration saved → {self.output_path}")
        for k, v in result.items():
            print(f"  {k}: {v}")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("\nControls:")
        print("  p  — pitch mode (click ground mark)")
        print("  r  — roll mode  (click left then right tape end)")
        print("  g  — toggle verification grid overlay")
        print("  u  — toggle undistortion")
        print("  s  — save calibration")
        print("  q  — quit\n")
        print(f"Config:  height={self.height_m} m   mark_distance={self.mark_dist_m} m")
        print("Step 1: press 'p', then click the ground mark in the image.")

        while True:
            cv2.imshow(self.win, self._draw())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                self.mode = self.MODE_PITCH
                print("Mode → PITCH: click the ground mark")
            elif key == ord("r"):
                self.roll_l = None
                self.roll_r = None
                self.mode = self.MODE_ROLL_L
                print("Mode → ROLL: click the LEFT tape end")
            elif key == ord("g"):
                self.show_grid = not self.show_grid
                if self.show_grid and self.pitch_deg is None:
                    print("[INFO] Calibrate pitch first to see the grid.")
            elif key == ord("u"):
                self.undistort = not self.undistort
                print(f"Undistort: {'ON' if self.undistort else 'OFF'}")
            elif key == ord("s"):
                self._save()

        cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"Config YAML (default: {DEFAULT_CONFIG})")
    ap.add_argument("--image",  default=None,
                    help="Override image_source in config")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        print("        Edit calibration_config.yaml and try again.")
        sys.exit(1)

    cfg = load_config(args.config)

    image_path = args.image or cfg.get("image_source", "")
    if not image_path or not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path!r}")
        print("        Set image_source in the config or use --image /path/to/frame.jpg")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        sys.exit(1)

    print(f"Image:       {image_path}  ({image.shape[1]}×{image.shape[0]})")

    intrinsics_path = cfg.get("intrinsics_yaml", "")
    if not intrinsics_path or not os.path.exists(intrinsics_path):
        print(f"[ERROR] intrinsics_yaml not found: {intrinsics_path!r}")
        print("        Set intrinsics_yaml in calibration_config.yaml.")
        sys.exit(1)

    # Scale intrinsics to the calibration photo's resolution — see
    # load_intrinsics docstring for why this matters (a resolution mismatch
    # is far more damaging than an ordinary intrinsics error).
    K, D = load_intrinsics(intrinsics_path, image.shape[1], image.shape[0])
    print(f"Intrinsics:  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    CalibrationTool(image, K, D, cfg).run()


if __name__ == "__main__":
    main()
