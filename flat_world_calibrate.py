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
                  height_m: float, mark_dist_cam_m: float) -> float:
    """
    Derive camera pitch from a ground mark clicked in the image.

    A pixel at row v has a ray that is atan((v-cy)/fy) radians below the
    optical axis.  A ground mark at distance D below a camera of height h
    subtends an elevation angle of atan(h/D) below horizontal.
    Combining these gives the camera pitch.

    mark_dist_cam_m MUST be the distance from the CAMERA (the point directly
    below it) to the mark — this is a camera-relative viewing-angle formula,
    not a vehicle-frame one. If your mark_distance_m config value is
    measured from the vehicle reference point instead, convert it first:
    mark_dist_cam_m = mark_distance_m - camera_x_offset_m.

    Returns pitch in degrees (positive = camera tilted downward).
    """
    angle_from_axis = math.atan((v_click - cy) / fy)          # below optical axis
    angle_to_ground = math.atan(height_m / mark_dist_cam_m)   # below horizontal
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
                       u_r: float, v_r: float, X_mark: float, X_tape: float,
                       half_width: float, K: np.ndarray, height_m: float,
                       x_offset: float = 0.0, y_offset: float = 0.0,
                       pitch_init: float = 0.0, roll_init: float = 0.0,
                       yaw_init: float = 0.0, pitch_span: float = 45.0,
                       roll_span: float = 60.0, yaw_span: float = 60.0,
                       rounds: int = 13):
    """
    Jointly solve pitch, roll, and yaw from all three reference clicks (the
    ground mark plus both tape endpoints — 3 known vehicle-frame points, 6
    observed pixel coordinates), minimizing total squared reprojection error.

    X_mark and X_tape are independent — the tape does NOT need to be at the
    same distance as the mark, only centred on the vehicle centreline. Using
    a shared distance for both when they were placed differently makes the
    3 reference points geometrically inconsistent with any single rigid
    camera pose, no matter how well pitch/roll/yaw are searched — that
    showed up as a fit RMS in the hundreds of pixels in practice.

    Pitch, roll, and yaw are NOT independent — each reference point's
    projected position depends on all three together, so solving them one
    at a time via separate closed forms (pitch from the mark, then roll from
    the tape, then yaw from the tape given that roll) compounds error: pitch
    computed while ignoring unknown roll/yaw is already biased once either
    is non-negligible, and that bias then propagates into roll and yaw too.
    This instead does a dependency-free coarse-to-fine grid search over all
    three angles jointly — exact (up to grid resolution), no small-angle
    assumptions, seeded from the cheap closed-form estimates for speed.

    The default spans are wide (not just a small window around the seeds)
    because the seeds themselves can be quite far off before refinement —
    especially at low camera heights, where compute_pitch/compute_roll's
    small-angle-friendly assumptions break down more. A narrow search window
    around a bad seed converges to a wrong answer stuck at its own boundary
    rather than the true best fit; grid density and round count (not span
    width) drive runtime, so wide spans cost nothing extra.

    Returns (pitch_deg, roll_deg, yaw_deg, residual) — residual is the final
    total squared pixel reprojection error across all 3 points; near zero
    means a clean fit, and a large value flags either non-convergence or
    that the clicks/measurements are inconsistent with any single rigid pose.
    """
    points = [(X_mark, 0.0,         u_p, v_p),
              (X_tape,  half_width, u_l, v_l),
              (X_tape, -half_width, u_r, v_r)]

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
    best_res = residual(pc, rc, yc)
    for _ in range(rounds):
        best = (pc, rc, yc, best_res)
        for dp in np.linspace(-ps, ps, 7):
            for dr in np.linspace(-rs, rs, 7):
                for dy in np.linspace(-ys, ys, 7):
                    e = residual(pc + dp, rc + dr, yc + dy)
                    if e < best[3]:
                        best = (pc + dp, rc + dr, yc + dy, e)
        pc, rc, yc, best_res = best
        ps, rs, ys = ps / 3.0, rs / 3.0, ys / 3.0
    # Cast away from numpy.float64 (accumulated via np.linspace above) back
    # to plain Python float — otherwise yaml.dump serializes these as
    # unreadable Python-specific object tags instead of plain YAML numbers,
    # which the C++ yaml-cpp parser in road_centerline_node can't load at all.
    return float(pc), float(rc), float(yc), float(best_res)


def compute_pose_joint_with_focal_diagnostic(
        u_p, v_p, u_l, v_l, u_r, v_r, X_mark, X_tape, half_width,
        K, height_m, x_offset=0.0, y_offset=0.0,
        pitch_init=0.0, roll_init=0.0, yaw_init=0.0,
        pitch_span=45.0, roll_span=60.0, yaw_span=60.0,
        f_scale_range=0.08, rounds=11):
    """
    DIAGNOSTIC ONLY — not used for the saved calibration. Like
    compute_pose_joint, but additionally solves for a single scale factor
    applied to both fx and fy together (aspect ratio preserved, cx/cy left
    untouched), to check whether a small, plausible intrinsics error could
    explain a persistent fit residual.

    Deliberately restricted to ONE shared parameter rather than independent
    fx/fy/cx/cy: with only 3 reference points (6 scalar constraints), each
    extra free parameter eats into how overdetermined the problem is.
    Independently adjusting fx, fy, cx, and cy (4 more unknowns against 6
    constraints) would leave the problem barely constrained at all, and the
    solver could find a deceptively perfect-looking fit — a real
    focal-length/pose degeneracy, not evidence the intrinsics were actually
    wrong. A single bounded shared scale factor keeps the problem
    meaningfully overdetermined, but this is still a coarse heuristic, not a
    real recalibration.

    Returns (pitch_deg, roll_deg, yaw_deg, f_scale, residual).
    Compare f_scale to 1.0: something close (e.g. within 1-2%) with a
    substantially lower residual than compute_pose_joint suggests a small,
    plausible intrinsics correction. A large f_scale, or a residual that's
    still high even with this extra freedom, both point away from
    intrinsics and toward click precision or physical measurements instead.
    """
    points = [(X_mark, 0.0,         u_p, v_p),
              (X_tape,  half_width, u_l, v_l),
              (X_tape, -half_width, u_r, v_r)]

    def residual(pitch_deg, roll_deg, yaw_deg, f_scale):
        Ks = K.copy()
        Ks[0, 0] *= f_scale
        Ks[1, 1] *= f_scale
        total = 0.0
        for X, Y, uc, vc in points:
            p = project_ground_point(X, Y, Ks, height_m, pitch_deg, roll_deg,
                                     x_offset, y_offset, yaw_deg)
            if p is None:
                return float("inf")
            total += (p[0] - uc) ** 2 + (p[1] - vc) ** 2
        return total

    pc, rc, yc, fc = pitch_init, roll_init, yaw_init, 1.0
    ps, rs, ys, fs = pitch_span, roll_span, yaw_span, f_scale_range
    best_res = residual(pc, rc, yc, fc)
    for _ in range(rounds):
        best = (pc, rc, yc, fc, best_res)
        for dp in np.linspace(-ps, ps, 5):
            for dr in np.linspace(-rs, rs, 5):
                for dy in np.linspace(-ys, ys, 5):
                    for df in np.linspace(-fs, fs, 5):
                        e = residual(pc + dp, rc + dr, yc + dy, fc + df)
                        if e < best[4]:
                            best = (pc + dp, rc + dr, yc + dy, fc + df, e)
        pc, rc, yc, fc, best_res = best
        ps, rs, ys, fs = ps / 3.0, rs / 3.0, ys / 3.0, fs / 3.0
    return float(pc), float(rc), float(yc), float(fc), float(best_res)


def build_verification_grid(K, height_m, pitch_deg, roll_deg,
                              x_offset, y_offset, img_shape, yaw_deg=0.0):
    """
    Return a list of (u, v, dist_m, lat_m) for a ground grid.
    Used to visually verify the calibration.

    dist is distance from the VEHICLE REFERENCE POINT (matching
    mark_distance_m's meaning and how pitch/roll/yaw are actually solved in
    _refine_pose) — it IS the vehicle-frame X coordinate directly, no
    x_offset adjustment here (project_ground_point applies that internally).
    """
    h, w = img_shape[:2]
    pts = []
    for dist in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for lat in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            r = project_ground_point(dist, lat, K, height_m, pitch_deg,
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
        # The tape doesn't have to be at the same distance as the mark —
        # defaults to mark_dist_m for backward compatibility with configs/
        # setups that place them together, but set tape_distance_m
        # explicitly if the tape was laid elsewhere (very common in
        # practice — same distance is a convenience, not a requirement).
        self.tape_dist_m = float(cfg.get("tape_distance_m", self.mark_dist_m))
        self.tape_width_m = float(cfg.get("tape_width_m",    0.0))
        self.x_offset    = float(cfg.get("camera_x_offset_m", 0.0))
        self.y_offset    = float(cfg.get("camera_y_offset_m", 0.0))
        self.output_path = cfg.get("output_yaml", "flat_world_extrinsic.yaml")
        self.output_intrinsics_path = cfg.get("output_intrinsics_yaml",
                                              "adjusted_intrinsics.yaml")

        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        # Status bar height (see _draw) — the displayed window is
        # vstack([bar, canvas]), so this offset must be subtracted from click
        # y-coordinates to convert them back into image (canvas) coordinates.
        self.bar_h = max(56, round(0.13 * self.orig.shape[0]))

        # Default ON: every calibration formula (compute_pitch, compute_roll,
        # compute_pose_joint, project_ground_point) assumes a pure pinhole
        # model with no lens distortion. Clicking on the raw (distorted)
        # view silently biases every click, and that bias isn't a rigid
        # rotation — it can't be absorbed by the pitch/roll/yaw solve, so it
        # shows up as an unfixable-looking high fit-RMS residual. Toggling
        # 'u' OFF is only for visual comparison; re-enable it before clicking.
        self.undistort   = True
        self.show_grid   = False
        self.mode        = self.MODE_PITCH

        self.pitch_click = None   # (u, v)
        self.roll_l      = None   # (u, v)
        self.roll_r      = None   # (u, v)

        self.pitch_deg   = None
        self.roll_deg    = 0.0
        self.yaw_deg     = 0.0

        self.mouse_pos   = None   # (u, v) in canvas space, for the magnifier

        # Focal-length diagnostic result (see compute_pose_joint_with_focal_diagnostic)
        # — set only when the normal fit RMS is high enough to trigger it;
        # None otherwise, which also hides its overlay in _draw.
        self.diag_pitch  = None
        self.diag_roll   = None
        self.diag_yaw    = None
        self.diag_f_scale = None

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
        the vehicle centreline, at its own configured distance (which does
        NOT have to match the pitch mark's distance — see tape_distance_m),
        so its endpoints sit at known vehicle-frame positions
        (tape_dist_m, +/-tape_width_m/2, 0).

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
        # mark_dist_m/tape_dist_m are measured from the VEHICLE reference
        # point (easier to measure in practice than from directly below the
        # camera), so they ARE the vehicle-frame X coordinates directly —
        # project_ground_point subtracts x_offset internally to get the true
        # camera-relative position.
        X_mark = self.mark_dist_m
        X_tape = self.tape_dist_m

        # compute_pitch's formula is inherently camera-relative (it's a
        # viewing-angle relationship), so convert here.
        pitch_seed = compute_pitch(self.pitch_click[1], self.cy, self.fy,
                                   self.height_m, self.mark_dist_m - self.x_offset)
        roll_seed  = compute_roll(*self.roll_l, *self.roll_r)

        self.pitch_deg, self.roll_deg, self.yaw_deg, residual = compute_pose_joint(
            self.pitch_click[0], self.pitch_click[1],
            self.roll_l[0], self.roll_l[1],
            self.roll_r[0], self.roll_r[1],
            X_mark, X_tape, half, self.K, self.height_m,
            self.x_offset, self.y_offset,
            pitch_init=pitch_seed, roll_init=roll_seed, yaw_init=0.0)

        rms_px = (residual / 6.0) ** 0.5   # 6 = 3 points x (u,v)
        print(f"  Refined pose → pitch={self.pitch_deg:.3f}°  "
              f"roll={self.roll_deg:.3f}°  yaw={self.yaw_deg:.3f}°  "
              f"(fit RMS error: {rms_px:.2f} px)")
        self.diag_pitch = self.diag_roll = self.diag_yaw = self.diag_f_scale = None
        if rms_px > 5.0:
            print(f"  [WARN] Fit RMS error is {rms_px:.1f} px — this pose "
                  f"doesn't closely reproduce your clicks. Either re-click "
                  f"more carefully, or double-check camera_height_m / "
                  f"mark_distance_m / tape_distance_m / tape_width_m — no "
                  f"single rigid camera pose fits your clicks and "
                  f"measurements well together. In particular, if the tape "
                  f"wasn't laid at the same distance as the mark, make sure "
                  f"tape_distance_m in the config reflects where it "
                  f"actually is.")

            # DIAGNOSTIC ONLY (does not affect self.pitch_deg/roll_deg/yaw_deg
            # or the saved calibration) — see compute_pose_joint_with_focal_diagnostic
            # docstring for why this is deliberately restricted to a single
            # bounded shared focal-length scale factor rather than freely
            # adjusting fx/fy/cx/cy independently.
            diag_pitch, diag_roll, diag_yaw, f_scale, focal_res = \
                compute_pose_joint_with_focal_diagnostic(
                    self.pitch_click[0], self.pitch_click[1],
                    self.roll_l[0], self.roll_l[1],
                    self.roll_r[0], self.roll_r[1],
                    X_mark, X_tape, half, self.K, self.height_m,
                    self.x_offset, self.y_offset,
                    pitch_init=pitch_seed, roll_init=roll_seed, yaw_init=0.0)
            focal_rms = (focal_res / 6.0) ** 0.5
            print(f"  [DIAG] If intrinsics were also allowed a small, bounded "
                  f"nudge (focal scale factor, +/-8% max): fit RMS would be "
                  f"{focal_rms:.2f} px at scale={f_scale:.4f} "
                  f"({(f_scale - 1) * 100:+.1f}%). This is diagnostic only — "
                  f"it does NOT change the saved calibration. Shown on-canvas "
                  f"as white markers, vs. magenta for the normal fit.")
            if abs(f_scale - 1.0) < 0.02 and focal_rms > rms_px * 0.7:
                print(f"  [DIAG] Scale factor is small and barely helped — "
                      f"intrinsics are likely NOT the cause of the remaining "
                      f"error. Look at click precision or physical "
                      f"measurements instead.")
            self.diag_pitch, self.diag_roll, self.diag_yaw, self.diag_f_scale = \
                diag_pitch, diag_roll, diag_yaw, f_scale

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            my = y - self.bar_h
            self.mouse_pos = (x, my) if my >= 0 else None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.undistort:
            print("  [WARN] Undistort is OFF — this click is on the raw "
                  "(distorted) image and will bias the calibration. Press "
                  "'u' to re-enable undistort before clicking reference points.")
        # The window shows vstack([bar, canvas]), so clicks arrive in that
        # combined coordinate space — convert back to image (canvas) space.
        y -= self.bar_h
        if y < 0:
            return   # click landed on the status bar, not the image
        if self.mode == self.MODE_PITCH:
            self.pitch_click = (x, y)
            self.pitch_deg = compute_pitch(y, self.cy, self.fy,
                                           self.height_m, self.mark_dist_m - self.x_offset)
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

    def _draw_magnifier(self, canvas: np.ndarray, s: float):
        """
        Draw a zoomed-in inset of the region around the cursor into the
        top-right corner of canvas, in place, with a crosshair marking the
        exact pixel a click would land on. No-op if the cursor is off-image
        or the canvas is too small to fit an inset.
        """
        if self.mouse_pos is None:
            return
        h, w = canvas.shape[:2]
        mx, my = self.mouse_pos
        if not (0 <= mx < w and 0 <= my < h):
            return

        zoom = 8
        half_crop = max(10, round(15 * s))
        size = 2 * half_crop
        x0 = int(np.clip(mx - half_crop, 0, max(0, w - size)))
        y0 = int(np.clip(my - half_crop, 0, max(0, h - size)))
        crop = canvas[y0:y0 + size, x0:x0 + size]
        if crop.shape[0] != size or crop.shape[1] != size:
            return

        mag = cv2.resize(crop, (size * zoom, size * zoom),
                         interpolation=cv2.INTER_NEAREST)
        cx, cy = (mx - x0) * zoom, (my - y0) * zoom
        CYAN = (255, 255, 0)
        cv2.line(mag, (cx, 0), (cx, mag.shape[0]), CYAN, 1, cv2.LINE_AA)
        cv2.line(mag, (0, cy), (mag.shape[1], cy), CYAN, 1, cv2.LINE_AA)

        mh, mw = mag.shape[:2]
        pad = max(4, round(6 * s))
        px0, py0 = w - mw - pad, pad
        if px0 < 0 or py0 + mh > h:
            return   # canvas too small for the inset — skip rather than crash
        canvas[py0:py0 + mh, px0:px0 + mw] = mag
        cv2.rectangle(canvas, (px0 - 1, py0 - 1), (px0 + mw, py0 + mh),
                     (255, 255, 255), 2, cv2.LINE_AA)

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

        # ── Predicted tape reprojection ──────────────────────────────────────
        # Where the tape's two ends SHOULD be, given the current pitch/roll/
        # yaw and tape_distance_m/tape_width_m, for direct visual comparison
        # against the actual clicked L/R markers above (orange). A close
        # match confirms self-consistency; a visible gap points at click
        # imprecision or tape_distance_m/tape_width_m not quite matching
        # reality — the same thing the fit RMS number reports, but visual.
        if self.pitch_deg is not None and self.tape_width_m > 0:
            half = self.tape_width_m / 2.0
            pl = project_ground_point(self.tape_dist_m,  half, self.K, self.height_m,
                                      self.pitch_deg, self.roll_deg, self.x_offset,
                                      self.y_offset, self.yaw_deg)
            pr = project_ground_point(self.tape_dist_m, -half, self.K, self.height_m,
                                      self.pitch_deg, self.roll_deg, self.x_offset,
                                      self.y_offset, self.yaw_deg)
            MAGENTA = (255, 0, 255)
            if pl and pr:
                pl_i = (int(round(pl[0])), int(round(pl[1])))
                pr_i = (int(round(pr[0])), int(round(pr[1])))
                cv2.line(canvas, pl_i, pr_i, MAGENTA, lw, cv2.LINE_AA)
                for pt in (pl_i, pr_i):
                    cv2.drawMarker(canvas, pt, MAGENTA, cv2.MARKER_TILTED_CROSS,
                                   max(14, round(20 * s)), lw, cv2.LINE_AA)

        # ── Focal-length diagnostic reprojection ─────────────────────────────
        # Where mark + tape ends land under the DIAGNOSTIC pose (see
        # compute_pose_joint_with_focal_diagnostic) — same idea as the
        # magenta markers above, but using the focal-length-adjusted fit
        # instead of the normal one, so you can visually compare how much
        # (if any) a small intrinsics nudge changes the reprojection. Only
        # shown when that diagnostic actually ran (rms was > 5px).
        if self.diag_f_scale is not None:
            Ks = self.K.copy()
            Ks[0, 0] *= self.diag_f_scale
            Ks[1, 1] *= self.diag_f_scale
            WHITE = (255, 255, 255)

            dm = project_ground_point(self.mark_dist_m, 0.0, Ks, self.height_m,
                                      self.diag_pitch, self.diag_roll,
                                      self.x_offset, self.y_offset, self.diag_yaw)
            if dm:
                dm_i = (int(round(dm[0])), int(round(dm[1])))
                cv2.drawMarker(canvas, dm_i, WHITE, cv2.MARKER_DIAMOND,
                               max(16, round(22 * s)), lw, cv2.LINE_AA)

            if self.tape_width_m > 0:
                half = self.tape_width_m / 2.0
                dl = project_ground_point(self.tape_dist_m,  half, Ks, self.height_m,
                                          self.diag_pitch, self.diag_roll,
                                          self.x_offset, self.y_offset, self.diag_yaw)
                dr = project_ground_point(self.tape_dist_m, -half, Ks, self.height_m,
                                          self.diag_pitch, self.diag_roll,
                                          self.x_offset, self.y_offset, self.diag_yaw)
                if dl and dr:
                    dl_i = (int(round(dl[0])), int(round(dl[1])))
                    dr_i = (int(round(dr[0])), int(round(dr[1])))
                    cv2.line(canvas, dl_i, dr_i, WHITE, lw, cv2.LINE_AA)
                    for pt in (dl_i, dr_i):
                        cv2.drawMarker(canvas, pt, WHITE, cv2.MARKER_DIAMOND,
                                       max(14, round(20 * s)), lw, cv2.LINE_AA)

        # ── Magnifier ─────────────────────────────────────────────────────────
        # Zoomed inset around the cursor with a crosshair at the exact pixel a
        # click would land on — full-resolution images are hard to click
        # precisely on otherwise, and click precision is the main remaining
        # error source once the geometry/config issues are fixed.
        self._draw_magnifier(canvas, s)

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
                "u = undistort    s = save    i = save adj. intrinsics    q = quit")
        cv2.putText(bar, hint, (8, bar_h - 6),
                    cv2.FONT_HERSHEY_PLAIN, max(0.4, bs * 0.75), (90, 90, 90), 1)

        return np.vstack([bar, canvas])

    # ── Save ───────────────────────────────────────────────────────────────────

    def _save(self):
        if self.pitch_deg is None:
            print("[WARN] Pitch not yet calibrated — click the ground mark first.")
            return

        # float(...) guards against numpy scalar types leaking into the
        # output — yaml.dump serializes those as unreadable Python-specific
        # object tags instead of plain numbers (see compute_pose_joint).
        result = {
            "camera_height_m":   round(float(self.height_m), 4),
            "camera_pitch_deg":  round(float(self.pitch_deg), 4),
            "camera_roll_deg":   round(float(self.roll_deg),  4),
            "camera_yaw_deg":    round(float(self.yaw_deg),   4),
            "camera_x_offset_m": round(float(self.x_offset), 4),
            "camera_y_offset_m": round(float(self.y_offset), 4),
        }

        with open(self.output_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)

        print(f"\nCalibration saved → {self.output_path}")
        for k, v in result.items():
            print(f"  {k}: {v}")

    def _save_adjusted_intrinsics(self):
        """
        Save fx/fy scaled by the diagnostic focal-length factor (see
        compute_pose_joint_with_focal_diagnostic) to a new ROS camera_info
        format YAML — loadable directly as intrinsics_yaml here or in
        road_centerline_node. cx/cy and distortion are left unchanged;
        only fx/fy are scaled.

        This is a derived, 3-point-diagnostic-informed adjustment, not a
        fresh independent calibration — it's only as trustworthy as the 3
        reference points it came from. If you want real confidence in a
        corrected focal length, the more rigorous path is redoing the
        checkerboard calibration with better coverage.
        """
        if self.diag_f_scale is None:
            print("[WARN] No focal-length diagnostic available yet — click "
                  "the mark and both tape ends first (with a fit RMS > 5px, "
                  "since that's what triggers the diagnostic).")
            return

        h, w = self.orig.shape[:2]
        K_adj = self.K.copy()
        K_adj[0, 0] *= self.diag_f_scale
        K_adj[1, 1] *= self.diag_f_scale

        data = {
            "image_width":  int(w),
            "image_height": int(h),
            "camera_name":  "adjusted",
            "camera_matrix": {
                "rows": 3, "cols": 3,
                "data": [round(float(x), 4) for x in K_adj.flatten()],
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1, "cols": len(self.D),
                "data": [round(float(x), 6) for x in self.D],
            },
        }

        with open(self.output_intrinsics_path, "w") as f:
            f.write(f"# Adjusted intrinsics — fx/fy scaled by "
                    f"{self.diag_f_scale:.4f} ({(self.diag_f_scale-1)*100:+.1f}%) "
                    f"relative to the original intrinsics_yaml, found by "
                    f"fitting 3 calibration reference points (ground mark +\n"
                    f"# 2 tape ends). This is a diagnostic-derived adjustment,"
                    f" not an independent recalibration — treat it with "
                    f"appropriate caution, especially if the scale factor is\n"
                    f"# more than a percent or two from 1.0. cx/cy and "
                    f"distortion are unchanged from the original.\n")
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"\nAdjusted intrinsics saved → {self.output_intrinsics_path}")
        print(f"  fx: {self.fx:.2f} → {K_adj[0,0]:.2f}   "
              f"fy: {self.fy:.2f} → {K_adj[1,1]:.2f}   "
              f"(scale {self.diag_f_scale:.4f})")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("\nControls:")
        print("  p  — pitch mode (click ground mark)")
        print("  r  — roll mode  (click left then right tape end)")
        print("  g  — toggle verification grid overlay")
        print("  u  — toggle undistortion")
        print("  s  — save calibration")
        print("  i  — save focal-length-adjusted intrinsics (only available")
        print("       once the focal-length diagnostic has run — see below)")
        print("  q  — quit")
        print("  A magnified inset follows your cursor for precise clicking.")
        print("  Magenta X markers show where the tape ends SHOULD be given the")
        print("  current pitch/roll/yaw/tape_distance_m/tape_width_m — compare")
        print("  against the orange L/R markers (your actual clicks) once both")
        print("  are set. If the fit RMS is high, white diamond markers also")
        print("  appear, showing the same reprojection under a small diagnostic")
        print("  focal-length adjustment (does not affect the saved result).\n")
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
            elif key == ord("i"):
                self._save_adjusted_intrinsics()

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
    ap.add_argument("--ignore-distortion", action="store_true",
                    help="TEMPORARY TEST: zero out the intrinsics' distortion "
                         "coefficients instead of using them. Use this if the "
                         "calibration photo may already be rectified in-camera "
                         "(e.g. phone cameras often apply lens correction "
                         "before saving) — applying a second, unrelated "
                         "distortion correction on top of an already-corrected "
                         "photo would hurt accuracy rather than help it.")
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

    if args.ignore_distortion:
        print(f"[TEST] --ignore-distortion set — original D={D.tolist()} "
              f"replaced with zeros. Distortion correction is now a no-op; "
              f"compare fit RMS against a normal run to see if this photo "
              f"was already rectified in-camera.")
        D = np.zeros_like(D)

    CalibrationTool(image, K, D, cfg).run()


if __name__ == "__main__":
    main()
