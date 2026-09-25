"""Whole-body 3D inverse dynamics of a pitch, closed at the trunk.

Arms and head are solved top-down from their free ends (the ball for the throwing hand).
Legs are solved bottom-up from the ground reaction: each force plate's load goes to the
foot that is on it. The two directions meet at the trunk, whose Newton-Euler residual is
the error of the whole model.

Building blocks (segments, the Newton-Euler step, filtering, force plates) are in
dynamics.py. Everything is lab frame: Z up, X anterior, Y mediolateral; N, N.m, m.
A joint's force/moment is what the PROXIMAL segment applies to the DISTAL one.

Run:  python inverse_dynamics.py <subject_id> [trial_name]
"""

import sys

import numpy

import analysis as A
import dynamics as D
from c3d_to_hdf5 import load_trial


def foot_plate_loads(trial: dict, ts, wrenches: dict, min_force_n: float = 20.0,
                     tolerance_m: float = 0.03, max_height_m: float = 0.15) -> tuple:
    """Give each plate's load to the foot that is on it.

    A plate counts as loaded when its vertical force exceeds `min_force_n`. The foot it
    loads is the one whose heel or toe marker lies over the plate's footprint (grown by
    `tolerance_m`) and within `max_height_m` of its surface -- a lifted foot swinging
    over a plate does not load it. If both feet are over the plate, the one nearer the center of
    pressure gets it and the frame is flagged ambiguous; if neither is, the foot nearest
    the CoP gets it and the frame is flagged unmatched.

    Returns ({"L": [load, ...], "R": [...]}, info). Each load is (force, moment, point)
    with zeros in frames where that foot is not on that plate. info holds the
    (nPlates, n) arrays "owner" (0 = L, 1 = R, -1 = none), "ambiguous" and "unmatched".
    """
    corners = trial["force_platforms"]["corners"]
    force, moment = wrenches["force"], wrenches["moment"]
    center, cop = wrenches["center"], wrenches["cop"]
    n_plates, n = force.shape[0], force.shape[1]
    markers = {side: numpy.stack([A.marker_point(ts, f"{side}{m}") for m in ("HEE", "TOE")]) for side in "LR"}
    feet = {side: markers[side][:, :, :2] for side in "LR"}

    owner = numpy.full((n_plates, n), -1)
    ambiguous = numpy.zeros((n_plates, n), bool)
    unmatched = numpy.zeros((n_plates, n), bool)
    for p in range(n_plates):
        lo = corners[p][:, :2].min(axis=0) - tolerance_m
        hi = corners[p][:, :2].max(axis=0) + tolerance_m
        surface_z = corners[p][:, 2].mean()
        over = {side: (((feet[side] >= lo) & (feet[side] <= hi)).all(axis=2)
                       & (markers[side][:, :, 2] - surface_z < max_height_m)).any(axis=0) for side in "LR"}
        near = {side: numpy.linalg.norm(feet[side] - cop[p][None], axis=2).min(axis=0) for side in "LR"}
        for k in numpy.flatnonzero(force[p, :, 2] > min_force_n):
            candidates = [i for i, side in enumerate("LR") if over[side][k]]
            if len(candidates) == 1:
                owner[p, k] = candidates[0]
            else:
                owner[p, k] = 0 if near["L"][k] <= near["R"][k] else 1
                ambiguous[p, k] = len(candidates) == 2
                unmatched[p, k] = len(candidates) == 0

    loads = {"L": [], "R": []}
    for i, side in enumerate("LR"):
        for p in range(n_plates):
            mine = (owner[p] == i)[:, None]
            if mine.any():
                loads[side].append((force[p] * mine, moment[p] * mine, numpy.broadcast_to(center[p], (n, 3))))
    return loads, {"owner": owner, "ambiguous": ambiguous, "unmatched": unmatched}


def fingertip(ts, side: str, tip_from_wrist: float = D.FINGERTIP_FROM_WRIST_M) -> numpy.ndarray:
    """(n,3) fingertip / ball position: the wrist center plus `tip_from_wrist` along the
    wrist -> finger-marker direction (there is no fingertip marker).
    """
    wrist = (A.marker_point(ts, f"{side}WRA") + A.marker_point(ts, f"{side}WRB")) / 2
    direction = A.marker_point(ts, f"{side}FIN") - wrist
    return wrist + tip_from_wrist * direction / numpy.linalg.norm(direction, axis=1, keepdims=True)


def detect_release(ts, dt: float, tip_from_wrist: float = D.FINGERTIP_FROM_WRIST_M) -> tuple:
    """(throwing side, release frame): the arm whose fingertip reaches the higher speed,
    and the frame of that peak. The ball is not measured; the fingertip-speed peak matches
    the measured ball-release time of the OpenBiomechanics dataset to 0.1 +/- 1.4 frames
    (360 Hz) over 59 pitches, where the wrist-speed peak was 4.6 frames early.
    """
    best = None
    for side in "LR":
        speed = numpy.linalg.norm(D.derivative(fingertip(ts, side, tip_from_wrist), dt), axis=1)
        if best is None or speed.max() > best[2]:
            best = (side, int(speed.argmax()), float(speed.max()))
    return best[0], best[1]


def ball_load(ts, side: str, release_frame: int, dt: float, mass: float = D.BALL_MASS_KG,
              tip_from_wrist: float = D.FINGERTIP_FROM_WRIST_M) -> tuple:
    """The load the ball puts on the hand until release: (force, moment, point), with
    force = -m_ball (a_ball - g) at the fingertip point. Zero from release on.
    """
    tip = fingertip(ts, side, tip_from_wrist)
    accel = D.derivative(D.derivative(tip, dt), dt)
    force = -mass * (accel - D.GRAVITY)
    force[release_frame + 1:] = 0.0
    return force, numpy.zeros_like(force), tip


def solve_full_body(trial: dict, trial_name: str, cutoff_hz: float = D.DEFAULT_CUTOFF_HZ,
                    ball_mass: float = D.BALL_MASS_KG, release_frame: int | None = None,
                    arm_cutoff_hz: float = D.ARM_CUTOFF_HZ,
                    tip_from_wrist: float = D.FINGERTIP_FROM_WRIST_M) -> dict:
    """Whole-body inverse dynamics of one throw. See the module docstring.

    cutoff_hz filters the legs, trunk, head and force plates; arm_cutoff_hz the arms (the
    throwing arm needs a higher one, see D.ARM_CUTOFF_HZ). `release_frame` overrides the
    fingertip-speed-peak release estimate (e.g. with a measured ball-release time).
    """
    meta = A.get_subject_metadata().get(trial_name)
    if meta is None:
        raise KeyError(f"no metadata (body mass) for {trial_name}")
    dt = 1.0 / float(trial["point_rate"])
    ts = A.trial_to_markers_ts(D.filtered_trial(trial, cutoff_hz))

    segments = D.build_body_segments(ts, meta["mass_kg"], meta)
    if arm_cutoff_hz != cutoff_hz:
        ts_arm = A.trial_to_markers_ts(D.filtered_trial(trial, arm_cutoff_hz))
        arm_segments = D.build_body_segments(ts_arm, meta["mass_kg"], meta)
        for key in arm_segments:
            if key.endswith(("_UpperArm", "_Forearm", "_Hand")):
                segments[key] = arm_segments[key]
    else:
        ts_arm = ts
    needed = ([f"{s}_{n}" for s in "LR" for n in ("Thigh", "Shank", "Foot", "UpperArm", "Forearm", "Hand")]
              + ["UpperTrunk", "MiddleTrunk", "LowerTrunk", "Head"])
    missing = [k for k in needed if k not in segments]
    if missing:
        raise KeyError(f"segments unavailable (missing markers): {missing}")
    for key, seg in segments.items():
        seg.compute_kinematics(dt, arm_cutoff_hz if key.endswith(("_UpperArm", "_Forearm", "_Hand")) else cutoff_hz)

    wrenches = D.plate_wrenches(trial, cutoff_hz)
    foot_loads, contact = foot_plate_loads(trial, ts, wrenches)
    throw_side, detected = detect_release(ts_arm, dt, tip_from_wrist)
    release = detected if release_frame is None else int(release_frame)

    thorax = D.merge_segments(segments["UpperTrunk"], segments["MiddleTrunk"], "Thorax")
    pelvis = segments["LowerTrunk"]
    joints, thorax_loads, pelvis_loads = {}, [], []
    for side in "LR":
        hand_loads = [ball_load(ts_arm, side, release, dt, ball_mass, tip_from_wrist)] if side == throw_side else []
        forearm_hand = D.merge_segments(segments[f"{side}_Forearm"], segments[f"{side}_Hand"], "Forearm+Hand")
        joints.update(D.solve_chain({f"{side}_Forearm+Hand": forearm_hand,
                                     f"{side}_UpperArm": segments[f"{side}_UpperArm"]}, hand_loads))
        joints.update(D.solve_chain({f"{side}_Foot": segments[f"{side}_Foot"],
                                     f"{side}_Shank": segments[f"{side}_Shank"],
                                     f"{side}_Thigh": segments[f"{side}_Thigh"]}, foot_loads[side]))
        j = joints[f"{side}_UpperArm"]
        thorax_loads.append((-j["force"], -j["moment"], j["point"]))
        j = joints[f"{side}_Thigh"]
        pelvis_loads.append((-j["force"], -j["moment"], j["point"]))
    head = D.solve_chain({"Head": segments["Head"]})["Head"]
    joints["Head"] = head
    thorax_loads.append((-head["force"], -head["moment"], head["point"]))

    # Lumbar joint (at the omphalion plane, where de Leva splits the trunk): what the
    # thorax applies to the pelvis, from the pelvis's own equations -- legs pushing up
    # from below, so this is the bottom-up estimate of the load the trunk passes down.
    f_lumbar, m_lumbar = D.newton_euler_step(pelvis, pelvis_loads)
    joints["Lumbar"] = {"force": f_lumbar, "moment": m_lumbar, "point": pelvis.joint_proximal}
    thorax_loads.append((-f_lumbar, -m_lumbar, pelvis.joint_proximal))  # Newton's third law

    force_res, moment_res = D.closure_residual(thorax, thorax_loads)
    return {"segments": segments, "thorax": thorax, "pelvis": pelvis, "joints": joints,
            "trunk_residual": {"force": force_res, "moment": moment_res},
            "throwing_side": throw_side, "release_frame": release, "contact": contact,
            "wrenches": wrenches, "dt": dt, "cutoff_hz": cutoff_hz, "arm_cutoff_hz": arm_cutoff_hz,
            "ts": ts}


def humerus_long_axis_moment(result: dict, side: str) -> numpy.ndarray:
    """Shoulder moment component about the humerus's long axis (N.m): the internal /
    external rotation torque. Sign is oriented so internal rotation (the motion that
    dominates the relative angular velocity at its peak, just before release) is positive.
    """
    seg = result["segments"][f"{side}_UpperArm"]
    trunk = result["thorax"]
    long_axis = seg.rotation[:, :, 1]
    moment = numpy.einsum("ni,ni->n", result["joints"][f"{side}_UpperArm"]["moment"], long_axis)
    relative = numpy.einsum("ni,ni->n", seg.omega - trunk.omega, long_axis)
    return moment * numpy.sign(relative[numpy.abs(relative).argmax()])


def elbow_varus_moment(result: dict, side: str) -> numpy.ndarray:
    """Elbow varus/valgus moment (N.m): the elbow moment about the ISB floating axis,
    perpendicular to the humerus flexion axis (its mediolateral axis) and the forearm
    long axis. Sign is oriented so the dominant load (the varus moment that resists the
    valgus stress of the throw) is positive.
    """
    humerus = result["segments"][f"{side}_UpperArm"]
    forearm = result["segments"][f"{side}_Forearm"]
    floating = numpy.cross(forearm.rotation[:, :, 1], humerus.rotation[:, :, 2])
    floating /= numpy.linalg.norm(floating, axis=1, keepdims=True)
    moment = numpy.einsum("ni,ni->n", result["joints"][f"{side}_Forearm+Hand"]["moment"], floating)
    return moment * numpy.sign(moment[numpy.abs(moment).argmax()])


def summarize(result: dict, trial_name: str) -> None:
    side, release = result["throwing_side"], result["release_frame"]
    dt = result["dt"]
    seg = result["segments"]
    j = result["joints"]
    norm = lambda v: numpy.linalg.norm(v, axis=1)
    print(f"{trial_name}: throwing arm {side}, release (fingertip speed peak) frame {release} = {release * dt:.3f} s")
    print(f"  cutoff {result['cutoff_hz']:g} Hz; body mass {sum(s.mass for s in seg.values()):.1f} kg")
    print(f"  shoulder force |F| peak {norm(j[f'{side}_UpperArm']['force']).max():6.0f} N   "
          f"|M| peak {norm(j[f'{side}_UpperArm']['moment']).max():6.1f} N.m   "
          f"IR/ER torque peak {numpy.abs(humerus_long_axis_moment(result, side)).max():6.1f} N.m")
    print(f"  elbow    force |F| peak {norm(j[f'{side}_Forearm+Hand']['force']).max():6.0f} N   "
          f"|M| peak {norm(j[f'{side}_Forearm+Hand']['moment']).max():6.1f} N.m")
    for s in "LR":
        print(f"  {s} hip    |F| peak {norm(j[f'{s}_Thigh']['force']).max():6.0f} N   |M| peak {norm(j[f'{s}_Thigh']['moment']).max():6.1f} N.m"
              f"   knee |M| {norm(j[f'{s}_Shank']['moment']).max():6.1f}   ankle |M| {norm(j[f'{s}_Foot']['moment']).max():6.1f}")
    res = result["trunk_residual"]
    sl = slice(20, -20)
    # scale for the residual: the largest load the trunk carries
    scale_f = max(norm(j[f"{s}_Thigh"]["force"])[sl].max() for s in "LR")
    scale_m = max(norm(j[f"{s}_Thigh"]["moment"])[sl].max() for s in "LR")
    rms = lambda v: float(numpy.sqrt((v[sl] ** 2).sum(axis=1).mean()))
    print(f"  trunk closure residual: force RMS {rms(res['force']):6.0f} N ({100 * rms(res['force']) / scale_f:.0f}% of peak hip force), "
          f"moment RMS {rms(res['moment']):6.1f} N.m ({100 * rms(res['moment']) / scale_m:.0f}% of peak hip moment)")
    c = result["contact"]
    for p in range(c["owner"].shape[0]):
        owner = c["owner"][p]
        if (owner >= 0).any():
            print(f"  plate {p + 1}: loaded {int((owner >= 0).sum())} frames "
                  f"(L {int((owner == 0).sum())}, R {int((owner == 1).sum())}), "
                  f"ambiguous {int(c['ambiguous'][p].sum())}, unmatched {int(c['unmatched'][p].sum())}")


def main() -> None:
    subject_id = sys.argv[1]
    import h5py
    with h5py.File(A.H5_PATH, "r") as f:
        trials = [t for t in sorted(f[subject_id]) if f[subject_id][t].attrs["trial_type"] == "motion"]
    if len(sys.argv) > 2:
        trials = [sys.argv[2]]
    for name in trials:
        try:
            summarize(solve_full_body(load_trial(A.H5_PATH, subject_id, name), name), name)
        except Exception as exc:
            print(f"[fail] {name}: {exc}")


if __name__ == "__main__":
    main()
