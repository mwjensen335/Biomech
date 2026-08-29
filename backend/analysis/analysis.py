"""Biomechanics analysis: joint angles and ground reaction forces, computed from
trials stored in the HDF5 archive built by c3d_to_hdf5.py.

Joint angles use the simple vector method: a joint center is the midpoint of its
medial/lateral marker pair, and the joint angle is the angle between the proximal
and distal segments meeting at that center (180 deg = fully extended). This does
not separate flexion/extension from valgus/varus or rotation.
"""

from pathlib import Path

import h5py
import numpy
import kineticstoolkit as ktk

from c3d_to_hdf5 import load_trial

H5_PATH = Path("data/master.h5")

height = ()
mass = ()

def trial_to_markers_ts(trial: dict) -> ktk.TimeSeries:
    """Build a ktk TimeSeries of markers from a trial loaded via load_trial()."""
    n_markers, n_frames, _ = trial["points"].shape
    time = numpy.arange(n_frames) / trial["point_rate"]

    ts = ktk.TimeSeries(time=time)
    for i, label in enumerate(trial["point_labels"]):
        xyz = trial["points"][i]  # (nFrames, 3)
        homogeneous = numpy.ones((n_frames, 4))
        homogeneous[:, :3] = xyz
        ts.data[label.strip()] = homogeneous

    return ts


def marker_point(ts: ktk.TimeSeries, name: str) -> numpy.ndarray:
    """(nFrames, 3) marker trajectory from a markers TimeSeries."""
    return ts.data[name][:, :3]


def midpoint(ts: ktk.TimeSeries, marker_a: str, marker_b: str) -> numpy.ndarray:
    """(nFrames, 3) midpoint between two markers -- used as an approximate joint center."""
    return (marker_point(ts, marker_a) + marker_point(ts, marker_b)) / 2


def segment_angle(center: numpy.ndarray, proximal_point: numpy.ndarray, distal_point: numpy.ndarray) -> numpy.ndarray:
    """Angle (degrees) between the two segments meeting at `center`, per frame.
    180 deg = fully extended (proximal and distal segments in a straight line).
    """
    v1 = proximal_point - center
    v2 = distal_point - center
    cos_angle = numpy.einsum("ij,ij->i", v1, v2) / (
        numpy.linalg.norm(v1, axis=1) * numpy.linalg.norm(v2, axis=1)
    )
    return numpy.degrees(numpy.arccos(numpy.clip(cos_angle, -1.0, 1.0)))

def compute_knee_angle(ts: ktk.TimeSeries, side: str) -> numpy.ndarray:
    hip = marker_point(ts, f"{side}ASI")
    knee_center = midpoint(ts, f"{side}KNE", f"{side}MKNE")
    ankle_center = midpoint(ts, f"{side}ANK", f"{side}MANK")
    return segment_angle(knee_center, hip, ankle_center)

def compute_ankle_angle(ts: ktk.TimeSeries, side: str) -> numpy.ndarray:
    knee_center = midpoint(ts, f"{side}KNE", f"{side}MKNE")
    ankle_center = midpoint(ts, f"{side}ANK", f"{side}MANK")
    toe = marker_point(ts, f"{side}TOE")
    return segment_angle(ankle_center, knee_center, toe)

def hip_joint_center(ts: ktk.TimeSeries, side: str) -> numpy.ndarray:
    """Approximate hip joint center from ASIS and PSIS markers."""
    asis = marker_point(ts, f"{side}ASI")
    psis = marker_point(ts, f"{side}PSI")
    right_x_axis = 0.36 * (asis[:,0])
    right_y_axis = -0.19 * (asis[:,1])
    right_z_axis = -0.3 * (asis[:,2])
    left_x_axis = -0.36 * (asis[:,0])
    left_y_axis = -0.19 * (asis[:,1])
    left_z_axis = -0.3 * (asis[:,2])
    return numpy.stack([right_x_axis, right_y_axis, right_z_axis], axis=1) if side == "R" else numpy.stack([left_x_axis, left_y_axis, left_z_axis], axis=1)

def shoulder_joint_center(ts: ktk.TimeSeries, side: str) -> numpy.ndarray:
    """Approximate shoulder joint center from acromion, jugular notch, C7 and the middle point of the jugular notch and C7"""
    """Regression equation"""
    acromion = marker_point(ts, f"{side}ACR")
    jugular_notch = marker_point(ts, "CLAV")
    c7 = marker_point (ts, "C7")
    CP = (jugular_notch + c7) / 2
    x_axis = 96.2 - 0.302 * (jugular_notch[:, 0] - c7[:, 0]) - 0.364 * (height/100) + 0.385 * mass 
    y_axis = -66.32 + 0.30 * (jugular_notch[:, 1] - c7[:, 1]) - 0.432 * mass 
    z_axis = 66.468 - 0.531 * (acromion[:, 3] - CP[:, 3]) + 0.571 * mass 
    return numpy.stack([x_axis, y_axis, z_axis], axis=1)

JOINT_FUNCS = {
    "knee": compute_knee_angle,
    "ankle": compute_ankle_angle
}


def compute_trial_angles(trial: dict) -> dict:
    """Return {"time": ..., "L_knee": ..., "R_knee": ..., ...} for one trial."""
    ts = trial_to_markers_ts(trial)
    result = {"time": ts.time}
    for side in ("L", "R"):
        for joint_name, joint_func in JOINT_FUNCS.items():
            try:
                result[f"{side}_{joint_name}"] = joint_func(ts, side)
            except KeyError:
                pass  # trial is missing a marker this joint needs
    return result


def compute_all_angles(h5_path: Path) -> dict:
    """Compute joint angles for every trial in the archive.

    Returns {(subject_id, trial_name): angles_dict}.
    """
    results = {}
    with h5py.File(h5_path, "r") as h5_file:
        subject_ids = sorted(h5_file.keys())
        for subject_id in subject_ids:
            for trial_name in sorted(h5_file[subject_id].keys()):
                trial = load_trial(h5_path, subject_id, trial_name)
                try:
                    results[(subject_id, trial_name)] = compute_trial_angles(trial)
                except Exception as exc:
                    print(f"[fail] {subject_id}/{trial_name}: {exc}")
    return results


def main() -> None:
    all_angles = compute_all_angles(H5_PATH)
    for (subject_id, trial_name), angles in all_angles.items():
        joint_keys = [k for k in angles if k != "time"]
        summary = ", ".join(
            f"{k}: {numpy.nanmin(angles[k]):.1f}-{numpy.nanmax(angles[k]):.1f} deg"
            for k in joint_keys
        )
        print(f"{subject_id}/{trial_name}  {summary}")


if __name__ == "__main__":
    main()
