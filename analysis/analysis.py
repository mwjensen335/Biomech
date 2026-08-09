"""Biomechanics analysis: joint angles and ground reaction forces, computed from
trials stored in the HDF5 archive built by c3d_to_hdf5.py.

Joint angles use the simple vector method: a joint center is the midpoint of its
medial/lateral marker pair, and the joint angle is the angle between the proximal
and distal segments meeting at that center (180 deg = fully extended). This does
not separate flexion/extension from valgus/varus or rotation.
"""

from pathlib import Path

import numpy

from c3d_to_hdf5 import get_analog, get_marker, load_trial

H5_PATH = Path("data/master.h5")


def joint_center(trial: dict, lateral: str, medial: str) -> numpy.ndarray:
    """Midpoint of a medial/lateral marker pair, per frame -> (nFrames, 3)."""
    return (get_marker(trial, lateral) + get_marker(trial, medial)) / 2


def vector_angle(proximal: numpy.ndarray, joint: numpy.ndarray, distal: numpy.ndarray) -> numpy.ndarray:
    """Angle in degrees at `joint`, formed by proximal-joint-distal, per frame.

    180 degrees = proximal/joint/distal collinear (fully extended); smaller = more flexed.
    """
    v1 = proximal - joint
    v2 = distal - joint
    dot = numpy.einsum("ij,ij->i", v1, v2)
    norms = numpy.linalg.norm(v1, axis=1) * numpy.linalg.norm(v2, axis=1)
    cos_angle = numpy.clip(dot / norms, -1.0, 1.0)
    return numpy.degrees(numpy.arccos(cos_angle))


def elbow_angle(trial: dict, side: str) -> numpy.ndarray:
    """side: 'R' or 'L'. Shoulder-elbow-wrist angle, per frame."""
    shoulder = get_marker(trial, f"{side}SHO")
    elbow = joint_center(trial, f"{side}ELB", f"{side}MELB")
    wrist = joint_center(trial, f"{side}WRA", f"{side}WRB")
    return vector_angle(shoulder, elbow, wrist)


def resultant_grf(trial: dict, plate: int) -> numpy.ndarray:
    """Resultant ground reaction force magnitude for one force plate (1-3), per analog sample."""
    fx = get_analog(trial, f"Fx{plate}")
    fy = get_analog(trial, f"Fy{plate}")
    fz = get_analog(trial, f"Fz{plate}")
    return numpy.sqrt(fx**2 + fy**2 + fz**2)


if __name__ == "__main__":
    trial = load_trial(H5_PATH, subject_id="000072", trial_name="000072_002996_76_229_001_FF_889")

    r_elbow = elbow_angle(trial, "R")
    print(f"R elbow angle over {len(r_elbow)} frames: "
          f"min={r_elbow.min():.1f} deg, max={r_elbow.max():.1f} deg")

    grf1 = resultant_grf(trial, 1)
    print(f"Plate 1 resultant GRF over {len(grf1)} samples: peak={grf1.max():.1f}")
