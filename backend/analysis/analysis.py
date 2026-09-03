"""Biomechanics analysis: joint angles and ground reaction forces, computed from
trials stored in the HDF5 archive built by c3d_to_hdf5.py.

Hip, knee, ankle and shoulder angles are ISB-style Cardan/Euler decompositions: each
segment (Pelvis, Thigh, Shank, Foot, Thorax, UpperArm) is built as a rigid local frame
with ktk.geometry.create_transform_series, and the joint angle is the distal segment's
rotation relative to the proximal one (ktk.geometry.get_local_coordinates), decomposed
into 3 components with ktk.geometry.get_angles -- see JOINT_ANGLE_SPECS for the
proximal/distal segment and rotation sequence used per joint. Trunk still uses the
older simple vector method (angle off vertical, no 3D decomposition).
"""

import csv
from pathlib import Path

import h5py
import numpy
import kineticstoolkit as ktk

from c3d_to_hdf5 import load_trial

H5_PATH = Path("data/master.h5")

# Per-pitch height/mass, keyed by trial name (the C3D filename minus its extension),
# from the Django app's anthropometrics export. Assumes analysis.py is run from
# backend/analysis, same as H5_PATH above.
METADATA_CSV_PATH = Path("../../biomechanics/metadata (2).csv")

_subject_metadata = None


def load_subject_metadata(csv_path: Path) -> dict:
    """{trial_name: {"height_cm": ..., "mass_kg": ...}} from the metadata CSV."""
    metadata = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            trial_name = Path(row["filename_new"]).stem
            metadata[trial_name] = {
                "height_cm": float(row["session_height_m"]) * 100,
                "mass_kg": float(row["session_mass_kg"]),
            }
    return metadata


def get_subject_metadata() -> dict:
    """Lazily load and cache the metadata CSV; returns {} (with a warning) if it's missing."""
    global _subject_metadata
    if _subject_metadata is None:
        try:
            _subject_metadata = load_subject_metadata(METADATA_CSV_PATH)
        except FileNotFoundError:
            print(f"[warn] metadata CSV not found at {METADATA_CSV_PATH}; "
                  "shoulder/hip/trunk will be skipped for every trial")
            _subject_metadata = {}
    return _subject_metadata

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

def ankle_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Ankle joint center as the midpoint of the lateral/medial malleoli."""
    return midpoint(ts, f"{side}ANK", f"{side}MANK")

def knee_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Knee joint center as the midpoint of the lateral/medial femoral epicondyles."""
    return midpoint(ts, f"{side}KNE", f"{side}MKNE")

def elbow_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Elbow joint center as the midpoint of the lateral/medial humeral epicondyles."""
    return midpoint(ts, f"{side}ELB", f"{side}MELB")

def wrist_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Wrist joint center as the midpoint of the radial/ulnar styloids."""
    return midpoint(ts, f"{side}WRA", f"{side}WRB")

def normalize(v: numpy.ndarray) -> numpy.ndarray:
    """Unit vectors, per frame."""
    return v / numpy.linalg.norm(v, axis=1, keepdims=True)

def pelvis_frame(ts: ktk.TimeSeries) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Pelvis anatomical frame from the four pelvis markers (Davis et al. 1991 style):
    origin at the ASIS midpoint, axes mediolateral / anterior / vertical. Returns
    (origin, x_ml, y_ant, z_up), each (nFrames, 3). x_ml points from left to right.
    """
    lasi = marker_point(ts, "LASI")
    rasi = marker_point(ts, "RASI")
    lpsi = marker_point(ts, "LPSI")
    rpsi = marker_point(ts, "RPSI")
    masis = (lasi + rasi) / 2
    mpsi = (lpsi + rpsi) / 2

    x_ml = normalize(rasi - lasi)
    z_up = normalize(numpy.cross(x_ml, masis - mpsi))
    y_ant = normalize(numpy.cross(z_up, x_ml))
    return masis, x_ml, y_ant, z_up

def hip_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Hip joint center via Bell et al. (1989/1990): an offset scaled by the inter-ASIS
    distance, expressed in the pelvis's own anatomical frame (mediolateral / anterior /
    vertical), then rotated and translated into the global frame. The mediolateral term
    flips sign between sides; anterior-posterior and vertical don't.
    """
    lasi = marker_point(ts, "LASI")
    rasi = marker_point(ts, "RASI")
    asis_distance = numpy.linalg.norm(rasi - lasi, axis=1)
    origin, x_ml, y_ant, z_up = pelvis_frame(ts)

    ml_offset = 0.36 * asis_distance
    ml_offset = ml_offset if side == "R" else -ml_offset
    ap_offset = -0.19 * asis_distance
    axial_offset = -0.3 * asis_distance

    return (origin
            + ml_offset[:, None] * x_ml
            + ap_offset[:, None] * y_ant
            + axial_offset[:, None] * z_up)

def thorax_frame(ts: ktk.TimeSeries) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Thorax/shoulder-girdle frame from CLAV, C7, LSHO, RSHO -- stands in for the true
    AcrCS (Campbell et al. 2009), which needs a 3-marker acromion triad per shoulder
    that this marker set doesn't have. Shared by both shoulders (same construction
    style as pelvis_frame, for the same reason: a per-side frame flips handedness
    between L and R). Returns (x_ml, y_ant, z_up), each (nFrames, 3). x_ml points from
    left to right.
    """
    lsho = marker_point(ts, "LSHO")
    rsho = marker_point(ts, "RSHO")
    clav = marker_point(ts, "CLAV")
    c7 = marker_point(ts, "C7")

    x_ml = normalize(rsho - lsho)
    z_up = normalize(numpy.cross(x_ml, clav - c7))
    y_ant = normalize(numpy.cross(z_up, x_ml))
    return x_ml, y_ant, z_up

def shoulder_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """GHJ approximated with the source paper's "3D simple offset" method: a fixed
    (12, 49, 6)mm offset -- the average MRI-measured GHJ location across their 15
    participants, relative to AcrCS -- applied here in the thorax_frame approximation
    of AcrCS instead, since we lack the marker triad for the real thing. Mediolateral
    offset flips sign by side (frame is shared, like pelvis_frame's ml_offset).
    """
    sho = marker_point(ts, f"{side}SHO")
    x_ml, y_ant, z_up = thorax_frame(ts)
    x_offset = 0.012 if side == "R" else -0.012  # meters
    y_offset, z_offset = 0.049, 0.006  # meters
    return sho + x_offset * x_ml + y_offset * y_ant + z_offset * z_up

# Joint centers computed once per side and shared by every angle/segment that touches them.
JOINT_CENTERS = {
    "ankle": ankle_joint_center,
    "knee": knee_joint_center,
    "hip": hip_joint_center,
    "shoulder": shoulder_joint_center,
    "elbow": elbow_joint_center,
    "wrist": wrist_joint_center,
}

def to_position_series(p: numpy.ndarray) -> numpy.ndarray:
    """(nFrames, 3) points -> (nFrames, 4) homogeneous positions (w=1), as required by
    ktk.geometry.create_transform_series.
    """
    out = numpy.ones((p.shape[0], 4))
    out[:, :3] = p
    return out

def to_vector_series(v: numpy.ndarray) -> numpy.ndarray:
    """(nFrames, 3) directions -> (nFrames, 4) homogeneous vectors (w=0), as required
    by ktk.geometry.create_transform_series.
    """
    out = numpy.zeros((v.shape[0], 4))
    out[:, :3] = v
    return out

def build_pelvis_segment(ts: ktk.TimeSeries) -> numpy.ndarray:
    """Pelvis segment frame for the hip's ZXY angle: Z = mediolateral (exact, the
    ASIS-ASIS line) -- this is the joint's flexion/extension axis, so it has to be the
    FIRST letter of "ZXY". Origin at the ASIS midpoint.
    """
    lasi = marker_point(ts, "LASI")
    rasi = marker_point(ts, "RASI")
    lpsi = marker_point(ts, "LPSI")
    rpsi = marker_point(ts, "RPSI")
    masis = (lasi + rasi) / 2
    mpsi = (lpsi + rpsi) / 2
    return ktk.geometry.create_transform_series(
        positions=to_position_series(masis),
        z=to_vector_series(rasi - lasi),
        xz=to_vector_series(masis - mpsi),
    )

def build_thorax_segment(ts: ktk.TimeSeries) -> numpy.ndarray:
    """Thorax segment frame for the shoulder's ZXZ angle, per Wu et al. (2005) SS2.3.1:
    origin at IJ (the CLAV marker); the paper's Y_t ("pointing upward", from the
    PX/T8 midpoint to the IJ/C7 midpoint) relabeled here as Z, since it has to be the
    FIRST letter of "ZXZ" (a repeated-axis Euler sequence needs both segments to have
    their own axis of the same name) -- our dataset also happens to call vertical "Z"
    globally, which is why this relabeling reads naturally here. X falls out as the
    paper's own anterior-pointing X_t (verified empirically: ~0.87 correlated with the
    toe-to-heel direction on a static trial).

    We don't have PX (xiphoid) or T8 markers, so STRN and T10 stand in for them.
    """
    clav = marker_point(ts, "CLAV")  # IJ
    c7 = marker_point(ts, "C7")
    strn = marker_point(ts, "STRN")  # PX proxy
    t10 = marker_point(ts, "T10")  # T8 proxy
    vertical = (clav + c7) / 2 - (strn + t10) / 2
    return ktk.geometry.create_transform_series(
        positions=to_position_series(clav),
        z=to_vector_series(vertical),
        xz=to_vector_series(clav - c7),
    )

def build_limb_segment(
    origin: numpy.ndarray,
    long_axis_vector: numpy.ndarray,
    ml_vector: numpy.ndarray,
    *,
    long_axis_letter: str = "y",
) -> numpy.ndarray:
    """Shared construction for Thigh/Shank/Foot (long_axis_letter="y", the middle
    letter of the hip/knee/ankle "ZXY" sequence -- their shared long/rotation axis)
    and UpperArm (long_axis_letter="z", the LAST letter of the shoulder's "ZXZ" --
    the humerus's own axial-rotation axis). The long axis is exact/preserved and
    points proximally; the mediolateral vector is the secondary, approximate helper.
    """
    return ktk.geometry.create_transform_series(
        positions=to_position_series(origin),
        yz=to_vector_series(ml_vector),
        **{long_axis_letter: to_vector_series(long_axis_vector)},
    )

def mirror_ml(v: numpy.ndarray, side: str) -> numpy.ndarray:
    """Negate the mediolateral (Y) component of a direction vector for the left side,
    per Wu et al. (2002/2005): "mirror the raw position data with respect to the
    sagittal plane" before building a segment's orientation with the same right-side
    formula. This dataset's raw mediolateral axis is Y, confirmed empirically: RASI-LASI
    is ~[0.001, -0.242, -0.007] -- almost entirely a Y component.

    For a leg that stays roughly vertical, or an arm that stays roughly down at the
    side, a vector's own Y component is small regardless of side, so this makes little
    difference -- but it matters a lot for e.g. an arm held out to the side (a T-pose,
    or partway through a throw), where Y dominates. Only for orientation vectors, not
    joint-center positions -- those are already correct per side and don't need this.
    """
    if side == "R":
        return v
    mirrored = v.copy()
    mirrored[:, 1] = -mirrored[:, 1]
    return mirrored

def build_side_segments(ts: ktk.TimeSeries, side: str, centers: dict) -> dict:
    """Thigh, Shank, Foot and UpperArm segment frames for one side, from that side's
    joint centers plus the medial/lateral marker pair at each segment's far joint.
    """
    segments = {}
    segments["Thigh"] = build_limb_segment(
        centers["hip"],
        mirror_ml(centers["hip"] - centers["knee"], side),
        mirror_ml(marker_point(ts, f"{side}KNE") - marker_point(ts, f"{side}MKNE"), side),
    )
    segments["Shank"] = build_limb_segment(
        centers["knee"],
        mirror_ml(centers["knee"] - centers["ankle"], side),
        mirror_ml(marker_point(ts, f"{side}ANK") - marker_point(ts, f"{side}MANK"), side),
    )
    segments["Foot"] = build_limb_segment(
        centers["ankle"],
        mirror_ml(centers["ankle"] - marker_point(ts, f"{side}TOE"), side),
        mirror_ml(marker_point(ts, f"{side}ANK") - marker_point(ts, f"{side}MANK"), side),
    )
    segments["UpperArm"] = build_limb_segment(
        centers["shoulder"],
        mirror_ml(centers["shoulder"] - centers["elbow"], side),
        # Negated once more than the other segments: this pairing (z primary + yz
        # secondary) needs the extra flip to land its derived X on the anterior side
        # rather than posterior -- verified empirically (~0.8 correlated with the
        # toe-to-heel direction, same sign both sides) on a static trial.
        -mirror_ml(marker_point(ts, f"{side}ELB") - marker_point(ts, f"{side}MELB"), side),
        long_axis_letter="z",
    )
    return segments

def unwrap_deg(angles: numpy.ndarray) -> numpy.ndarray:
    """Remove artificial +-360 deg jumps from a per-frame angle sequence -- a
    continuous rotation that legitimately crosses the +-180 deg wraparound boundary
    would otherwise show as a discontinuous jump between two adjacent frames that are
    physically almost identical (e.g. -179 deg to +179 deg in one frame).
    """
    return numpy.degrees(numpy.unwrap(numpy.radians(angles)))

def compute_joint_angles(proximal_segment: numpy.ndarray, distal_segment: numpy.ndarray, seq: str) -> numpy.ndarray:
    """3D joint angles (deg): the distal segment's rotation relative to the proximal
    segment, decomposed with Cardan/Euler sequence `seq` and unwrapped along time.
    Returns (nFrames, 3).
    """
    relative = ktk.geometry.get_local_coordinates(distal_segment, proximal_segment)
    angles = ktk.geometry.get_angles(relative, seq, degrees=True)
    return numpy.column_stack([unwrap_deg(angles[:, i]) for i in range(angles.shape[1])])

# joint_name: (proximal segment, distal segment, ISB-style Cardan/Euler sequence).
# Sequence letters refer to each segment's OWN local axes as built above -- e.g. hip's
# "Z" is Pelvis's mediolateral axis (its own Z) followed by Thigh's own long axis (Y),
# not a single shared lab-frame axis.
JOINT_ANGLE_SPECS = {
    "hip": ("Pelvis", "Thigh", "ZXY"),
    "knee": ("Thigh", "Shank", "ZXY"),
    "ankle": ("Shank", "Foot", "ZXY"),
    # Wu et al. (2005) SS2.4.7 specifies "Y-X-Y" order for the GH joint (both Thorax
    # and Humerus use their own "Y" for the repeated vertical/long axis); relabeled to
    # "Z" here since that's what our dataset calls vertical, and both build_thorax_
    # segment and build_side_segments' UpperArm construction follow that relabeling.
    "shoulder": ("Thorax", "UpperArm", "ZXZ"),
}

ANGLE_COMPONENT_NAMES = {
    "ZXY": ("flexion", "adduction", "rotation"),
    "ZXZ": ("plane_of_elevation", "elevation", "rotation"),
}

def compute_trunk_angle(centers_by_side: dict) -> numpy.ndarray:
    """Trunk lean: one value for the whole body (not per side), the angle between the
    pelvis-to-thorax line and vertical, where pelvis/thorax are the midpoints of the
    L/R hip and shoulder joint centers. 180 deg = upright, same convention as the
    other joint angles.

    Assumes a Z-up marker coordinate system -- flip to the Y axis if your lab's
    capture volume is Y-up.
    """
    pelvis = (centers_by_side["L"]["hip"] + centers_by_side["R"]["hip"]) / 2
    thorax = (centers_by_side["L"]["shoulder"] + centers_by_side["R"]["shoulder"]) / 2
    straight_up = thorax + numpy.array([0.0, 0.0, 1.0])
    return segment_angle(thorax, pelvis, straight_up)


def compute_trial_angles(trial: dict, trial_name: str) -> dict:
    """Return {"time": ..., "L_hip_flexion": ..., "L_hip_adduction": ...,
    "L_hip_rotation": ..., ..., "trunk": ...} for one trial. Hip/knee/ankle/shoulder
    are 3-component ISB-style Cardan/Euler angles (see JOINT_ANGLE_SPECS); trunk is
    still the older single-scalar lean angle.
    """
    ts = trial_to_markers_ts(trial)
    result = {"time": ts.time}
    anthro = get_subject_metadata().get(trial_name)

    centers_by_side = {}
    for side in ("L", "R"):
        centers = {}
        for center_name, center_func in JOINT_CENTERS.items():
            try:
                centers[center_name] = center_func(ts, side, anthro)
            except (KeyError, TypeError):
                pass  # trial is missing a marker (or anthropometry) this center needs
        centers_by_side[side] = centers

    shared_segments = {}
    try:
        shared_segments["Pelvis"] = build_pelvis_segment(ts)
    except KeyError:
        pass  # missing a pelvis marker
    try:
        shared_segments["Thorax"] = build_thorax_segment(ts)
    except KeyError:
        pass  # missing a thorax marker

    for side in ("L", "R"):
        try:
            segments = {**shared_segments, **build_side_segments(ts, side, centers_by_side[side])}
        except KeyError:
            segments = dict(shared_segments)  # this side's limb segments are missing a marker/center

        for joint_name, (proximal_name, distal_name, seq) in JOINT_ANGLE_SPECS.items():
            if proximal_name not in segments or distal_name not in segments:
                continue
            angles = compute_joint_angles(segments[proximal_name], segments[distal_name], seq)
            for i, component in enumerate(ANGLE_COMPONENT_NAMES[seq]):
                result[f"{side}_{joint_name}_{component}"] = angles[:, i]

    try:
        result["trunk"] = compute_trunk_angle(centers_by_side)
    except KeyError:
        pass  # missing a center the trunk segment needs
    return result


def compute_static_offsets(trial: dict, trial_name: str) -> dict:
    """Per-component mean angle from a subject's static trial: their own "neutral
    pose" reference. Subtracted from that subject's other trials (see
    apply_static_offsets) so reported angles reflect motion relative to their own
    anatomy and marker placement, not an absolute number that bakes those in --
    the same reason Vicon Plug-in-Gait and similar models use a static calibration.
    """
    raw = compute_trial_angles(trial, trial_name)
    return {k: float(numpy.nanmean(v)) for k, v in raw.items() if k != "time"}


def apply_static_offsets(angles: dict, offsets: dict) -> dict:
    """Subtract a subject's static-trial offsets from one of their trials' raw angles.
    A component missing from `offsets` (e.g. the static trial itself was missing a
    marker this component needs) is passed through unadjusted.
    """
    return {k: (v - offsets[k] if k != "time" and k in offsets else v) for k, v in angles.items()}


def compute_all_angles(h5_path: Path) -> dict:
    """Compute joint angles for every trial in the archive, referenced to each
    subject's own static-trial neutral pose (see compute_static_offsets).

    Returns {(subject_id, trial_name): angles_dict}.
    """
    results = {}
    with h5py.File(h5_path, "r") as h5_file:
        subject_ids = sorted(h5_file.keys())
        for subject_id in subject_ids:
            trial_names = sorted(h5_file[subject_id].keys())

            static_trial_names = [
                t for t in trial_names
                if h5_file[subject_id][t].attrs.get("trial_type") == "static"
            ]
            static_offsets = {}
            if static_trial_names:
                static_trial_name = static_trial_names[0]
                try:
                    static_trial = load_trial(h5_path, subject_id, static_trial_name)
                    static_offsets = compute_static_offsets(static_trial, static_trial_name)
                except Exception as exc:
                    print(f"[warn] {subject_id}/{static_trial_name}: couldn't compute "
                          f"static offsets ({exc}); this subject's trials will use raw angles")
            else:
                print(f"[warn] {subject_id}: no static trial found; using raw angles")

            for trial_name in trial_names:
                trial = load_trial(h5_path, subject_id, trial_name)
                try:
                    raw = compute_trial_angles(trial, trial_name)
                    results[(subject_id, trial_name)] = apply_static_offsets(raw, static_offsets)
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
