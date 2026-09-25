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

# Hip joint center: fixed offsets in the pelvis frame (origin at the ASIS midpoint), as a
# fraction of the inter-ASIS distance -- [mediolateral (+right), anteroposterior (+anterior),
# vertical]. Fitted to the hip centers the OpenBiomechanics Project publishes for these same
# pitches (landmarks.csv, 59 pitches, 17 subjects): mean +/- sd over all frames was
# ML 0.534 +/- 0.035, AP -0.278 +/- 0.03, vertical -0.446 +/- 0.03, the same for both hips
# and every subject, i.e. a fixed-offset method. It replaces the Bell et al. (1989/1990)
# coefficients I started with (0.36, -0.19, -0.30), which put the hip centers ~6.6 cm from
# theirs (17 cm between hips for a 104 kg male; theirs is 26 cm) and made the lead-hip
# moment 35% too low. The knee/ankle/elbow/wrist centers already agree with theirs to 1-3 mm.
HIP_CENTER_OFFSET = (0.534, -0.278, -0.446)

def hip_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """Hip joint center: HIP_CENTER_OFFSET (scaled by the inter-ASIS distance) expressed in
    the pelvis's own frame (mediolateral / anterior / vertical), then rotated and
    translated into the global frame. The mediolateral term flips sign between sides.
    """
    lasi = marker_point(ts, "LASI")
    rasi = marker_point(ts, "RASI")
    asis_distance = numpy.linalg.norm(rasi - lasi, axis=1)
    origin, x_ml, y_ant, z_up = pelvis_frame(ts)

    ml_fraction, ap_fraction, axial_fraction = HIP_CENTER_OFFSET
    ml_offset = ml_fraction * asis_distance
    ml_offset = ml_offset if side == "R" else -ml_offset
    ap_offset = ap_fraction * asis_distance
    axial_offset = axial_fraction * asis_distance

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

# Shoulder (glenohumeral) center: fixed offset from the acromion (SHO) marker in the thorax
# frame, [mediolateral (+right), anterior, vertical] in meters. Fitted to the centers the
# OpenBiomechanics Project publishes for these pitches: 0.0 +/- 0.005, +0.019 +/- 0.005,
# -0.046 +/- 0.004 m, identical for both shoulders. The "3D simple offset" I started with
# (0.012, 0.049, 0.006 m, from Campbell et al. 2009's AcrCS) put the center 6.2 cm from theirs.
SHOULDER_CENTER_OFFSET = (0.0, 0.019, -0.046)

def shoulder_joint_center(ts: ktk.TimeSeries, side: str, anthro: dict | None) -> numpy.ndarray:
    """GHJ as SHOULDER_CENTER_OFFSET from the acromion marker, in the thorax_frame
    approximation (shared by both shoulders, so the mediolateral term flips sign by side).
    """
    sho = marker_point(ts, f"{side}SHO")
    x_ml, y_ant, z_up = thorax_frame(ts)
    x_offset, y_offset, z_offset = SHOULDER_CENTER_OFFSET
    x_offset = x_offset if side == "R" else -x_offset
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

def mirror_markers(ts: ktk.TimeSeries) -> ktk.TimeSeries:
    """Reflect every marker through the sagittal plane (negate Y, the mediolateral axis
    -- RASI-LASI is ~[0.001, -0.242, -0.007]) and swap the L/R names, per Wu et al.
    (2002/2005): the LEFT side is analysed as a right side of a mirrored subject, using
    the exact same right-side formulas. Everything has to be reflected together
    (pelvis and thorax included, not only the limb being measured): a proximal frame left
    unmirrored disagrees with a mirrored distal one as soon as the trunk rotates, which
    is most of a throw.
    """
    mirrored = ktk.TimeSeries(time=ts.time)
    for name, xyz in ts.data.items():
        swapped = {"L": "R", "R": "L"}.get(name[0], "") + name[1:]
        new = xyz.copy()
        new[:, 1] = -new[:, 1]
        mirrored.data[swapped if swapped in ts.data else name] = new
    return mirrored

def build_side_segments(ts: ktk.TimeSeries, side: str, centers: dict) -> dict:
    """Thigh, Shank, Foot and UpperArm segment frames for one side, from that side's
    joint centers plus the medial/lateral marker pair at each segment's far joint.
    Right-side formulas only: for the left, pass a mirror_markers() series as `ts` (and
    side="R").
    """
    segments = {}
    segments["Thigh"] = build_limb_segment(
        centers["hip"],
        (centers["hip"] - centers["knee"]),
        (marker_point(ts, f"{side}KNE") - marker_point(ts, f"{side}MKNE")),
    )
    segments["Shank"] = build_limb_segment(
        centers["knee"],
        (centers["knee"] - centers["ankle"]),
        (marker_point(ts, f"{side}ANK") - marker_point(ts, f"{side}MANK")),
    )
    segments["Foot"] = build_limb_segment(
        centers["ankle"],
        (centers["ankle"] - marker_point(ts, f"{side}TOE")),
        (marker_point(ts, f"{side}ANK") - marker_point(ts, f"{side}MANK")),
    )
    segments["UpperArm"] = build_limb_segment(
        centers["shoulder"],
        (centers["shoulder"] - centers["elbow"]),
        # Negated once more than the other segments: this pairing (z primary + yz
        # secondary) needs the extra flip to land its derived X on the anterior side
        # rather than posterior -- verified empirically (~0.8 correlated with the
        # toe-to-heel direction, same sign both sides) on a static trial.
        -(marker_point(ts, f"{side}ELB") - marker_point(ts, f"{side}MELB")),
        long_axis_letter="z",
    )
    return segments

# de Leva (1996) "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters",
# J. Biomech 29(9):1223-1230, Table 4, MALE columns -- checked against the paper itself.
# Per segment:
#   mass_frac -- fraction of total body mass
#   com_frac  -- COM distance from the segment's PROXIMAL end, fraction of segment length
#   r_sagittal / r_transverse / r_longitudinal -- radii of gyration about the COM as a
#       fraction of segment length. Inertia about an axis = mass * (r * length)**2.
#       I assign "sagittal" to the anteroposterior axis and "transverse" to the
#       mediolateral one; they differ by <=8% for the limbs, and the (much smaller)
#       longitudinal radius is the one that matters.
# Segment endpoints the percentages refer to (paper's Table 4 abbreviations in brackets):
#   Head: vertex -> C7 [VERT-CERV]     Trunk: C7 -> mid-hip [CERV-MIDH]
#   UpperTrunk: suprasternale (CLAV marker) -> xiphoid (STRN marker) [SUPR-XYPH]
#   MiddleTrunk: xiphoid -> omphalion [XYPH-OMPH]     LowerTrunk: omphalion -> mid-hip [OMPH-MIDH]
#   UpperArm: shoulder -> elbow    Forearm: elbow -> wrist    Hand: wrist -> 3rd metacarpal
#   (our finger marker, 8.8 cm from the wrist, matches the paper's 86 mm hand)
#   Thigh: hip -> knee    Shank: knee -> ankle joint center [KJC-AJC]    Foot: heel -> toe tip.
# "Trunk" is kept for reference/whole-trunk use; UpperTrunk + MiddleTrunk + LowerTrunk
# are the same 43.46% of body mass split three ways. Masses of the non-overlapping set
# (Head, the three trunk parts, and 2 x each limb segment) sum to 1.0.
SEGMENT_ANTHROPOMETRY = {
    "Head": {"mass_frac": 0.0694, "com_frac": 0.5002, "r_sagittal": 0.303, "r_transverse": 0.315, "r_longitudinal": 0.261},
    "Trunk": {"mass_frac": 0.4346, "com_frac": 0.5138, "r_sagittal": 0.328, "r_transverse": 0.306, "r_longitudinal": 0.169},
    "UpperTrunk": {"mass_frac": 0.1596, "com_frac": 0.2999, "r_sagittal": 0.716, "r_transverse": 0.454, "r_longitudinal": 0.659},
    "MiddleTrunk": {"mass_frac": 0.1633, "com_frac": 0.4502, "r_sagittal": 0.482, "r_transverse": 0.383, "r_longitudinal": 0.468},
    "LowerTrunk": {"mass_frac": 0.1117, "com_frac": 0.6115, "r_sagittal": 0.615, "r_transverse": 0.551, "r_longitudinal": 0.587},
    "UpperArm": {"mass_frac": 0.0271, "com_frac": 0.5772, "r_sagittal": 0.285, "r_transverse": 0.269, "r_longitudinal": 0.158},
    "Forearm": {"mass_frac": 0.0162, "com_frac": 0.4574, "r_sagittal": 0.276, "r_transverse": 0.265, "r_longitudinal": 0.121},
    "Hand": {"mass_frac": 0.0061, "com_frac": 0.7900, "r_sagittal": 0.628, "r_transverse": 0.513, "r_longitudinal": 0.401},
    "Thigh": {"mass_frac": 0.1416, "com_frac": 0.4095, "r_sagittal": 0.329, "r_transverse": 0.329, "r_longitudinal": 0.149},
    "Shank": {"mass_frac": 0.0433, "com_frac": 0.4395, "r_sagittal": 0.251, "r_transverse": 0.246, "r_longitudinal": 0.102},
    "Foot": {"mass_frac": 0.0137, "com_frac": 0.4415, "r_sagittal": 0.257, "r_transverse": 0.245, "r_longitudinal": 0.124},
}

def unwrap_deg(angles: numpy.ndarray) -> numpy.ndarray:
    """Remove artificial +-360 deg jumps from a per-frame angle sequence -- a
    continuous rotation that legitimately crosses the +-180 deg wraparound boundary
    would otherwise show as a discontinuous jump between two adjacent frames that are
    physically almost identical (e.g. -179 deg to +179 deg in one frame).
    """
    return numpy.degrees(numpy.unwrap(numpy.radians(angles)))

def relative_rotations(proximal_segment: numpy.ndarray, distal_segment: numpy.ndarray) -> numpy.ndarray:
    """Distal segment's orientation in the proximal segment's frame, (nFrames, 3, 3)."""
    return ktk.geometry.get_local_coordinates(distal_segment, proximal_segment)[:, :3, :3]

def mean_rotation(rotations: numpy.ndarray) -> numpy.ndarray:
    """Average of a stack of rotation matrices, projected back onto SO(3) (SVD), so the
    result is a proper right-handed rotation rather than just an element-wise mean.
    """
    valid = rotations[~numpy.isnan(rotations).any(axis=(1, 2))]
    u, _, vt = numpy.linalg.svd(valid.mean(axis=0))
    d = numpy.sign(numpy.linalg.det(u @ vt))
    return u @ numpy.diag([1.0, 1.0, d]) @ vt

def compute_joint_angles(proximal_segment: numpy.ndarray, distal_segment: numpy.ndarray, seq: str,
                         reference: numpy.ndarray | None = None) -> numpy.ndarray:
    """3D joint angles (deg): the distal segment's rotation relative to the proximal
    segment, decomposed with Cardan/Euler sequence `seq` and unwrapped along time.
    Returns (nFrames, 3).

    `reference` is the joint's neutral-pose rotation (3x3, from the static trial). It is
    removed by composition, R_rel @ reference.T, i.e. the distal frame is redefined so it
    coincides with the proximal frame in the neutral pose -- exact for any range of
    motion, unlike subtracting per-component Euler angles (only valid for small angles).
    """
    rotations = relative_rotations(proximal_segment, distal_segment)
    if reference is not None:
        rotations = rotations @ reference.T
    relative = numpy.tile(numpy.eye(4), (len(rotations), 1, 1))
    relative[:, :3, :3] = rotations
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

# Joints whose zero is the subject's static-trial neutral pose. The shoulder is NOT here:
# the static trial is a T-pose (arms out, ~70 deg elevation), and referencing a ZXZ
# angle to it would put neutral exactly on the sequence's singularity (elevation = 0,
# where plane-of-elevation and rotation are indeterminate). The shoulder keeps its
# anatomical Wu et al. (2005) zero instead: arm hanging alongside the trunk.
STATIC_REFERENCED_JOINTS = {"hip", "knee", "ankle"}

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


def build_trial_segments(trial: dict, trial_name: str) -> tuple:
    """Return (ts, centers_by_side, segments_by_side) for one trial. Anything whose
    markers (or anthropometry) are missing is simply left out. centers_by_side is in
    the lab frame (for the trunk lean); segments_by_side["L"] is built in the
    sagittally mirrored frame (see mirror_markers) so both sides share one convention.
    """
    ts = trial_to_markers_ts(trial)
    anthro = get_subject_metadata().get(trial_name)
    frames = {"R": (ts, "R"), "L": (mirror_markers(ts), "R")}  # side: (series, side name in it)

    centers_by_side, segments_by_side = {}, {}
    for side in ("L", "R"):
        for target, (series, name_side) in (("centers", (ts, side)), ("segments", frames[side])):
            centers = {}
            for center_name, center_func in JOINT_CENTERS.items():
                try:
                    centers[center_name] = center_func(series, name_side, anthro)
                except (KeyError, TypeError):
                    pass  # trial is missing a marker (or anthropometry) this center needs
            if target == "centers":
                centers_by_side[side] = centers
                continue

            segments = {}
            try:
                segments["Pelvis"] = build_pelvis_segment(series)
            except KeyError:
                pass  # missing a pelvis marker
            try:
                segments["Thorax"] = build_thorax_segment(series)
            except KeyError:
                pass  # missing a thorax marker
            try:
                segments.update(build_side_segments(series, name_side, centers))
            except KeyError:
                pass  # this side's limb segments are missing a marker/center
            segments_by_side[side] = segments
    return ts, centers_by_side, segments_by_side


def compute_trial_angles(trial: dict, trial_name: str, references: dict | None = None) -> dict:
    """Return {"time": ..., "L_hip_flexion": ..., "L_hip_adduction": ...,
    "L_hip_rotation": ..., ..., "trunk": ...} for one trial. Hip/knee/ankle/shoulder
    are 3-component ISB-style Cardan/Euler angles (see JOINT_ANGLE_SPECS); trunk is
    still the older single-scalar lean angle.

    `references` ({"L_hip": 3x3 neutral rotation, ...}, see compute_static_references)
    is removed from each joint by rotation composition; joints without one are raw.
    """
    ts, centers_by_side, segments_by_side = build_trial_segments(trial, trial_name)
    result = {"time": ts.time}

    for side, segments in segments_by_side.items():
        for joint_name, (proximal_name, distal_name, seq) in JOINT_ANGLE_SPECS.items():
            if proximal_name not in segments or distal_name not in segments:
                continue
            reference = (references or {}).get(f"{side}_{joint_name}")
            angles = compute_joint_angles(segments[proximal_name], segments[distal_name], seq, reference)
            for i, component in enumerate(ANGLE_COMPONENT_NAMES[seq]):
                result[f"{side}_{joint_name}_{component}"] = angles[:, i]

    try:
        result["trunk"] = compute_trunk_angle(centers_by_side)
    except KeyError:
        pass  # missing a center the trunk segment needs
    return result


def compute_static_references(trial: dict, trial_name: str) -> dict:
    """A subject's neutral pose from their static trial: per joint, the mean relative
    rotation of distal in proximal (kept as a rotation matrix), plus the scalar mean of
    the trunk lean angle. Passed to compute_trial_angles / apply_static_offsets.
    """
    _, centers_by_side, segments_by_side = build_trial_segments(trial, trial_name)
    references = {}
    for side, segments in segments_by_side.items():
        for joint_name, (proximal_name, distal_name, _) in JOINT_ANGLE_SPECS.items():
            if joint_name in STATIC_REFERENCED_JOINTS and proximal_name in segments and distal_name in segments:
                references[f"{side}_{joint_name}"] = mean_rotation(
                    relative_rotations(segments[proximal_name], segments[distal_name]))
    try:
        references["trunk"] = float(numpy.nanmean(compute_trunk_angle(centers_by_side)))
    except KeyError:
        pass
    return references


def apply_static_offsets(angles: dict, offsets: dict) -> dict:
    """Subtract the scalar trunk-lean offset (the only scalar angle left; joint angles
    are already referenced by rotation composition in compute_trial_angles).
    """
    return {k: (v - offsets[k] if k == "trunk" and k in offsets else v) for k, v in angles.items()}


def compute_all_angles(h5_path: Path) -> dict:
    """Compute joint angles for every trial in the archive, referenced to each
    subject's own static-trial neutral pose (see compute_static_references).

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
                    static_offsets = compute_static_references(static_trial, static_trial_name)
                except Exception as exc:
                    print(f"[warn] {subject_id}/{static_trial_name}: couldn't compute "
                          f"static offsets ({exc}); this subject's trials will use raw angles")
            else:
                print(f"[warn] {subject_id}: no static trial found; using raw angles")

            for trial_name in trial_names:
                trial = load_trial(h5_path, subject_id, trial_name)
                try:
                    raw = compute_trial_angles(trial, trial_name, static_offsets)
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


#Now to compute GRFs, need to load the force plates in the X,Y, Z directions and determine the GRFs in those planes.
#With that, can use inverse dynamics to compute the joint moments and forces for each pitcher and throw in the HDF5 files 
def compute_grfs(trial: dict) -> dict:
    """Compute ground reaction forces (GRFs) from force plate data in the trial.
    Returns a dictionary with GRF components in X, Y, Z directions.
    """
    grfs = {}
    try:
        # Assuming the trial contains force plate data in 'force_plates' key
        force_plates = trial['force_plates']  # This should be a numpy array of shape (nFrames, 3)
        grfs['GRF_X'] = force_plates[:, 0]  # X direction
        grfs['GRF_Y'] = force_plates[:, 1]  # Y direction
        grfs['GRF_Z'] = force_plates[:, 2]  # Z direction
    except KeyError:
        print(f"[warn] Force plate data not found in trial; GRFs will be skipped.")
    return grfs

#Define segment properties for inverse dynamics calculations
# (segment, proximal joint center / marker, distal joint center / marker), per side.
# Foot uses the heel marker (HEE) when the trial has one, since Winter's foot length and
# COM are measured from the heel.
SEGMENT_ENDPOINTS = {
    "Thigh": ("hip", "knee"),
    "Shank": ("knee", "ankle"),
    "UpperArm": ("shoulder", "elbow"),
    "Forearm": ("elbow", "wrist"),
}


