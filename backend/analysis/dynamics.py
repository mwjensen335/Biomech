"""3D inverse-dynamics building blocks: filtered kinematics, rigid-body segments with a
full inertia tensor (de Leva 1996, see analysis.SEGMENT_ANTHROPOMETRY), the Newton-Euler
step, and force-plate handling.

Everything here is in the LAB frame (Z up, Y mediolateral, X anterior; meters, kg, N).
The mirrored left-side frames analysis.py builds for joint-angle reporting are NOT used:
a diagonal inertia tensor does not care about the sign of an axis, and the 
unmirrored right-handed frame per segment is for dynamics.

Sign convention for the results: the force/moment "at a joint" is what the PROXIMAL
segment applies to the DISTAL one (e.g. the shoulder moment is the moment the trunk
applies to the upper arm) -- the usual "net joint moment" reading.
"""

from dataclasses import dataclass, field

import numpy
from scipy.signal import butter, filtfilt

import analysis as A

GRAVITY = numpy.array([0.0, 0.0, -9.81])


DEFAULT_CUTOFF_HZ = 12.0

# The throwing arm needs a much higher cutoff than the rest of the body. Chosen against
# an independent reference: the OpenBiomechanics dataset's own joint kinetics for the same
# 59 pitches (see validate_against_obp.py). Shoulder/elbow moment and force errors are
# lowest at 20-25 Hz (at 12 Hz the peaks are 20-25% too low; at 40 Hz they are 10-40%
# too high, from noise amplification).
ARM_CUTOFF_HZ = 22.0
BALL_MASS_KG = 0.142  # 5 oz
FINGERTIP_FROM_WRIST_M = 0.19  # ball held at the fingertips; no fingertip marker exists

# Every dynamics frame is built long-axis = local y, mediolateral = local z, so
# anteroposterior = local x. (analysis.build_limb_segment, long_axis_letter="y").
# Radii of gyration are assigned to the local axes accordingly.
LOCAL_AXIS_RADII = ("r_sagittal", "r_longitudinal", "r_transverse")  # about local x, y, z


# --------------------------------------------------------------------------- filtering

def lowpass(x: numpy.ndarray, fs: float, cutoff_hz: float, axis: int = 0, order: int = 2) -> numpy.ndarray:
    """Zero-lag Butterworth low-pass. filtfilt runs the filter forward and backward, so
    `order=2` is an effective 4th-order, no-phase-shift filter.
    """
    b, a = butter(order, cutoff_hz / (fs / 2))
    return filtfilt(b, a, x, axis=axis)


def filtered_trial(trial: dict, cutoff_hz: float = DEFAULT_CUTOFF_HZ) -> dict:
    """Copy of `trial` with the marker trajectories low-passed. Filter BEFORE
    differentiating: each derivative amplifies noise.
    """
    if numpy.isnan(trial["points"]).any():
        raise ValueError("marker data has gaps (NaN); fill them before filtering")
    out = dict(trial)
    out["points"] = lowpass(trial["points"], float(trial["point_rate"]), cutoff_hz, axis=1)
    return out


def derivative(x: numpy.ndarray, dt: float) -> numpy.ndarray:
    """Central-difference d/dt along axis 0 (second-order accurate at the ends too)."""
    return numpy.gradient(x, dt, axis=0, edge_order=2)


def angular_velocity(rotations: numpy.ndarray, dt: float) -> numpy.ndarray:
    """Angular velocity (lab frame, rad/s) from a (n,3,3) rotation series: the axial
    vector of the skew-symmetric matrix R_dot @ R^T. Never differentiate Euler angles.
    """
    r_dot = derivative(rotations, dt)
    w = r_dot @ numpy.swapaxes(rotations, 1, 2)
    w = (w - numpy.swapaxes(w, 1, 2)) / 2  # remove numerical asymmetry
    return numpy.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], axis=1)


def to_local(vectors: numpy.ndarray, rotations: numpy.ndarray) -> numpy.ndarray:
    """Express lab-frame (n,3) vectors in the frames given by (n,3,3) rotations."""
    return numpy.einsum("nji,nj->ni", rotations, vectors)


# --------------------------------------------------------------------------- segments

@dataclass
class Segment:
    name: str
    mass: float                    # kg
    proximal: numpy.ndarray        # (n,3) proximal joint center / end point
    distal: numpy.ndarray          # (n,3)
    rotation: numpy.ndarray        # (n,3,3) segment frame in the lab frame
    inertia_local: numpy.ndarray   # (3,) principal moments about the COM, kg m^2
    com_frac: float
    length: float = 0.0            # m, median over the trial (a rigid body's length is constant)
    joint_proximal: numpy.ndarray = field(default=None)  # (n,3) where the proximal neighbour attaches
                                   # (defaults to `proximal`; differs for the foot: ankle, not heel)
    com: numpy.ndarray = field(default=None)
    velocity: numpy.ndarray = field(default=None)
    acceleration: numpy.ndarray = field(default=None)
    omega: numpy.ndarray = field(default=None)      # rad/s, lab frame
    alpha: numpy.ndarray = field(default=None)      # rad/s^2, lab frame
    inertia_lab: numpy.ndarray = field(default=None)  # (n,3,3)

    def compute_kinematics(self, dt: float, cutoff_hz: float | None = None) -> "Segment":
        fs = 1.0 / dt
        if self.joint_proximal is None:
            self.joint_proximal = self.proximal
        self.com = self.proximal + self.com_frac * (self.distal - self.proximal)
        self.velocity = derivative(self.com, dt)
        self.acceleration = derivative(self.velocity, dt)
        self.omega = angular_velocity(self.rotation, dt)
        omega = lowpass(self.omega, fs, cutoff_hz) if cutoff_hz else self.omega
        self.alpha = derivative(omega, dt)
        self.inertia_lab = self.rotation @ numpy.diag(self.inertia_local) @ numpy.swapaxes(self.rotation, 1, 2)
        return self


def make_segment(name: str, body_mass: float, proximal: numpy.ndarray, distal: numpy.ndarray,
                 long_axis: numpy.ndarray | None, ml_axis: numpy.ndarray | None,
                 joint_proximal: numpy.ndarray | None = None, frame: numpy.ndarray | None = None,
                 radii_order: tuple = LOCAL_AXIS_RADII) -> Segment:
    """Rigid segment from its two end points. `long_axis`/`ml_axis` orient the frame:
    only their directions matter (sign is irrelevant for a diagonal inertia tensor).
    Alternatively pass a ready-made `frame` (n,3,3) and `radii_order`, the radii of
    gyration for the frame's local x, y, z.
    """
    anthro = A.SEGMENT_ANTHROPOMETRY[name]
    length = float(numpy.median(numpy.linalg.norm(distal - proximal, axis=1)))
    mass = anthro["mass_frac"] * body_mass
    if frame is None:
        frame = A.build_limb_segment(proximal, long_axis, ml_axis)[:, :3, :3]
    principal = numpy.array([mass * (anthro[k] * length) ** 2 for k in radii_order])
    return Segment(name=name, mass=mass, proximal=proximal, distal=distal, rotation=frame,
                   inertia_local=principal, com_frac=anthro["com_frac"], length=length,
                   joint_proximal=joint_proximal)


def build_body_segments(ts, body_mass: float, anthro: dict | None = None) -> dict:
    """{"R_Forearm": Segment, ..., "Trunk": Segment, "Head": Segment}. A segment whose
    markers are missing is omitted. Hand length is wrist center -> finger marker, which
    stops at the knuckles rather than the fingertip, so the hand's inertia is a bit small.
    """
    pt = lambda name: A.marker_point(ts, name)
    segments = {}

    def add(key, name, proximal, distal, long_axis, ml_axis, joint=None):
        try:
            segments[key] = make_segment(name, body_mass, proximal(), distal(), long_axis(), ml_axis(),
                                         joint() if joint else None)
        except KeyError:
            pass  # a marker/center this segment needs is missing

    centers = {}
    for side in ("L", "R"):
        centers[side] = {}
        for cname, cfunc in A.JOINT_CENTERS.items():
            try:
                centers[side][cname] = cfunc(ts, side, anthro)
            except (KeyError, TypeError):
                pass

    for side in ("L", "R"):
        c = centers[side]
        ml_knee = lambda s=side: pt(f"{s}KNE") - pt(f"{s}MKNE")
        ml_ankle = lambda s=side: pt(f"{s}ANK") - pt(f"{s}MANK")
        ml_elbow = lambda s=side: pt(f"{s}ELB") - pt(f"{s}MELB")
        ml_wrist = lambda s=side: pt(f"{s}WRA") - pt(f"{s}WRB")
        add(f"{side}_Thigh", "Thigh", lambda: c["hip"], lambda: c["knee"], lambda: c["hip"] - c["knee"], ml_knee)
        add(f"{side}_Shank", "Shank", lambda: c["knee"], lambda: c["ankle"], lambda: c["knee"] - c["ankle"], ml_ankle)
        add(f"{side}_Foot", "Foot", lambda s=side: pt(f"{s}HEE"), lambda s=side: pt(f"{s}TOE"),
            lambda s=side: pt(f"{s}HEE") - pt(f"{s}TOE"), ml_ankle, joint=lambda: c["ankle"])
        add(f"{side}_UpperArm", "UpperArm", lambda: c["shoulder"], lambda: c["elbow"],
            lambda: c["shoulder"] - c["elbow"], ml_elbow)
        add(f"{side}_Forearm", "Forearm", lambda: c["elbow"], lambda: c["wrist"],
            lambda: c["elbow"] - c["wrist"], ml_wrist)
        add(f"{side}_Hand", "Hand", lambda: c["wrist"], lambda s=side: pt(f"{s}FIN"),
            lambda s=side: c["wrist"] - pt(f"{s}FIN"), ml_wrist)

    try:
        mid_hip = (centers["L"]["hip"] + centers["R"]["hip"]) / 2
        c7 = pt("C7")

        # Trunk in three de Leva parts. Upper + middle ride on the thorax markers/frame,
        # the lower part on the pelvis markers/frame. The omphalion (navel) has no marker:
        # it is placed on the xiphoid -> mid-hip line at the fraction the paper's mean
        # part lengths give (middle 215.5 mm, lower 145.7 mm).
        omphalion_fraction = 215.5 / (215.5 + 145.7)
        # de Leva's landmarks lie on the trunk's central longitudinal axis, but CLAV/STRN
        # are on the front surface, so a COM placed on the CLAV-STRN line
        # swings with the thorax's fast twist. Use the mid-thickness points instead --
        # the same ones Wu et al. (2005) use for the thorax long axis (IJ/C7, PX/T8).
        clav, strn = (pt("CLAV") + c7) / 2, (pt("STRN") + pt("T10")) / 2
        omph = strn + omphalion_fraction * (mid_hip - strn)
        thorax_frame = A.build_thorax_segment(ts)[:, :3, :3]   # x ant, y ML, z long
        pelvis_frame = A.build_pelvis_segment(ts)[:, :3, :3]   # x ant, y long, z ML
        thorax_radii = ("r_sagittal", "r_transverse", "r_longitudinal")
        for key, prox, dist, frame, radii in (
                ("UpperTrunk", clav, strn, thorax_frame, thorax_radii),
                ("MiddleTrunk", strn, omph, thorax_frame, thorax_radii),
                ("LowerTrunk", omph, mid_hip, pelvis_frame, LOCAL_AXIS_RADII)):
            segments[key] = make_segment(key, body_mass, prox, dist, None, None,
                                         frame=frame, radii_order=radii)

        # No vertex marker: the head-marker centroid stands in for the head COM (de Leva
        # puts the COM ~50% of the way from the vertex to C7), so the vertex is C7
        # mirrored through the centroid.
        head = (pt("LFHD") + pt("RFHD") + pt("LBHD") + pt("RBHD")) / 4
        vertex = 2 * head - c7
        add("Head", "Head", lambda: vertex, lambda: c7, lambda: vertex - c7,
            lambda: (pt("RFHD") + pt("RBHD")) / 2 - (pt("LFHD") + pt("LBHD")) / 2,
            joint=lambda: c7)  # the head hangs from its C7 end, not the vertex
    except KeyError:
        pass
    return segments


# --------------------------------------------------------------------------- Newton-Euler

# A "load" is (force (n,3), moment (n,3), point (n,3) or (3,)): a force applied at `point`
# plus a pure couple, both in the lab frame, acting ON the segment.

def _net_effort(seg: Segment, loads: list) -> tuple:
    """What the (unknown) proximal joint must supply for Newton-Euler to hold.

        F_p + sum F_i + m g                      = m a
        M_p + sum (M_i + r_i x F_i) + r_p x F_p  = I alpha + omega x (I omega)      (about the COM)

    Returns (m(a-g) - sum F_i,  I alpha + omega x (I omega) - sum (M_i + r_i x F_i)).
    The omega x (I omega) gyroscopic term is what makes 3D different from 2D, and it is
    large in a throw.
    """
    force = seg.mass * (seg.acceleration - GRAVITY)
    i_omega = numpy.einsum("nij,nj->ni", seg.inertia_lab, seg.omega)
    moment = (numpy.einsum("nij,nj->ni", seg.inertia_lab, seg.alpha) + numpy.cross(seg.omega, i_omega))
    for f_i, m_i, point in loads:
        force = force - f_i
        moment = moment - m_i - numpy.cross(numpy.asarray(point) - seg.com, f_i)
    return force, moment


def newton_euler_step(seg: Segment, loads: list) -> tuple:
    """(F_p, M_p): the force and moment the PROXIMAL neighbor applies to `seg`, at
    seg.joint_proximal, given every other load on it (lab frame).
    """
    force, moment = _net_effort(seg, loads)
    return force, moment - numpy.cross(seg.joint_proximal - seg.com, force)


def closure_residual(seg: Segment, loads: list) -> tuple:
    """For a segment whose loads are ALL known (e.g. the trunk, with the arms and head
    coming down and the legs coming up), how far Newton-Euler is from holding:
    (force residual (n,3) N, moment residual (n,3) N.m about the COM). Zero in a perfect model.
    """
    return _net_effort(seg, loads)


def solve_chain(chain: dict, end_loads: list | None = None) -> dict:
    """Distal -> proximal solve of an ordered {key: Segment} chain, e.g.
    {"R_Forearm+Hand": fh, "R_UpperArm": ua} or {"L_Foot": f, "L_Shank": s, "L_Thigh": t}.
    `end_loads` are the external loads on the FIRST (distal-most) segment: the ball, the
    ground reaction. Each segment's reaction is passed on, reversed, to the next.

    Returns {key: {"force": (n,3), "moment": (n,3), "point": (n,3)}}: what the proximal
    neighbor applies to that segment at that point (lab frame).
    """
    loads = list(end_loads or [])
    joints = {}
    for key, seg in chain.items():
        f_prox, m_prox = newton_euler_step(seg, loads)
        joints[key] = {"force": f_prox, "moment": m_prox, "point": seg.joint_proximal}
        loads = [(-f_prox, -m_prox, seg.joint_proximal)]  # Newton's third law at the joint
    return joints


def merge_segments(a: Segment, b: Segment, name: str) -> Segment:
    """Rigid composite of `a` (proximal) and `b`, e.g. forearm + hand. COM and its
    acceleration are mass-weighted; angular kinematics are taken from `a`; inertia is
    summed about the composite COM with the parallel-axis theorem. Merging is common for
    the hand (small, and its orientation from a 9 cm baseline is too noisy alone).
    """
    mass = a.mass + b.mass
    com = (a.mass * a.com + b.mass * b.com) / mass
    inertia = numpy.zeros_like(a.inertia_lab)
    eye = numpy.eye(3)
    for seg in (a, b):
        d = seg.com - com
        inertia += seg.inertia_lab + seg.mass * (
            numpy.einsum("ni,ni->n", d, d)[:, None, None] * eye - numpy.einsum("ni,nj->nij", d, d))
    return Segment(name=name, mass=mass, proximal=a.proximal, distal=b.distal, rotation=a.rotation,
                   inertia_local=a.inertia_local, com_frac=a.com_frac, length=a.length,
                   joint_proximal=a.joint_proximal, com=com,
                   velocity=(a.mass * a.velocity + b.mass * b.velocity) / mass,
                   acceleration=(a.mass * a.acceleration + b.mass * b.acceleration) / mass,
                   omega=a.omega, alpha=a.alpha, inertia_lab=inertia)


# --------------------------------------------------------------------------- force plates

# Which in-plane edge direction is each plate's x axis. Determined from the data, not
# assumed: for every motion trial of all 17 subjects, plate x = -u  gave a
# whole-body Newton residual of 40-220 N against 300-1200 N
# choices (sum of plate forces vs M(a_COM - g) from de Leva segments). With it the plate
# axes are x = +X (anterior, toward home plate), y = -Y, z = down.
PLATE_X_CHOICE = 2


def plate_axes(corners: numpy.ndarray, x_choice: int = None) -> numpy.ndarray:
    """Lab-frame rotation (3,3; columns = plate x, y, z axes) of one plate from its four
    C3D corners (4,3). The plate z axis points DOWN (the C3D convention for a Type-2
    platform), so the raw vertical force reads negative for a body standing on it, and
    R @ F_raw is then the force the plate applies to the subject, in lab coordinates.

    The plate's x axis is one of the four in-plane edge directions; `x_choice` (0-3)
    selects (+u, +w, -u, -w) where u = corner1->corner2 and w = corner1->corner4.
    The corners alone do not settle which; PLATE_X_CHOICE was determined empirically
    (see the note there).
    """
    x_choice = PLATE_X_CHOICE if x_choice is None else x_choice
    u = corners[1] - corners[0]
    w = corners[3] - corners[0]
    u, w = u / numpy.linalg.norm(u), w / numpy.linalg.norm(w)
    normal = numpy.cross(u, w)
    if normal[2] < 0:
        normal = -normal
    z_axis = -normal  # down
    x_axis = (u, w, -u, -w)[x_choice]
    y_axis = numpy.cross(z_axis, x_axis)
    return numpy.stack([x_axis, y_axis, z_axis], axis=1)


def plate_forces(trial: dict, cutoff_hz: float | None = DEFAULT_CUTOFF_HZ, x_choices=None) -> numpy.ndarray:
    """Force each plate applies to the subject, lab frame: (nPlates, nFrames, 3), sampled
    at the MARKER rate (analog sample k*spf lines up with marker frame k). Low-passed at
    the same cutoff as the kinematics so both sides of Newton's equation have the same
    bandwidth.
    """
    fp = trial["force_platforms"]
    if fp is None:
        raise KeyError("trial has no force-platform geometry; run c3d_to_hdf5 backfill-platforms")
    spf = int(round(float(trial["analog_samples_per_frame"])))
    fs = float(trial["analog_rate"])
    n_frames = int(trial["n_frames"])
    out = []
    for p in range(len(fp["corners"])):
        rows = [c - 1 for c in fp["channel"][p][:3]]
        raw = trial["analogs"][rows].T.astype(float)  # (nAnalog, 3)
        if cutoff_hz:
            raw = lowpass(raw, fs, cutoff_hz)
        choice = None if x_choices is None else x_choices[p]
        lab = raw @ plate_axes(fp["corners"][p], choice).T
        out.append(lab[::spf][:n_frames])
    return numpy.array(out)


def whole_body_com_acceleration(segments: dict) -> tuple:
    """(total mass, COM (n,3), COM acceleration (n,3)) from a full set of segments."""
    mass = sum(s.mass for s in segments.values())
    com = sum(s.mass * s.com for s in segments.values()) / mass
    return mass, com, sum(s.mass * s.acceleration for s in segments.values()) / mass


def plate_wrenches(trial: dict, cutoff_hz: float | None = DEFAULT_CUTOFF_HZ) -> dict:
    """Everything a plate applies to the subject, lab frame, at the marker rate:

    "force"   (nPlates, n, 3)  N
    "moment"  (nPlates, n, 3)  N.m about the plate's CENTER point (below), lab axes
    "center"  (nPlates, 3)     m, the plate-surface center the moment is taken about
    "cop"     (nPlates, n, 2)  m, lab X/Y of the center of pressure (NaN below 20 N)

    A wrench (F, M_c) at point c acts on a segment with COM at r as a force F and a
    moment M_c + (c - r) x F about the COM -- no center of pressure needed for that.
    Assumes the C3D ORIGIN is (0, 0, 0), i.e. moments are already about the surface
    center (true for this dataset).
    """
    fp = trial["force_platforms"]
    if fp is None:
        raise KeyError("trial has no force-platform geometry; run c3d_to_hdf5 backfill-platforms")
    if numpy.abs(fp["origin"]).max() > 1e-9:
        raise NotImplementedError("non-zero C3D platform ORIGIN is not handled")
    spf = int(round(float(trial["analog_samples_per_frame"])))
    fs = float(trial["analog_rate"])
    n_frames = int(trial["n_frames"])
    forces, moments, centers, cops = [], [], [], []
    for p in range(len(fp["corners"])):
        rows = [c - 1 for c in fp["channel"][p][:6]]
        raw = trial["analogs"][rows].T.astype(float)  # (nAnalog, 6)
        if cutoff_hz:
            raw = lowpass(raw, fs, cutoff_hz)
        rot = plate_axes(fp["corners"][p])
        f = (raw[:, :3] @ rot.T)[::spf][:n_frames]
        m = (raw[:, 3:] @ rot.T)[::spf][:n_frames]
        c = fp["corners"][p].mean(axis=0).astype(float)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            up = f[:, 2]
            cop = numpy.stack([c[0] - m[:, 1] / up, c[1] + m[:, 0] / up], axis=1)
        cop[numpy.abs(up) < 20] = numpy.nan
        forces.append(f), moments.append(m), centers.append(c), cops.append(cop)
    return {"force": numpy.array(forces), "moment": numpy.array(moments),
            "center": numpy.array(centers), "cop": numpy.array(cops)}
