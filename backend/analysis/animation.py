#This file will take the analysis function and using matplotlibs animation abilites,
#use the angles and forces to give instantenous and angles/forces real time along side the pitcher actually throwing
#Using the angles and forces, can compute the joint moments and forces for each pitcher and throw in the HDF5 files
#As well using KTK toolkit animation functions, can create a 3D animation of the pitcher throwing the ball with the angles and forces being displayed in real time.

"""Animate a pitch from its C3D markers with the pelvis/torso angles and the shoulder loads
running alongside, in slow motion.

What the dashboard shows (all from the validated pipeline: inverse_dynamics.py):
  * left: the pitcher in 3D (markers -> joint centers), the three force plates lit by the
    vertical ground-reaction force with GRF arrows at the center of pressure, the ball in the
    hand until release and then in flight, and two arrows at the throwing shoulder -- the
    total shoulder moment (grey) and its internal-rotation part along the humerus (red);
  * top right: pelvis rotation and torso rotation, 0 deg = facing home plate, closed (turned
    away) negative;
  * middle right: shoulder internal-rotation moment and elbow varus moment for the whole
    throw, a moving cursor, the part already played shaded, and the peaks marked -- so the
    full range is visible at every instant;
  * bottom right: live bars for the shoulder internal-rotation torque and elbow varus torque
    against their range over the throw.
Time is shown relative to ball release (0 s).

Run:  python animation.py <subject_id> [trial_name] [--out FILE.gif] [--speed 0.25] [--fps 30]
                          [--snapshot] [--view arm|glove|front] [--player]
  --snapshot  also saves a PNG of the dashboard at ball release
  --player    opens the interactive KTK 3D player instead (needs a display; not headless)
GIF writing needs only Pillow. (MP4 would need ffmpeg, which is not installed here.)
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # file output; --player uses KTK's own Qt window instead
import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import analysis as A
import dynamics as D
import inverse_dynamics as ID
from c3d_to_hdf5 import load_trial
from export_results import arm_frame_table, lead_foot_contact_frame

DRAW_CUTOFF_HZ = 20.0       # smoothing for what is drawn (the dynamics use their own cutoffs)
MOMENT_ARROW_M_PER_NM = 0.0022   # 300 N.m draws as 0.66 m
GRF_ARROW_M_PER_N = 1.0 / 2500.0
PLATE_FULL_SCALE_N = 2500.0

COLORS = {"throw": "#e8710a", "glove": "#1a73e8", "rear": "#188038", "lead": "#8430ce",
          "pelvis": "#d93025", "torso": "#1a73e8", "sep": "#188038", "ir": "#d93025",
          "varus": "#8430ce", "grf": "#00838f", "trunk": "#5f6368"}


# --------------------------------------------------------------------------- data

def prepare(subject: str, trial_name: str) -> dict:
    """Everything the dashboard needs, precomputed per frame."""
    trial = load_trial(A.H5_PATH, subject, trial_name)
    result = ID.solve_full_body(trial, trial_name)
    side = result["throwing_side"]
    glove = "L" if side == "R" else "R"
    rear, lead = side, glove
    dt, release = result["dt"], result["release_frame"]
    ts = A.trial_to_markers_ts(D.filtered_trial(trial, DRAW_CUTOFF_HZ))
    pt = lambda name: A.marker_point(ts, name)
    n = len(ts.time)

    centers = {s: {name: func(ts, s, None) for name, func in A.JOINT_CENTERS.items()} for s in "LR"}
    mid = lambda a, b: (a + b) / 2
    mid_hip = mid(centers["L"]["hip"], centers["R"]["hip"])
    mid_shoulder = mid(centers["L"]["shoulder"], centers["R"]["shoulder"])
    head = (pt("LFHD") + pt("RFHD") + pt("LBHD") + pt("RBHD")) / 4
    spine = numpy.stack([mid_hip, mid(pt("STRN"), pt("T10")), mid(pt("CLAV"), pt("C7")), head], axis=1)

    lines = {"spine": (spine, COLORS["trunk"], 3.0), "hips": (numpy.stack([centers["L"]["hip"], centers["R"]["hip"]], 1), COLORS["trunk"], 3.0),
             "shoulders": (numpy.stack([centers["L"]["shoulder"], mid(pt("CLAV"), pt("C7")), centers["R"]["shoulder"]], 1), COLORS["trunk"], 3.0),
             "head": (numpy.stack([pt("LFHD"), pt("RFHD"), pt("RBHD"), pt("LBHD"), pt("LFHD")], 1), COLORS["trunk"], 1.5)}
    for s in "LR":
        c = centers[s]
        arm_color = COLORS["throw"] if s == side else COLORS["glove"]
        leg_color = COLORS["rear"] if s == rear else COLORS["lead"]
        lines[f"{s}_arm"] = (numpy.stack([c["shoulder"], c["elbow"], c["wrist"], pt(f"{s}FIN")], 1), arm_color, 3.5 if s == side else 2.5)
        lines[f"{s}_leg"] = (numpy.stack([c["hip"], c["knee"], c["ankle"], pt(f"{s}HEE"), pt(f"{s}TOE"), c["ankle"]], 1), leg_color, 3.0)
    polys = {"pelvis": (numpy.stack([pt("LASI"), pt("RASI"), pt("RPSI"), pt("LPSI")], 1), COLORS["pelvis"], 0.35),
             "torso": (numpy.stack([centers["L"]["shoulder"], centers["R"]["shoulder"], centers["R"]["hip"], centers["L"]["hip"]], 1), COLORS["trunk"], 0.12)}

    # ball: fingertip until release, then ballistic with the fingertip's release velocity
    tip = ID.fingertip(ts, side)
    tip_velocity = D.derivative(tip, dt)
    ball = tip.copy()
    tau = (numpy.arange(n) - release) * dt
    after = numpy.arange(n) > release
    ball[after] = tip[release] + tip_velocity[release] * tau[after, None] + 0.5 * D.GRAVITY * tau[after, None] ** 2

    # shoulder loads at the throwing shoulder
    upper_arm = result["segments"][f"{side}_UpperArm"]
    shoulder_moment = result["joints"][f"{side}_UpperArm"]["moment"]
    long_axis = upper_arm.rotation[:, :, 1]
    ir_vector = numpy.einsum("ni,ni->n", shoulder_moment, long_axis)[:, None] * long_axis

    # pelvis / torso rotation about the vertical, 0 = facing home plate (+X), closed < 0
    sign = 1.0 if side == "R" else -1.0
    rotation = {}
    for name, frame in (("pelvis", A.build_pelvis_segment(result["ts"])), ("torso", A.build_thorax_segment(result["ts"]))):
        anterior = frame[:, :3, 0]
        rotation[name] = sign * numpy.degrees(numpy.unwrap(numpy.arctan2(anterior[:, 1], anterior[:, 0])))
    rotation["separation"] = rotation["pelvis"] - rotation["torso"]  # + = pelvis open ahead of the torso

    wrenches = result["wrenches"]
    plates = {"corners": load_trial(A.H5_PATH, subject, trial_name)["force_platforms"]["corners"],
              "force": wrenches["force"], "cop": wrenches["cop"], "center": wrenches["center"]}
    table = arm_frame_table(result)
    markers = numpy.stack([A.marker_point(ts, label) for label in trial["point_labels"]], axis=1)  # (n, nMarkers, 3)
    return {"result": result, "trial": trial_name, "markers": markers, "subject": subject, "side": side, "dt": dt, "n": n, "release": release,
            "contact": lead_foot_contact_frame(result), "lines": lines, "polys": polys, "ball": ball,
            "shoulder": centers[side]["shoulder"], "shoulder_moment": shoulder_moment, "ir_vector": ir_vector,
            "rotation": rotation, "plates": plates, "table": table,
            "mph": int(trial_name.split("_")[-1]) / 10.0}


# --------------------------------------------------------------------------- figure

def arrow_artists(ax, color, width):
    (line,) = ax.plot([numpy.nan] * 2, [numpy.nan] * 2, [numpy.nan] * 2, color=color, lw=width, solid_capstyle="round")
    (tip,) = ax.plot([numpy.nan], [numpy.nan], [numpy.nan], marker="o", ms=width * 1.6, color=color)
    return line, tip


def set_arrow(artists, base, vector):
    line, tip = artists
    end = base + vector
    line.set_data_3d([base[0], end[0]], [base[1], end[1]], [base[2], end[2]])
    tip.set_data_3d([end[0]], [end[1]], [end[2]])


def build_figure(d: dict, view: str) -> dict:
    fig = plt.figure(figsize=(13, 7.3), facecolor="white")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1.05, 0.8],
                            left=0.02, right=0.975, top=0.90, bottom=0.07, hspace=0.42, wspace=0.10)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax_ang, ax_mom, ax_bar = (fig.add_subplot(grid[i, 1]) for i in range(3))
    art = {"fig": fig, "ax3": ax3}

    # ---- 3D scene
    points = numpy.concatenate([v[0].reshape(-1, 3) for v in d["lines"].values()])
    lo, hi = numpy.nanmin(points, axis=0) - 0.25, numpy.nanmax(points, axis=0) + 0.25
    lo[2] = 0.0
    ax3.set_xlim(lo[0], hi[0]), ax3.set_ylim(lo[1], hi[1]), ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(tuple(hi - lo), zoom=1.3)
    # camera azimuth (deg, 0 = camera on the +X side looking back at the pitcher): on the
    # throwing-arm side (-Y for a right-hander), turned 18 deg toward the front for depth
    azimuth = {"arm": -72 if d["side"] == "R" else 72, "glove": 72 if d["side"] == "R" else -72, "front": 0}[view]
    ax3.view_init(elev=10, azim=azimuth)
    ax3.set_axis_off()
    ax3.text2D(0.02, 0.97, "Pitcher's view: +X is toward home plate", transform=ax3.transAxes, fontsize=8, color="#5f6368", va="top")

    (art["markers"],) = ax3.plot([], [], [], ".", color="#3c4043", ms=5, alpha=0.75)  # the C3D markers themselves
    plate_polys = Poly3DCollection(d["plates"]["corners"], facecolor="#e8eaed", edgecolor="#5f6368", lw=1.0, alpha=0.9)
    ax3.add_collection3d(plate_polys)
    art["plates"] = plate_polys
    art["grf"] = [arrow_artists(ax3, COLORS["grf"], 2.6) for _ in range(len(d["plates"]["corners"]))]
    art["lines"] = {k: ax3.plot([], [], [], color=c, lw=w, solid_capstyle="round")[0] for k, (_, c, w) in d["lines"].items()}
    art["polys"] = {}
    for k, (_, c, a) in d["polys"].items():
        poly = Poly3DCollection([numpy.zeros((4, 3))], facecolor=c, edgecolor=c, alpha=a)
        ax3.add_collection3d(poly)
        art["polys"][k] = poly
    (art["ball"],) = ax3.plot([numpy.nan], [numpy.nan], [numpy.nan], marker="o", ms=9, mfc="white", mec="black", mew=1.6, ls="")
    art["moment"] = arrow_artists(ax3, "#9aa0a6", 3.0)
    art["ir"] = arrow_artists(ax3, COLORS["ir"], 5.0)
    art["title"] = fig.suptitle("", fontsize=13, fontweight="bold", y=0.975)
    art["event"] = fig.text(0.02, 0.05, "", fontsize=12, fontweight="bold", color="#c5221f")

    # ---- curves
    t = d["table"].time_s.to_numpy() - d["release"] * d["dt"]
    d["t"] = t
    ir, varus = d["table"].shoulder_ir_moment.to_numpy(), d["table"].elbow_varus_moment.to_numpy()
    for ax in (ax_ang, ax_mom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
        ax.set_xlim(t[0], t[-1])
        for frame, label in ((d["contact"], "lead foot\ncontact"), (d["release"], "ball\nrelease")):
            if frame is not None:
                ax.axvline(t[frame], color="#9aa0a6", ls=":", lw=1.2)
                ax.text(t[frame], 1.0, label, transform=ax.get_xaxis_transform(), fontsize=7, color="#5f6368", ha="center", va="bottom")
    rot = d["rotation"]
    ax_ang.plot(t, rot["pelvis"], color=COLORS["pelvis"], lw=2, label="Pelvis rotation")
    ax_ang.plot(t, rot["torso"], color=COLORS["torso"], lw=2, label="Torso rotation")
    ax_ang.axhline(0, color="black", lw=0.6)
    ax_ang.set_ylabel("deg  (0 = facing home)", fontsize=9)
    ax_ang.legend(loc="lower right", fontsize=8, frameon=False, ncol=1)
    ax_ang.set_title("Pelvis and torso angles", fontsize=10, loc="left", pad=16)
    ax_mom.plot(t, ir, color=COLORS["ir"], lw=2, label="Shoulder internal-rotation moment")
    ax_mom.plot(t, varus, color=COLORS["varus"], lw=2, label="Elbow varus moment")
    ax_mom.axhline(0, color="black", lw=0.6)
    for series, color, offset in ((ir, COLORS["ir"], (-62, -4)), (varus, COLORS["varus"], (8, 2))):
        i = int(numpy.abs(series).argmax())
        ax_mom.plot(t[i], series[i], "v", color=color, ms=7)
        ax_mom.annotate(f"peak {series[i]:.0f}", (t[i], series[i]), textcoords="offset points", xytext=offset, fontsize=8, color=color)
    ax_mom.set_ylabel("N.m", fontsize=9)
    ax_mom.set_xlabel("time from ball release (s)", fontsize=9)
    ax_mom.legend(loc="upper left", fontsize=8, frameon=False)
    ax_mom.set_title("Shoulder and elbow moments (net joint moment, throwing arm)", fontsize=10, loc="left", pad=16)

    art["cursors"] = [ax.axvline(t[0], color="black", lw=1.6) for ax in (ax_ang, ax_mom)]
    art["dots"] = {
        "pelvis": ax_ang.plot([], [], "o", color=COLORS["pelvis"])[0], "torso": ax_ang.plot([], [], "o", color=COLORS["torso"])[0],
        "ir": ax_mom.plot([], [], "o", color=COLORS["ir"], ms=8)[0], "varus": ax_mom.plot([], [], "o", color=COLORS["varus"], ms=8)[0]}
    art["ir_fill"] = None
    art["ax_mom"] = ax_mom

    # ---- live bars: value against its range over the throw
    quantities = [("Shoulder internal-rotation torque (N.m)", ir, COLORS["ir"]), ("Elbow varus torque (N.m)", varus, COLORS["varus"])]
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.75, 1.75 * len(quantities) - 1.0)
    ax_bar.axis("off")
    ax_bar.set_title("Live values against the range of the throw", fontsize=10, loc="left")
    art["bars"] = []
    for k, (label, series, color) in enumerate(quantities):
        y = 1.75 * (len(quantities) - 1 - k)
        lo_v, hi_v = min(0.0, series.min()), series.max()
        span = (hi_v - lo_v) or 1.0
        zero = -lo_v / span
        ax_bar.barh(y, 1, height=0.34, color="#f1f3f4", edgecolor="none")
        bar = ax_bar.barh(y, 0, left=zero, height=0.34, color=color)[0]
        ax_bar.plot([zero, zero], [y - 0.22, y + 0.22], color="black", lw=1)
        ax_bar.text(0, y + 0.28, label, fontsize=8.5, va="bottom")
        value = ax_bar.text(1.0, y + 0.28, "", fontsize=10, fontweight="bold", ha="right", va="bottom", color=color)
        ax_bar.text(0, y - 0.24, f"{lo_v:.0f}", fontsize=7, color="#5f6368", va="top")
        ax_bar.text(1, y - 0.24, f"peak {hi_v:.0f}", fontsize=7, color="#5f6368", va="top", ha="right")
        art["bars"].append((bar, value, series, zero, span))
    return art


def draw_frame(d: dict, art: dict, i: int) -> None:
    art["markers"].set_data_3d(*d["markers"][i].T)
    for k, (arr, _, _) in d["lines"].items():
        art["lines"][k].set_data_3d(*arr[i].T)
    for k, (arr, _, _) in d["polys"].items():
        art["polys"][k].set_verts([arr[i]])
    ball = d["ball"][i]
    inside = all(lim[0] <= v <= lim[1] for v, lim in zip(ball, (art["ax3"].get_xlim(), art["ax3"].get_ylim(), art["ax3"].get_zlim())))
    art["ball"].set_data_3d([ball[0] if inside else numpy.nan], [ball[1] if inside else numpy.nan], [ball[2] if inside else numpy.nan])

    shoulder = d["shoulder"][i]
    set_arrow(art["moment"], shoulder, d["shoulder_moment"][i] * MOMENT_ARROW_M_PER_NM)
    set_arrow(art["ir"], shoulder, d["ir_vector"][i] * MOMENT_ARROW_M_PER_NM)

    plates = d["plates"]
    colors = []
    for p in range(len(plates["corners"])):
        up = plates["force"][p, i, 2]
        colors.append(plt.cm.YlOrRd(min(max(up, 0) / PLATE_FULL_SCALE_N, 1.0)) if up > 20 else (0.91, 0.92, 0.93, 0.9))
        cop = plates["cop"][p, i]
        base = numpy.array([cop[0], cop[1], plates["center"][p, 2]])
        set_arrow(art["grf"][p], base, plates["force"][p, i] * GRF_ARROW_M_PER_N if up > 20 else numpy.full(3, numpy.nan))
    art["plates"].set_facecolor(colors)

    t = d["t"][i]
    for cursor in art["cursors"]:
        cursor.set_xdata([t, t])
    rot = d["rotation"]
    for key, name in (("pelvis", "pelvis"), ("torso", "torso")):
        art["dots"][key].set_data([t], [rot[name][i]])
    art["dots"]["ir"].set_data([t], [d["table"].shoulder_ir_moment.iat[i]])
    art["dots"]["varus"].set_data([t], [d["table"].elbow_varus_moment.iat[i]])
    if art["ir_fill"] is not None:
        art["ir_fill"].remove()
    art["ir_fill"] = art["ax_mom"].fill_between(d["t"][:i + 1], d["table"].shoulder_ir_moment.to_numpy()[:i + 1], color=COLORS["ir"], alpha=0.18)

    for bar, text, series, zero, span in art["bars"]:
        value = series[i]
        bar.set_x(zero + min(value, 0) / span)
        bar.set_width(abs(value) / span)
        text.set_text(f"{value:.0f}")

    art["title"].set_text(f"Subject {d['subject']}   {d['mph']:.1f} mph fastball   {'right' if d['side'] == 'R' else 'left'}-handed   "
                          f"t = {t * 1000:+.0f} ms")
    label = ""
    if d["contact"] is not None and abs(i - d["contact"]) <= 6:
        label = "LEAD FOOT CONTACT"
    elif abs(i - d["release"]) <= 6:
        label = "BALL RELEASE"
    art["event"].set_text(label)


# --------------------------------------------------------------------------- outputs

def animate(d: dict, out: Path, speed: float, fps: int, dpi: int, view: str, t_start: float | None, t_end: float | None) -> None:
    art = build_figure(d, view)
    step = 360.0 * speed / fps  # data frames advanced per video frame
    first = 0 if t_start is None else int(round((d["release"] * d["dt"] + t_start) / d["dt"]))
    last = d["n"] - 1 if t_end is None else int(round((d["release"] * d["dt"] + t_end) / d["dt"]))
    frames = numpy.unique(numpy.clip(numpy.round(numpy.arange(max(first, 0), min(last, d["n"] - 1) + 1, step)).astype(int), 0, d["n"] - 1))
    animation = FuncAnimation(art["fig"], lambda i: draw_frame(d, art, int(i)), frames=frames, interval=1000 / fps)
    animation.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(art["fig"])
    print(f"wrote {out} ({len(frames)} frames, {speed:g}x slow-motion, {out.stat().st_size / 1e6:.1f} MB)")


def snapshot(d: dict, out: Path, frame: int, dpi: int, view: str) -> None:
    art = build_figure(d, view)
    draw_frame(d, art, frame)
    art["fig"].savefig(out, dpi=dpi)
    plt.close(art["fig"])
    print(f"wrote {out}")


def open_ktk_player(d: dict) -> None:
    """Interactive KTK 3D player: markers, joint centers and the force plates, with the
    shoulder moment and GRF as KTK vectors. Needs a display (opens a Qt window).
    """
    import kineticstoolkit as ktk
    trial = load_trial(A.H5_PATH, d["subject"], d["trial"])
    ts = A.trial_to_markers_ts(D.filtered_trial(trial, DRAW_CUTOFF_HZ))
    homogeneous = lambda xyz, w: numpy.concatenate([xyz, numpy.full((len(xyz), 1), w)], axis=1)
    side = d["side"]
    ts.data["ShoulderJC"] = homogeneous(d["shoulder"], 1.0)
    ts.data["ShoulderMoment"] = homogeneous(d["shoulder_moment"], 0.0)
    ts.data["IRMoment"] = homogeneous(d["ir_vector"], 0.0)
    for p, corners in enumerate(d["plates"]["corners"]):
        for c, xyz in enumerate(corners, start=1):
            ts.data[f"Plate{p + 1}_Corner{c}"] = homogeneous(numpy.tile(xyz, (d["n"], 1)), 1.0)
    links = [[f"{side}SHO", f"{side}ELB", f"{side}WRA", f"{side}FIN"], ["LASI", "RASI", "RPSI", "LPSI", "LASI"],
             ["LSHO", "CLAV", "RSHO"], ["C7", "T10"], ["CLAV", "STRN"],
             ["LFHD", "RFHD", "RBHD", "LBHD", "LFHD"]]
    for s in "LR":
        links += [[f"{s}KNE", f"{s}ANK", f"{s}HEE", f"{s}TOE", f"{s}ANK"], [f"{s}ELB", f"{s}WRA"]]
    player = ktk.Player(
        ts, up="z", anterior="x", azimuth=-1.5708 if side == "R" else 1.5708, elevation=0.2, track=True,
        interconnections={"Body": {"Links": links, "Color": (0.9, 0.9, 0.9)},
                          "Plates": {"Links": [[f"Plate{p}_Corner{c}", f"Plate{p}_Corner{c % 4 + 1}"] for p in (1, 2, 3) for c in (1, 2, 3, 4)],
                                     "Color": (0.5, 0.0, 1.0)}},
        vectors={"ShoulderMoment": {"Origin": "ShoulderJC", "Scale": MOMENT_ARROW_M_PER_NM, "Color": (0.6, 0.6, 0.6)},
                 "IRMoment": {"Origin": "ShoulderJC", "Scale": MOMENT_ARROW_M_PER_NM, "Color": (1.0, 0.2, 0.2)}},
        playback_speed=0.25, current_index=d["release"] - 120)
    player.play()
    input("KTK player is open -- press Enter here to close it.")
    player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject")
    parser.add_argument("trial", nargs="?", help="trial name (default: the subject's first motion trial)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--speed", type=float, default=0.25, help="playback speed, 1 = real time (default 0.25)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--view", choices=("arm", "glove", "front"), default="arm")
    parser.add_argument("--t-start", type=float, help="start, seconds relative to ball release (e.g. -1.0)")
    parser.add_argument("--t-end", type=float, help="end, seconds relative to ball release (e.g. 0.3)")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--player", action="store_true")
    args = parser.parse_args()

    trial = args.trial
    if trial is None:
        with h5py.File(A.H5_PATH, "r") as f:
            trial = next(t for t in sorted(f[args.subject]) if f[args.subject][t].attrs["trial_type"] == "motion")
    data = prepare(args.subject, trial)
    if args.player:
        open_ktk_player(data)
        return
    out = args.out or Path("results") / f"{trial}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot(data, out.with_suffix(".png"), data["release"], args.dpi * 2, args.view)
    animate(data, out, args.speed, args.fps, args.dpi, args.view, args.t_start, args.t_end)


if __name__ == "__main__":
    main()
