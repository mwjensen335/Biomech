"""Validate inverse_dynamics.py against an independent reference: the joint kinetics the
OpenBiomechanics Project (Driveline) published for the same pitches.

All 60 trials in master.h5 are in that dataset (matched by session_pitch through the
metadata CSV). Its forces/moments are internal joint loads in the proximal segment's frame,
so this compares frame-independent vector NORMS over time, plus the two headline peaks
it tabulates (shoulder internal-rotation moment, elbow varus moment) against ours.

Run:  python validate_against_obp.py [--obp-dir DIR] [--sweep] [--out results.csv]
Needs the OBP repo (https://github.com/drivelineresearch/openbiomechanics) on disk.
"""

import argparse
from pathlib import Path

import h5py
import numpy
import pandas as pd

import analysis as A
import inverse_dynamics as ID
from c3d_to_hdf5 import load_trial

DEFAULT_OBP_DIR = Path(r"C:\Users\mikey\Thesis_Work\openbiomechanics-main\openbiomechanics-main\baseball_pitching\data")

# our joint key (side is throwing/non-throwing -> rear/lead leg) -> OBP column prefix
ARM_JOINTS = {"shoulder": ("UpperArm", "shoulder_upper_arm"), "elbow": ("Forearm+Hand", "elbow")}
LEG_JOINTS = {"hip": ("Thigh", "{leg}_hip_{leg}_thigh"), "knee": ("Shank", "{leg}_knee"), "ankle": ("Foot", "{leg}_ankle")}


def load_obp(obp_dir: Path, session_pitches: set) -> tuple:
    """(forces_moments rows for our pitches, poi_metrics indexed by session_pitch)."""
    chunks = pd.read_csv(obp_dir / "poi" / "full_sig" / "forces_moments" / "forces_moments.csv", chunksize=200_000)
    fm = pd.concat([c[c.session_pitch.isin(session_pitches)] for c in chunks])
    poi = pd.read_csv(obp_dir / "poi" / "poi_metrics.csv").set_index("session_pitch")
    return fm, poi


def norm_series(group: pd.DataFrame, prefix: str, kind: str) -> numpy.ndarray:
    return numpy.linalg.norm(group[[f"{prefix}_{kind}_{a}" for a in "xyz"]].to_numpy(), axis=1)


def compare(result: dict, group: pd.DataFrame, poi_row: pd.Series) -> dict:
    """Metrics for one pitch. corr/ratio use the window from 50 ms before foot contact to
    100 ms after ball release; ratio = our peak / their peak of the norm over that window.
    """
    dt = result["dt"]
    t_me = numpy.arange(len(result["ts"].time)) * dt
    t = group.time.to_numpy()
    window = (t > group.fp_10_time.iloc[0] - 0.05) & (t < group.BR_time.iloc[0] + 0.1)
    throw = result["throwing_side"]
    rear = "R" if poi_row.p_throws == "R" else "L"  # rear (drive) leg = throwing-side leg
    lead = "L" if rear == "R" else "R"
    out = {"release_error_frames": result["release_frame"] - group.BR_time.iloc[0] * 360.0}

    def one(name, mine_j, prefix_kind):
        for kind, key in (("moment", "moment"), ("force", "force")):
            theirs = norm_series(group, prefix_kind, kind)
            mine = numpy.interp(t, t_me, numpy.linalg.norm(result["joints"][mine_j][key], axis=1))
            ok = window & ~numpy.isnan(theirs)
            out[f"{name}_{kind}_corr"] = numpy.corrcoef(mine[ok], theirs[ok])[0, 1]
            out[f"{name}_{kind}_peak_ratio"] = mine[ok].max() / theirs[ok].max()

    for name, (seg, prefix) in ARM_JOINTS.items():
        one(name, f"{throw}_{seg}", prefix)
    for leg_name, side in (("rear", rear), ("lead", lead)):
        for name, (seg, prefix) in LEG_JOINTS.items():
            one(f"{leg_name}_{name}", f"{side}_{seg}", prefix.format(leg=leg_name))

    # headline peaks vs the summary table
    ir = ID.humerus_long_axis_moment(result, throw)
    varus = ID.elbow_varus_moment(result, throw)
    out["shoulder_ir_peak"], out["shoulder_ir_peak_obp"] = float(numpy.abs(ir[numpy.interp(t_me, t, window) > 0.5]).max()), poi_row.shoulder_internal_rotation_moment
    out["elbow_varus_peak"], out["elbow_varus_peak_obp"] = float(numpy.abs(varus[numpy.interp(t_me, t, window) > 0.5]).max()), poi_row.elbow_varus_moment
    return out


def run(obp_dir: Path, **solver_kwargs) -> pd.DataFrame:
    meta = pd.read_csv(A.METADATA_CSV_PATH)
    meta["trial"] = meta["filename_new"].str.replace(".c3d", "", regex=False)
    session_of = dict(zip(meta.trial, meta.session_pitch))
    with h5py.File(A.H5_PATH, "r") as f:
        trials = [(s, t) for s in sorted(f) for t in sorted(f[s]) if f[s][t].attrs["trial_type"] == "motion"]
    fm, poi = load_obp(obp_dir, {session_of[t] for _, t in trials})
    groups = {k: v.sort_values("time").reset_index(drop=True) for k, v in fm.groupby("session_pitch")}

    rows = []
    for subject, trial in trials:
        group = groups.get(session_of[trial])
        if group is None or group[["fp_10_time", "BR_time"]].iloc[0].isna().any():
            continue  # OBP has no event times for this pitch
        try:
            result = ID.solve_full_body(load_trial(A.H5_PATH, subject, trial), trial, **solver_kwargs)
            rows.append({"subject": subject, "trial": trial, **compare(result, group, poi.loc[session_of[trial]])})
        except Exception as exc:
            print(f"[fail] {trial}: {exc}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    print(f"{len(df)} pitches compared")
    print(f"release error vs ball release: {df.release_error_frames.mean():+.1f} +/- {df.release_error_frames.std():.1f} frames")
    print(f"{'joint':14s} {'moment corr':>11s} {'moment peak ratio':>18s} {'force corr':>11s} {'force peak ratio':>17s}")
    for name in ("shoulder", "elbow", "rear_hip", "rear_knee", "rear_ankle", "lead_hip", "lead_knee", "lead_ankle"):
        print(f"{name:14s} {df[f'{name}_moment_corr'].mean():11.2f} "
              f"{df[f'{name}_moment_peak_ratio'].mean():10.2f} +/- {df[f'{name}_moment_peak_ratio'].std():4.2f} "
              f"{df[f'{name}_force_corr'].mean():11.2f} {df[f'{name}_force_peak_ratio'].mean():9.2f} +/- {df[f'{name}_force_peak_ratio'].std():4.2f}")
    for label, mine, theirs in (("shoulder internal-rotation moment peak", "shoulder_ir_peak", "shoulder_ir_peak_obp"),
                                ("elbow varus moment peak", "elbow_varus_peak", "elbow_varus_peak_obp")):
        ratio = df[mine] / df[theirs]
        print(f"{label}: ours {df[mine].mean():.0f} vs OBP {df[theirs].mean():.0f} N.m; ratio {ratio.mean():.2f} +/- {ratio.std():.2f}; "
              f"across-pitch r = {numpy.corrcoef(df[mine], df[theirs])[0, 1]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obp-dir", type=Path, default=DEFAULT_OBP_DIR)
    parser.add_argument("--sweep", action="store_true", help="repeat for several body/arm cutoffs")
    parser.add_argument("--out", type=Path, help="write per-pitch metrics to this CSV")
    args = parser.parse_args()
    if args.sweep:
        for body in (8, 12, 15, 20):
            for arm in (18, 22, 26):
                print(f"\n=== body cutoff {body} Hz, arm cutoff {arm} Hz ===")
                summarize(run(args.obp_dir, cutoff_hz=body, arm_cutoff_hz=arm))
        return
    df = run(args.obp_dir)
    summarize(df)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
