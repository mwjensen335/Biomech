"""Run the whole-body inverse dynamics on every motion trial and write the results out.

Outputs (in --out-dir, default ./results):
    summary.csv         one row per pitch: identifiers, events, headline peaks, model-quality
                        flags, and (if the OpenBiomechanics metrics file is found) Driveline's
                        published peaks for the same pitch to compare against
    arm_timeseries.csv  long format, one row per pitch per frame: the shoulder and elbow
                        loads an animation needs (see the column notes below)
    kinetics.h5         every joint's force and moment (lab frame), per pitch, plus the
                        segment-frame components used for the arm columns

Conventions -- moments in N.m, forces in N, time in s (marker rate, 360 Hz):
  * A joint's load is what the PROXIMAL segment applies to the DISTAL one (shoulder: what the
    trunk applies to the upper arm; elbow: what the upper arm applies to the forearm+hand).
  * Lab frame: +X anterior (toward home plate), +Y mediolateral (left), +Z up.
  * Upper-arm frame (arm_timeseries "_ua_" columns): x anteroposterior, y along the humerus
    pointing toward the shoulder, z mediolateral.
  * shoulder_ir_moment is the moment about the humerus long axis, positive = internal rotation.
  * elbow_varus_moment is about the ISB floating axis, positive = the varus moment that
    resists the valgus stress of the throw.
  * shoulder_distraction_force is the force along the humerus toward the shoulder (the joint
    holding the arm in), positive = distraction.
  * ball_on_hand is 1 until the fingertip-speed-peak release estimate, then 0 (the ball's
    load on the hand is only modelled while it is in the hand).
Moments are NOT normalised; body mass and height are in summary.csv.

Run:  python export_results.py [--out-dir DIR] [--obp-dir DIR]
"""

import argparse
from pathlib import Path

import h5py
import numpy
import pandas as pd

import analysis as A
import dynamics as D
import inverse_dynamics as ID
from c3d_to_hdf5 import load_trial
from validate_against_obp import DEFAULT_OBP_DIR

# a pitch whose trunk closure error (RMS force residual / peak hip force) exceeds this, or
# with many unmatched plate frames, is flagged in summary.csv
FLAG_CLOSURE_FRACTION = 0.40
FLAG_UNMATCHED_FRAMES = 300


def lead_foot_contact_frame(result: dict) -> int | None:
    """First frame, after the lead knee's highest point (peak of the leg lift), at which the
    lead foot carries plate load: the landing. None if it never does.
    """
    lead = "L" if result["throwing_side"] == "R" else "R"
    lifted = int(result["segments"][f"{lead}_Shank"].proximal[:, 2].argmax())
    owner = result["contact"]["owner"] == (0 if lead == "L" else 1)
    loaded = numpy.flatnonzero(owner.any(axis=0))
    loaded = loaded[loaded > lifted]
    return int(loaded[0]) if len(loaded) else None


def arm_frame_table(result: dict) -> pd.DataFrame:
    """Per-frame shoulder/elbow loads for the throwing arm (columns documented above)."""
    side = result["throwing_side"]
    dt, n = result["dt"], len(result["ts"].time)
    upper_arm = result["segments"][f"{side}_UpperArm"]
    shoulder, elbow = result["joints"][f"{side}_UpperArm"], result["joints"][f"{side}_Forearm+Hand"]

    shoulder_moment_ua = D.to_local(shoulder["moment"], upper_arm.rotation)
    shoulder_force_ua = D.to_local(shoulder["force"], upper_arm.rotation)
    elbow_moment_ua = D.to_local(elbow["moment"], upper_arm.rotation)
    columns = {
        "frame": numpy.arange(n),
        "time_s": numpy.arange(n) * dt,
        "ball_on_hand": (numpy.arange(n) <= result["release_frame"]).astype(int),
        "shoulder_ir_moment": ID.humerus_long_axis_moment(result, side),
        "elbow_varus_moment": ID.elbow_varus_moment(result, side),
        "shoulder_distraction_force": shoulder_force_ua[:, 1],
        "shoulder_moment_norm": numpy.linalg.norm(shoulder["moment"], axis=1),
        "elbow_moment_norm": numpy.linalg.norm(elbow["moment"], axis=1),
        "shoulder_force_norm": numpy.linalg.norm(shoulder["force"], axis=1),
        "elbow_force_norm": numpy.linalg.norm(elbow["force"], axis=1),
    }
    for label, vec in (("shoulder_moment_lab", shoulder["moment"]), ("shoulder_force_lab", shoulder["force"]),
                       ("elbow_moment_lab", elbow["moment"]), ("elbow_force_lab", elbow["force"]),
                       ("shoulder_moment_ua", shoulder_moment_ua), ("shoulder_force_ua", shoulder_force_ua),
                       ("elbow_moment_ua", elbow_moment_ua)):
        for i, axis in enumerate("xyz"):
            columns[f"{label}_{axis}"] = vec[:, i]
    return pd.DataFrame(columns)


def summary_row(subject: str, trial: str, result: dict, meta_row: pd.Series, poi_row) -> dict:
    side, dt = result["throwing_side"], result["dt"]
    frames = arm_frame_table(result)
    release = result["release_frame"]
    contact = lead_foot_contact_frame(result)
    # peaks are taken from lead-foot contact to shortly after release: the throw itself
    start = contact if contact is not None else 0
    window = frames[(frames.frame >= start) & (frames.frame <= release + int(0.1 / dt))]
    joints = result["joints"]
    peak = lambda key, kind: float(numpy.linalg.norm(joints[key][kind], axis=1)[start:release + int(0.1 / dt)].max())
    rear, lead = side, ("L" if side == "R" else "R")
    residual = result["trunk_residual"]["force"]
    hip_peak = max(numpy.linalg.norm(joints[f"{s}_Thigh"]["force"], axis=1)[20:-20].max() for s in "LR")
    closure = float(numpy.sqrt((residual[20:-20] ** 2).sum(axis=1).mean()) / hip_peak)
    unmatched = int(result["contact"]["unmatched"].sum())
    row = {
        "subject": subject, "trial": trial, "session_pitch": meta_row.session_pitch,
        "throwing_hand": side, "pitch_speed_mph": float(meta_row.pitch_speed_mph),
        "body_mass_kg": float(meta_row.session_mass_kg), "height_m": float(meta_row.session_height_m),
        "release_time_s": release * dt,
        "lead_foot_contact_time_s": None if contact is None else contact * dt,
        "shoulder_ir_moment_peak_Nm": float(window.shoulder_ir_moment.abs().max()),
        "shoulder_ir_moment_peak_time_s": float(window.loc[window.shoulder_ir_moment.abs().idxmax(), "time_s"]),
        "elbow_varus_moment_peak_Nm": float(window.elbow_varus_moment.abs().max()),
        "elbow_varus_moment_peak_time_s": float(window.loc[window.elbow_varus_moment.abs().idxmax(), "time_s"]),
        "shoulder_distraction_force_peak_N": float(window.shoulder_distraction_force.max()),
        "shoulder_moment_norm_peak_Nm": float(window.shoulder_moment_norm.max()),
        "elbow_moment_norm_peak_Nm": float(window.elbow_moment_norm.max()),
        "shoulder_force_norm_peak_N": float(window.shoulder_force_norm.max()),
        "elbow_force_norm_peak_N": float(window.elbow_force_norm.max()),
        "rear_hip_moment_peak_Nm": peak(f"{rear}_Thigh", "moment"), "rear_knee_moment_peak_Nm": peak(f"{rear}_Shank", "moment"),
        "rear_ankle_moment_peak_Nm": peak(f"{rear}_Foot", "moment"),
        "lead_hip_moment_peak_Nm": peak(f"{lead}_Thigh", "moment"), "lead_knee_moment_peak_Nm": peak(f"{lead}_Shank", "moment"),
        "lead_ankle_moment_peak_Nm": peak(f"{lead}_Foot", "moment"),
        "lumbar_moment_peak_Nm": peak("Lumbar", "moment"),
        "trunk_closure_residual_fraction": closure, "unmatched_plate_frames": unmatched,
        "flag": ";".join(filter(None, [
            "closure" if closure > FLAG_CLOSURE_FRACTION else "",
            "plates" if unmatched > FLAG_UNMATCHED_FRAMES else "",
            "no_lead_contact" if contact is None else ""])),
    }
    if poi_row is not None:
        row["obp_shoulder_ir_moment_Nm"] = float(poi_row.shoulder_internal_rotation_moment)
        row["obp_elbow_varus_moment_Nm"] = float(poi_row.elbow_varus_moment)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--obp-dir", type=Path, default=DEFAULT_OBP_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(A.METADATA_CSV_PATH)
    meta["trial"] = meta["filename_new"].str.replace(".c3d", "", regex=False)
    meta = meta.set_index("trial")
    poi_path = args.obp_dir / "poi" / "poi_metrics.csv"
    poi = pd.read_csv(poi_path).set_index("session_pitch") if poi_path.exists() else None

    with h5py.File(A.H5_PATH, "r") as f:
        trials = [(s, t) for s in sorted(f) for t in sorted(f[s]) if f[s][t].attrs["trial_type"] == "motion"]

    rows, tables = [], []
    with h5py.File(args.out_dir / "kinetics.h5", "w") as out:
        for subject, trial in trials:
            try:
                result = ID.solve_full_body(load_trial(A.H5_PATH, subject, trial), trial)
            except Exception as exc:
                print(f"[fail] {trial}: {exc}")
                continue
            meta_row = meta.loc[trial]
            poi_row = poi.loc[meta_row.session_pitch] if poi is not None and meta_row.session_pitch in poi.index else None
            rows.append(summary_row(subject, trial, result, meta_row, poi_row))
            table = arm_frame_table(result)
            table.insert(0, "trial", trial)
            table.insert(0, "subject", subject)
            tables.append(table)

            group = out.create_group(f"{subject}/{trial}")
            group.attrs.update({"throwing_hand": result["throwing_side"], "release_frame": result["release_frame"],
                                "dt": result["dt"], "cutoff_hz": result["cutoff_hz"], "arm_cutoff_hz": result["arm_cutoff_hz"]})
            for key, load in result["joints"].items():
                group.create_dataset(f"{key}/force", data=load["force"], compression="gzip")
                group.create_dataset(f"{key}/moment", data=load["moment"], compression="gzip")
            group.create_dataset("trunk_closure_force_residual", data=result["trunk_residual"]["force"], compression="gzip")
            group.create_dataset("trunk_closure_moment_residual", data=result["trunk_residual"]["moment"], compression="gzip")
            print(f"[ok] {trial}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False, float_format="%.4f")
    pd.concat(tables).to_csv(args.out_dir / "arm_timeseries.csv", index=False, float_format="%.4f")
    print(f"\nwrote {len(summary)} pitches to {args.out_dir.resolve()}: summary.csv, arm_timeseries.csv, kinetics.h5")
    print(f"flagged: {int((summary.flag != '').sum())} ({', '.join(summary.loc[summary.flag != '', 'trial'].str[-16:])})")


if __name__ == "__main__":
    main()
