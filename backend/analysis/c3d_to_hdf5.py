"""Convert C3D motion capture files into a single HDF5 archive.

Expects one folder per subject, e.g.:

    data/c3d/000072/000072_002996_model.c3d               (static calibration)
    data/c3d/000072/000072_002996_76_229_001_FF_889.c3d    (motion trial)

Layout: one HDF5 group per subject, containing one subgroup per trial.

    /<subject_id>/<trial_name>/points          (nMarkers, nFrames, 3) float32, gzip
    /<subject_id>/<trial_name>/point_labels    (nMarkers,) variable-length UTF-8 strings
    /<subject_id>/<trial_name>/analogs         (nAnalogs, nAnalogFrames) float32, gzip
    /<subject_id>/<trial_name>/analog_labels   (nAnalogs,) variable-length UTF-8 strings
    /<subject_id>/<trial_name> attrs: source_file, subject_id, session_id, trial_type,
                          point_rate, analog_rate, n_markers, n_frames,
                          duration_seconds, analog_samples_per_frame
"""

import argparse
from pathlib import Path

import ezc3d
import h5py
import numpy

STR_DTYPE = h5py.string_dtype(encoding="utf-8")


def parse_trial_info(c3d_path: Path, c3d_dir: Path) -> tuple[str, str | None, str]:
    stem = c3d_path.stem
    tokens = stem.split("_")

    # Prefer the parent folder as subject_id (data/c3d/<subject_id>/<file>.c3d);
    # fall back to the filename's leading token for files sitting directly in c3d_dir.
    subject_id = c3d_path.parent.name if c3d_path.parent != c3d_dir else tokens[0]
    session_id = tokens[1] if len(tokens) > 1 else None
    trial_type = "static" if "model" in stem.lower() else "motion"

    return subject_id, session_id, trial_type


def convert_trial(c3d_path: Path, c3d_dir: Path, h5_file: h5py.File, overwrite: bool = False) -> str:
    trial_name = c3d_path.stem
    subject_id, session_id, trial_type = parse_trial_info(c3d_path, c3d_dir)
    group_path = f"{subject_id}/{trial_name}"

    if group_path in h5_file:
        if not overwrite:
            return group_path
        del h5_file[group_path]

    c3d = ezc3d.c3d(str(c3d_path))
    group = h5_file.require_group(subject_id).create_group(trial_name)

    points = numpy.asarray(c3d["data"]["points"][:3], dtype="float32")
    points = numpy.moveaxis(points, 0, -1)  # (nMarkers, nFrames, 3)
    group.create_dataset("points", data=points, compression="gzip")

    point_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    group.create_dataset("point_labels", data=point_labels, dtype=STR_DTYPE)

    analogs = numpy.asarray(c3d["data"]["analogs"][0], dtype="float32")
    group.create_dataset("analogs", data=analogs, compression="gzip")

    analog_labels = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
    group.create_dataset("analog_labels", data=analog_labels, dtype=STR_DTYPE)

    point_rate = c3d["header"]["points"]["frame_rate"]
    analog_rate = c3d["header"]["analogs"]["frame_rate"]
    n_frames = points.shape[1]

    group.attrs["source_file"] = str(c3d_path)
    group.attrs["subject_id"] = subject_id
    group.attrs["session_id"] = session_id or ""
    group.attrs["trial_type"] = trial_type
    group.attrs["point_rate"] = point_rate
    group.attrs["analog_rate"] = analog_rate
    group.attrs["n_markers"] = points.shape[0]
    group.attrs["n_frames"] = n_frames
    group.attrs["duration_seconds"] = n_frames / point_rate if point_rate else 0.0

    samples_per_frame = analogs.shape[1] / n_frames if n_frames else 0.0
    group.attrs["analog_samples_per_frame"] = samples_per_frame
    if analogs.shape[1] and abs(samples_per_frame - round(samples_per_frame)) > 1e-6:
        print(f"[warn] {c3d_path}: analog samples ({analogs.shape[1]}) don't divide "
              f"evenly into point frames ({n_frames}) -> {samples_per_frame:.4f} per frame")

    return group_path


def build_master_hdf5(c3d_dir: Path, h5_path: Path, overwrite: bool = False) -> None:
    c3d_files = sorted(c3d_dir.rglob("*.c3d"))
    if not c3d_files:
        raise FileNotFoundError(f"No .c3d files found under {c3d_dir}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with h5py.File(h5_path, "a") as h5_file:
        for c3d_path in c3d_files:
            try:
                group_path = convert_trial(c3d_path, c3d_dir, h5_file, overwrite=overwrite)
                print(f"[ok] {c3d_path} -> /{group_path}")
            except Exception as exc:
                failures.append((c3d_path, exc))
                print(f"[fail] {c3d_path}: {exc}")

    if failures:
        print(f"\n{len(failures)} of {len(c3d_files)} files failed to convert:")
        for c3d_path, exc in failures:
            print(f"  {c3d_path}: {exc}")


def load_trial(h5_path: Path, subject_id: str, trial_name: str) -> dict:
    with h5py.File(h5_path, "r") as h5_file:
        group = h5_file[f"{subject_id}/{trial_name}"]
        return {
            "points": group["points"][()],
            "point_labels": [label.decode() if isinstance(label, bytes) else label
                              for label in group["point_labels"][()]],
            "analogs": group["analogs"][()],
            "analog_labels": [label.decode() if isinstance(label, bytes) else label
                               for label in group["analog_labels"][()]],
            "subject_id": group.attrs["subject_id"],
            "session_id": group.attrs["session_id"],
            "trial_type": group.attrs["trial_type"],
            "point_rate": group.attrs["point_rate"],
            "analog_rate": group.attrs["analog_rate"],
            "n_markers": group.attrs["n_markers"],
            "n_frames": group.attrs["n_frames"],
            "duration_seconds": group.attrs["duration_seconds"],
            "analog_samples_per_frame": group.attrs["analog_samples_per_frame"],
        }


def get_marker(trial: dict, marker_name: str) -> numpy.ndarray:
    """Return one marker's (nFrames, 3) trajectory from a trial loaded via load_trial()."""
    try:
        index = trial["point_labels"].index(marker_name)
    except ValueError:
        raise KeyError(f"Marker '{marker_name}' not found. Available: {trial['point_labels']}")
    return trial["points"][index]


def get_analog(trial: dict, channel_name: str) -> numpy.ndarray:
    """Return one analog channel's (nAnalogFrames,) series from a trial loaded via load_trial()."""
    try:
        index = trial["analog_labels"].index(channel_name)
    except ValueError:
        raise KeyError(f"Analog channel '{channel_name}' not found. Available: {trial['analog_labels']}")
    return trial["analogs"][index]


def list_archive(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        subject_ids = sorted(h5_file.keys())
        for subject_id in subject_ids:
            trial_names = sorted(h5_file[subject_id].keys())
            print(f"{subject_id} ({len(trial_names)} trials)")
            for trial_name in trial_names:
                attrs = h5_file[subject_id][trial_name].attrs
                print(f"  {trial_name}  [{attrs['trial_type']}, "
                      f"{attrs['n_frames']} frames, {attrs['duration_seconds']:.2f}s]")
        print(f"\n{len(subject_ids)} subjects total")


def describe_trial(h5_path: Path, subject_id: str, trial_name: str) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        group = h5_file[f"{subject_id}/{trial_name}"]

        print(f"{subject_id}/{trial_name}")
        for key, value in group.attrs.items():
            print(f"  {key}: {value}")

        markers = [label.decode() for label in group["point_labels"][()]]
        print(f"  markers ({len(markers)}): {markers}")

        channels = [label.decode() for label in group["analog_labels"][()]]
        print(f"  analog channels ({len(channels)}): {channels}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert a directory of .c3d files into an HDF5 archive")
    convert_parser.add_argument("c3d_dir", type=Path, help="Directory to scan recursively for .c3d files")
    convert_parser.add_argument("h5_path", type=Path, help="Path to the master .h5 file to create/append to")
    convert_parser.add_argument("--overwrite", action="store_true", help="Re-convert trials that already exist")

    list_parser = subparsers.add_parser("list", help="List every subject/trial in an HDF5 archive")
    list_parser.add_argument("h5_path", type=Path)

    describe_parser = subparsers.add_parser("describe", help="Show markers, forces, and metadata for one trial")
    describe_parser.add_argument("h5_path", type=Path)
    describe_parser.add_argument("subject_id")
    describe_parser.add_argument("trial_name")

    args = parser.parse_args()

    if args.command == "convert":
        build_master_hdf5(args.c3d_dir, args.h5_path, overwrite=args.overwrite)
    elif args.command == "list":
        list_archive(args.h5_path)
    elif args.command == "describe":
        describe_trial(args.h5_path, args.subject_id, args.trial_name)


if __name__ == "__main__":
    main()
