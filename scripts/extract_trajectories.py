#!/usr/bin/env python3
"""
Extract per-player pitch trajectories from SoccerNet-GSR Labels-GameState.json files.

CPU only. Reads the JSON labels directly from the split zips (no need to unzip
the ~30GB of images), and writes one tidy table per split:

    split, sequence, frame, track_id, role, team, jersey,
    x_pitch, y_pitch,                 # bottom-middle of the player on the pitch (meters)
    bbox_x, bbox_y, bbox_w, bbox_h    # image-space bbox (pixels)

Scope: off-screen periods are NOT modeled. In the GSR labels a player is only
annotated while visible, so long gaps inside a track mean the player left the
camera view. Each track is therefore cut into on-screen segments wherever the
gap exceeds --max-gap frames (default 25 = 1s). Extra output columns:

    segment_id   index of the on-screen segment within the track (0, 1, ...)
    segment_uid  unique id "<sequence>_<track_id>_<segment_id>", use this as the
                 trajectory key downstream
    gap_before   short gap (<= max_gap missing frames) right before this row,
                 0 if none; these are the real occlusions worth imputing

Also writes a gap summary per split (all gaps, off-screen gaps that were cut,
short gaps kept, segment lengths).

Usage:
    python scripts/extract_trajectories.py \
        --data-dir ~/data/SoccerNetGS/gamestate-2024 \
        --out-dir  ~/data/processed \
        --splits train valid test

    # optional: keep the ball / referees too
    python scripts/extract_trajectories.py ... --roles player goalkeeper referee ball
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROLES = ["player", "goalkeeper"]


def frame_from_image(img: dict, fallback: int) -> int:
    """Get a 1-based frame index from an image entry (file_name like '000123.jpg')."""
    name = img.get("file_name", "")
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else fallback


def parse_labels(labels: dict, split: str, sequence: str, roles: set) -> list[dict]:
    # image_id -> frame index
    images = labels.get("images", [])
    img2frame = {
        str(img["image_id"]): frame_from_image(img, i + 1) for i, img in enumerate(images)
    }

    rows = []
    for ann in labels.get("annotations", []):
        # skip pitch-line and camera annotations, keep only objects (people / ball)
        if ann.get("supercategory", "object") != "object":
            continue

        attrs = ann.get("attributes") or {}
        role = attrs.get("role")
        if roles and role not in roles:
            continue

        frame = img2frame.get(str(ann.get("image_id")))
        if frame is None:
            continue

        pitch = ann.get("bbox_pitch") or {}
        img_box = ann.get("bbox_image") or {}

        rows.append(
            {
                "split": split,
                "sequence": sequence,
                "frame": frame,
                "track_id": ann.get("track_id"),
                "role": role,
                "team": attrs.get("team"),
                "jersey": attrs.get("jersey"),
                "x_pitch": pitch.get("x_bottom_middle"),
                "y_pitch": pitch.get("y_bottom_middle"),
                "bbox_x": img_box.get("x"),
                "bbox_y": img_box.get("y"),
                "bbox_w": img_box.get("w"),
                "bbox_h": img_box.get("h"),
            }
        )
    return rows


def iter_label_files_s3(s3_zip: str):
    """
    Same as iter_label_files but reads a zip that lives on S3 without downloading it.
    zipfile only seeks to the central directory and the label JSON entries, so only a
    small fraction of the (multi-GB) zip is actually transferred. Needs `pip install s3fs`.
    """
    import s3fs

    fs = s3fs.S3FileSystem()
    with fs.open(s3_zip, "rb", block_size=8 * 1024 * 1024) as f:
        yield from iter_label_files(f)


def iter_label_files(split_zip):
    """Yield (sequence_name, labels_dict) for every Labels-GameState.json in a zip (path or file object)."""
    with zipfile.ZipFile(split_zip) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith("Labels-GameState.json"))
        for m in members:
            sequence = Path(m).parent.name  # e.g. SNGS-021
            with zf.open(m) as f:
                yield sequence, json.load(f)


def iter_label_files_dir(split_dir: Path):
    """Same as above but for an already-extracted split folder."""
    for p in sorted(split_dir.rglob("Labels-GameState.json")):
        with open(p) as f:
            yield p.parent.name, json.load(f)


def split_into_segments(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    Split each (sequence, track_id) into on-screen segments.

    A gap of <= max_gap missing frames is treated as a short occlusion and stays
    inside the segment (a candidate for imputation). A longer gap is treated as
    the player leaving the camera view: the track is cut there and the off-screen
    part is NOT modeled.
    """
    df = df.drop_duplicates(["sequence", "track_id", "frame"])
    df = df.sort_values(["sequence", "track_id", "frame"]).reset_index(drop=True)

    missing = df.groupby(["sequence", "track_id"])["frame"].diff().fillna(1) - 1
    new_segment = missing > max_gap
    df["segment_id"] = new_segment.groupby([df["sequence"], df["track_id"]]).cumsum().astype(int)
    df["segment_uid"] = (
        df["sequence"] + "_" + df["track_id"].astype(str) + "_" + df["segment_id"].astype(str)
    )
    df["gap_before"] = missing.where(~new_segment, 0).astype(int)  # short gap right before this row
    return df


def drop_short_segments(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    lengths = df.groupby("segment_uid")["frame"].transform("size")
    return df[lengths >= min_len].reset_index(drop=True)


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": round(float(x.mean()), 1),
        "median": float(np.median(x)),
        "p90": round(float(np.percentile(x, 90)), 1),
        "max": int(x.max()),
    }


def gap_summary(raw: pd.DataFrame, seg: pd.DataFrame, max_gap: int, min_len: int, fps: int) -> dict:
    """Gap statistics: short (imputable) gaps vs off-screen gaps, plus segment lengths."""
    all_gaps = []
    for _, g in raw.groupby(["sequence", "track_id"]):
        d = np.diff(np.sort(g["frame"].unique()))
        all_gaps.extend((d[d > 1] - 1).tolist())
    all_gaps = np.array(all_gaps, dtype=int)

    short_gaps = seg.loc[seg["gap_before"] > 0, "gap_before"].to_numpy()
    seg_lengths = seg.groupby("segment_uid")["frame"].size().to_numpy()

    return {
        "settings": {"max_gap_frames": max_gap, "min_segment_frames": min_len, "fps": fps},
        "n_sequences": int(raw["sequence"].nunique()),
        "n_tracks": int(raw.groupby(["sequence", "track_id"]).ngroups),
        "n_rows_raw": int(len(raw)),
        "n_rows_kept": int(len(seg)),
        "gaps_all": _stats(all_gaps),
        "gaps_offscreen_cut": int((all_gaps > max_gap).sum()),
        "gaps_short_imputable": _stats(short_gaps),
        "segments": _stats(seg_lengths),
        "missing_pitch_coords": int(seg["x_pitch"].isna().sum()),
        "missing_jersey": int(seg["jersey"].isna().sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Folder containing train.zip/valid.zip/... (or extracted split folders). "
                         "Can also be an S3 prefix, e.g. s3://cmu-soccer-gsr-data/SoccerNetGS/gamestate-2024")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--roles", nargs="+", default=DEFAULT_ROLES)
    ap.add_argument("--max-gap", type=int, default=25,
                    help="Gaps longer than this many missing frames are treated as off-screen "
                         "and cut the track into separate segments (default 25 = 1s at 25fps)")
    ap.add_argument("--min-segment-len", type=int, default=25,
                    help="Drop on-screen segments shorter than this many frames (default 25)")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    on_s3 = args.data_dir.startswith("s3://")
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = set(args.roles)

    for split in args.splits:
        if on_s3:
            source = iter_label_files_s3(f"{args.data_dir.rstrip('/')}/{split}.zip")
        else:
            data_dir = Path(args.data_dir).expanduser()
            split_zip, split_dir = data_dir / f"{split}.zip", data_dir / split
        if on_s3:
            pass
        elif split_dir.is_dir():
            source = iter_label_files_dir(split_dir)
        elif split_zip.exists():
            source = iter_label_files(split_zip)
        else:
            print(f"[skip] {split}: no {split_zip.name} or {split}/ found")
            continue

        rows = []
        for i, (sequence, labels) in enumerate(source, 1):
            rows.extend(parse_labels(labels, split, sequence, roles))
            print(f"\r[{split}] parsed {i} sequences", end="", flush=True)
        print()

        if not rows:
            print(f"[warn] {split}: no rows extracted (check roles / file layout)")
            continue

        raw = pd.DataFrame(rows)
        seg = split_into_segments(raw, args.max_gap)
        seg = drop_short_segments(seg, args.min_segment_len)

        out_csv = out_dir / f"trajectories_{split}.csv"
        seg.to_csv(out_csv, index=False)

        s = gap_summary(raw, seg, args.max_gap, args.min_segment_len, args.fps)
        with open(out_dir / f"gap_summary_{split}.json", "w") as f:
            json.dump(s, f, indent=2)

        print(f"[{split}] {s['n_tracks']} tracks -> {s['segments']['count']} on-screen segments, "
              f"{s['n_rows_kept']:,}/{s['n_rows_raw']:,} rows kept -> {out_csv}")
        print(f"[{split}] cut {s['gaps_offscreen_cut']} off-screen gaps (> {args.max_gap} frames); "
              f"{s['gaps_short_imputable']['count']} short gaps kept inside segments")


if __name__ == "__main__":
    main()
