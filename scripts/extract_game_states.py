#!/usr/bin/env python3
"""
Extract full game states from SoccerNet-GSR labels: every labeled person AND the ball,
per frame, with pitch positions. Needed for ball-carrier, pass and dribble labels.

    # local zips or extracted split folders
    python scripts/extract_game_states.py --data-dir ~/data/SoccerNetGS/gamestate-2024 \
        --out-dir data/processed --splits valid
    # straight from S3 (e.g. in AWS CloudShell; needs `pip install s3fs`)
    python scripts/extract_game_states.py --data-dir s3://cmu-soccer-gsr-data/SoccerNetGS/gamestate-2024 \
        --out-dir processed --splits train valid test

Writes game_states_<split>.csv.gz with columns:
    sequence, frame, track_id, role, team, jersey, x, y, bx, by, bw, bh
(frame is 1-based as in the image file names; x, y are pitch metres, empty if unknown;
bx..bh is the image box in pixels).
"""
import argparse
import csv
import gzip
import json
import re
import zipfile
from pathlib import Path


def iter_labels(data_dir, split):
    if data_dir.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        f = fs.open(f"{data_dir.rstrip('/')}/{split}.zip", "rb", block_size=8 * 1024 * 1024)
        zf = zipfile.ZipFile(f)
    else:
        d = Path(data_dir).expanduser()
        if (d / split).is_dir():
            for p in sorted((d / split).rglob("Labels-GameState.json")):
                yield p.parent.name, json.loads(p.read_text())
            return
        zf = zipfile.ZipFile(d / f"{split}.zip")
    for m in sorted(n for n in zf.namelist() if n.endswith("Labels-GameState.json")):
        with zf.open(m) as fh:
            yield Path(m).parent.name, json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        path = a.out_dir / f"game_states_{split}.csv.gz"
        n_rows = n_seq = 0
        with gzip.open(path, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "frame", "track_id", "role", "team", "jersey", "x", "y", "bx", "by", "bw", "bh"])
            for seq, lab in iter_labels(a.data_dir, split):
                n_seq += 1
                fr = {str(im["image_id"]): int(re.findall(r"\d+", im["file_name"])[-1]) for im in lab["images"]}
                for o in lab["annotations"]:
                    if o.get("supercategory") != "object":
                        continue
                    at = o.get("attributes") or {}
                    if at.get("role") in (None, "other") or str(o["image_id"]) not in fr:
                        continue
                    p, b = o.get("bbox_pitch") or {}, o.get("bbox_image") or {}
                    x, y = p.get("x_bottom_middle"), p.get("y_bottom_middle")
                    w.writerow([seq, fr[str(o["image_id"])], o["track_id"], at.get("role"), at.get("team") or "",
                                at.get("jersey") or "", "" if x is None else round(x, 3), "" if y is None else round(y, 3),
                                b.get("x"), b.get("y"), b.get("w"), b.get("h")])
                    n_rows += 1
                print(f"\r[{split}] {n_seq} sequences, {n_rows:,} rows", end="", flush=True)
        print(f"\n[{split}] -> {path}")


if __name__ == "__main__":
    main()
