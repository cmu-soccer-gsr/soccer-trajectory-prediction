#!/usr/bin/env python3
"""Bridge between a GSR tracker state (.pklz) and the Koshkina & Elder jersey pipeline.

Runs in the sn-gamestate env ($PY in the notebook), like tools/gsr_tools.py,
and reuses that file's state reading and writing helpers.

Commands
--------
export     Cut player crops out of the video frames for every tracklet in a
           state, in the folder layout Koshkina's code expects:
               <out>/<tracklet>/<tracklet>_<frame>.jpg
writeback  Write Koshkina's per-tracklet numbers (final_results.json) into a
           copy of the state's `jersey_number` column, ready to re-score with
           TrackLab exactly like the no-jersey and oracle copies.

Tracklet names are "<clip>-<track_id>", e.g. "021-17". There must be no
underscore in the name: Koshkina's code treats everything before the first
"_" of a crop's file name as the tracklet name, and it puts all torso crops
in one folder, so file names must also be unique across clips.

Examples
--------
    $PY src/jersey/koshkina_bridge.py export "$SLIM" --gt-dir "$GT_DIR" \
        --out /content/kosh/images --clips SNGS-021 SNGS-022
    $PY src/jersey/koshkina_bridge.py writeback "$SLIM" states/edited/koshkina.pklz \
        --pred-json /content/kosh/work/final_results.json --clips SNGS-021 SNGS-022
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Reuse the team's helpers. In Colab they live in $REPO_DIR/tools (written by
# notebook cell 8); in the repo they are committed under notebooks/.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_REPO, "tools"), os.path.join(_REPO, "notebooks")):
    if os.path.isfile(os.path.join(_p, "gsr_tools.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
from gsr_tools import (SPLITS, is_null, iter_state_detections,  # noqa: E402
                       video_key, write_edited_state)


def tracklet_name(vkey, track_id):
    """'021', 17.0 -> '021-17' (no underscore, see module docstring)."""
    return "%s-%d" % (vkey, int(track_id))


# ==========================================================================
# export
# ==========================================================================
def clip_folders(gt_dir):
    """Video key -> clip folder name, e.g. '021' -> 'SNGS-021'."""
    out = {}
    for name in sorted(os.listdir(gt_dir)):
        if os.path.isfile(os.path.join(gt_dir, name, "Labels-GameState.json")):
            out[video_key(name)] = name
    if not out:
        raise SystemExit("No clip folders with Labels-GameState.json under %s" % gt_dir)
    return out


def image_files(gt_dir, folder):
    """image_id (str) -> absolute path of that frame's jpg."""
    clip_dir = os.path.join(gt_dir, folder)
    with open(os.path.join(clip_dir, "Labels-GameState.json")) as f:
        images = json.load(f)["images"]
    files = {}
    for im in images:
        name = im["file_name"]
        path = os.path.join(clip_dir, "img1", name)
        if not os.path.isfile(path):          # fall back to a flat layout
            path = os.path.join(clip_dir, name)
        files[str(im["image_id"])] = path
    first = next(iter(files.values()))
    if not os.path.isfile(first):
        raise SystemExit("Frame images not found (looked for %s). Is the split unzipped?" % first)
    return files


def lookup_image(iid, files, split, vkey):
    """Find the frame for a predicted image id (same rule as gsr_tools.map_pred_image_ids)."""
    if is_null(iid):
        return None
    s = str(int(iid)) if isinstance(iid, (int, np.integer, float, np.floating)) else str(iid)
    if s in files:
        return files[s]
    if len(s) != 10 and len(s) >= 4 and s[-4:].isdigit():   # old-format id
        split_id = SPLITS.index(split) + 1 if split in SPLITS else 2
        return files.get("%d%s%06d" % (split_id, vkey, int(s[-4:]) + 1))
    return None


def pick_rows(g, every, max_per_track):
    """Pick which detections of one tracklet to export.

    Keep every `every`-th detection in frame order, then thin evenly to at
    most `max_per_track`. Defaults keep everything. Subsampling saves pose/STR
    time, but Koshkina's vote needs enough confident frames (see
    run_koshkina.py --sum-threshold).
    """
    g = g.sort_values("image_id")
    idx = np.arange(0, len(g), max(1, every))
    if max_per_track and len(idx) > max_per_track:
        idx = idx[np.linspace(0, len(idx) - 1, max_per_track).round().astype(int)]
    return g.iloc[np.unique(idx)]


def cmd_export(args):
    import cv2  # in the sn-gamestate env

    folders = clip_folders(args.gt_dir)
    split = os.path.basename(os.path.normpath(args.gt_dir))
    videos = set(video_key(v) for v in args.clips) if args.clips else None
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    stats = Counter()

    for vid, df in iter_state_detections(args.state, videos):
        vkey = video_key(vid)
        if vkey not in folders:
            print("skip %s: no clip folder under %s" % (vid, args.gt_dir))
            continue
        files = image_files(args.gt_dir, folders[vkey])
        df = df[df["track_id"].notna() & df["bbox_ltwh"].notna()]

        # Choose the detections to export, then group them by frame so each
        # frame is read from disk only once.
        jobs = defaultdict(list)   # image path -> [(tracklet, frame stem, bbox)]
        for tid, g in df.groupby("track_id"):
            name = tracklet_name(vkey, tid)
            chosen = pick_rows(g, args.every, args.max_per_track)
            manifest[name] = {"clip": folders[vkey], "track_id": int(tid),
                              "n_dets": int(len(g)), "n_exported": 0}
            for _, r in chosen.iterrows():
                path = lookup_image(r["image_id"], files, split, vkey)
                if path is None:
                    stats["no_frame_for_image_id"] += 1
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                jobs[path].append((name, stem, r["bbox_ltwh"]))

        for path, items in jobs.items():
            img = cv2.imread(path)
            if img is None:
                stats["unreadable_frame"] += len(items)
                continue
            H, W = img.shape[:2]
            for name, stem, b in items:
                l, t, w, h = [float(v) for v in b]
                if h < args.min_height:
                    stats["too_small"] += 1
                    continue
                x1, y1 = max(0, int(round(l))), max(0, int(round(t)))
                x2, y2 = min(W, int(round(l + w))), min(H, int(round(t + h)))
                if x2 <= x1 or y2 <= y1:
                    stats["empty_box"] += 1
                    continue
                d = os.path.join(args.out, name)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, "%s_%s.jpg" % (name, stem)), img[y1:y2, x1:x2])
                manifest[name]["n_exported"] += 1
                stats["crops"] += 1
        print("[%s] %d tracklets, %d crops so far" % (folders[vkey], len(df["track_id"].unique()),
                                                     stats["crops"]))

    # Koshkina's code lists every folder in --out as a tracklet, so the
    # manifest goes next to it, not inside it.
    manifest_path = args.manifest or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                  "crops_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    n_folders = sum(1 for m in manifest.values() if m["n_exported"] > 0)
    print("Wrote %d crops for %d tracklets to %s" % (stats["crops"], n_folders, args.out))
    for k, v in sorted(stats.items()):
        if k != "crops":
            print("  %s: %d" % (k, v))
    print("Manifest: %s" % manifest_path)


# ==========================================================================
# writeback
# ==========================================================================
def cmd_writeback(args):
    with open(args.pred_json) as f:
        preds = json.load(f)   # {"021-17": 7, "021-18": -1, ...}
    videos = set(video_key(v) for v in args.clips) if args.clips else None
    stats = Counter()

    def to_jersey(v):
        """Koshkina label -> the string form GS-HOTA compares ("7"), or None."""
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        return str(v) if 0 < v < 100 else None   # -1 means unreadable

    def edit(vid, df):
        vkey = video_key(vid)
        df = df.copy()
        df["jersey_number"] = df["jersey_number"].astype(object)   # holds str or None
        old = df["jersey_number"].values
        new = []
        for tid, o in zip(df["track_id"].values, old):
            key = None if is_null(tid) else tracklet_name(vkey, tid)
            if key in preds:
                new.append(to_jersey(preds[key]))
                stats["detections_from_koshkina"] += 1
            else:
                # No prediction: tracklet had no exported crops (or no track id).
                new.append(o if args.keep_missing else None)
                stats["detections_without_prediction"] += 1
        df["jersey_number"] = new
        return df

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_edited_state(args.state, args.out, edit, videos)

    labels = Counter("unreadable" if to_jersey(v) is None else "number"
                     for k, v in preds.items() if k != "imgs")
    print("Wrote %s" % args.out)
    print("  tracklets in %s: %d with a number, %d unreadable"
          % (os.path.basename(args.pred_json), labels["number"], labels["unreadable"]))
    for k, v in sorted(stats.items()):
        print("  %s: %d" % (k, v))
    print("  missing tracklets %s" % ("keep the original number" if args.keep_missing
                                      else "are set to null"))


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("export", help="cut tracklet crops for Koshkina's pipeline")
    s.add_argument("state", help="input .pklz tracker state")
    s.add_argument("--gt-dir", required=True, help="split folder with SNGS-*/ clip folders")
    s.add_argument("--out", required=True, help="output folder, one subfolder per tracklet")
    s.add_argument("--clips", nargs="*", help="only these clips (e.g. SNGS-021)")
    s.add_argument("--every", type=int, default=1,
                   help="keep every Nth detection per tracklet (1 = all, matches their setup)")
    s.add_argument("--max-per-track", type=int, default=0,
                   help="cap on crops per tracklet (0 = no cap)")
    s.add_argument("--min-height", type=float, default=0,
                   help="skip boxes shorter than this many pixels")
    s.add_argument("--manifest", help="where to write crops_manifest.json "
                                      "(default: next to --out)")

    s = sub.add_parser("writeback", help="write Koshkina's numbers into a copy of a state")
    s.add_argument("state", help="input .pklz tracker state")
    s.add_argument("out", help="output .pklz path")
    s.add_argument("--pred-json", required=True, help="Koshkina's final_results.json")
    s.add_argument("--clips", nargs="*", help="only edit these clips; others are copied unchanged")
    s.add_argument("--keep-missing", action="store_true",
                   help="tracklets without a prediction keep their old number (default: null)")

    args = p.parse_args(argv)
    if args.cmd == "export":
        cmd_export(args)
    elif args.cmd == "writeback":
        cmd_writeback(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
