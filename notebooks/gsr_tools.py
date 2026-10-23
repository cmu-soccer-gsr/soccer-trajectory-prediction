"""Helpers for the SoccerNet-GSR midterm experiments (Group 15).

What this script does
---------------------
It works on TrackLab "tracker states" (.pklz files): the saved output of the
sn-gamestate pipeline. Because the state already holds every detection, track
ID, pitch position, team, role and jersey number, we can change one column
(the jersey numbers) and re-score GS-HOTA in about a minute, without re-running
any of the models. This is the "cached-state" approach from the TA feedback.

Commands
--------
inspect     Print what is inside a .pklz state (files, columns, example values).
            Run this first on any new state, to confirm the column names.
edit        Write a copy of a state with the jersey numbers changed:
              nojersey      every jersey set to null (floor: no jersey reader)
              oracle        each predicted tracklet gets the true number of the
                            ground-truth player it overlaps most (ceiling for a
                            reader that outputs one number per tracklet)
              oracle-frame  each detection gets the true number of the player it
                            matches in that frame (also undoes ID switches)
jersey-acc  Tracklet-level accuracy of the state's jersey reader (MMOCR in the
            baseline), with an error breakdown (item 2a of the midterm plan).
results     Collect GS-HOTA / DetA / AssA (overall and per clip) from one or more
            TrackLab run folders into tables for the report.

Examples (from the sn-gamestate folder, using its virtualenv):
    .venv/bin/python tools/gsr_tools.py inspect states/baseline-valid-slim.pklz
    .venv/bin/python tools/gsr_tools.py edit states/baseline-valid-slim.pklz \\
        states/edited/oracle.pklz --mode oracle --gt-dir data/SoccerNetGS/valid
    .venv/bin/python tools/gsr_tools.py jersey-acc states/baseline-valid-slim.pklz \\
        --gt-dir data/SoccerNetGS/valid --out-dir results/jersey --plot
    .venv/bin/python tools/gsr_tools.py results outputs/runs/cached_58 outputs/runs/oracle_58

Why the sn-gamestate virtualenv (Python 3.9, pandas 2.2): the state files are
pandas pickles, so we read and write them with the same pandas version TrackLab
uses. Writing them with a different pandas version could make them unreadable.

Key terms
---------
detection   One player box in one frame.
tracklet    The chain of detections the tracker links together as one player
            (same track_id). One real player can be split into several tracklets.
GT          Ground truth: the human labels in each clip's Labels-GameState.json.

How the GS-HOTA evaluator treats jerseys (sn-trackeval 0.4.0), which this
script mirrors:
  * it scores the tracklet-level column `jersey_number` (the result of voting
    over the per-frame reads in `jersey_number_detection`);
  * jerseys are compared as strings with ==, so "7" != 7 and "07" != "7";
  * jerseys only count for role == "player" (not goalkeepers or referees);
  * null vs null counts as a match (no number visible, none predicted).
"""
import argparse
import glob
import json
import os
import pickle
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm for matching

# Order matters: the dataset encodes the split as a digit in image ids
# (train=1, valid=2, test=3, challenge=4).
SPLITS = ["train", "valid", "test", "challenge"]


# ==========================================================================
# Small utilities
# ==========================================================================
def is_null(x):
    """True for None and NaN; False for anything else (including strings)."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):  # pd.isna on lists/dicts returns arrays
        return False


def jersey_str(x):
    """Normalise a jersey value to the string form the evaluator compares.

    7 / 7.0 / "7" -> "7"; null -> None. Other strings are kept as-is, so "07"
    stays "07" (the evaluator would also treat it as different from "7").
    """
    if is_null(x):
        return None
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(x)) if float(x).is_integer() else str(x)
    s = str(x).strip()
    return s if s != "" else None


def pitch_xy(b):
    """Return the player's position on the pitch, in meters, or None.

    `bbox_pitch` is a dict of the box's bottom edge projected onto the pitch.
    GS-HOTA uses the bottom-middle point (where the feet are), so we do too.
    """
    if isinstance(b, dict):
        x, y = b.get("x_bottom_middle"), b.get("y_bottom_middle")
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            return float(x), float(y)
    return None


def video_key(v):
    """Common key for a clip, whatever form its id comes in.

    The state names clips '021', the data folders are 'SNGS-021', and some
    places use 21. All of these map to '021'.
    """
    s = str(v).split("-")[-1]
    return s.zfill(3) if s.isdigit() else s


# ==========================================================================
# Reading and writing tracker states (.pklz)
#
# A .pklz file is a zip archive with, per clip:
#   <clip>.pkl        DataFrame of predicted detections (one row per box)
#   <clip>_image.pkl  DataFrame of per-frame info (camera parameters, etc.)
# plus summary.json listing the columns. We only ever edit <clip>.pkl.
# ==========================================================================
def list_state_videos(state_path):
    """Clip ids stored in a state, e.g. ['021', '022', ...]."""
    with zipfile.ZipFile(state_path) as zf:
        names = zf.namelist()
    return sorted(n[:-4] for n in names
                  if n.endswith(".pkl") and not n.endswith("_image.pkl"))


def iter_state_detections(state_path, videos=None):
    """Yield (clip_id, detections DataFrame) one clip at a time.

    Reading one clip at a time keeps memory low, which matters on Colab when
    the long end-to-end run is using the same machine. `videos` optionally
    restricts which clips are read (a set of video_key values).
    """
    with zipfile.ZipFile(state_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.endswith(".pkl") and not n.endswith("_image.pkl"))
        for name in names:
            vid = name[:-4]
            if videos is not None and video_key(vid) not in videos:
                continue
            with zf.open(name, force_zip64=True) as f:
                yield vid, pd.read_pickle(f)


def write_edited_state(src, dst, edit_fn, videos=None):
    """Copy a state to `dst`, passing each clip's detections through edit_fn.

    edit_fn(clip_id, df) must return the edited DataFrame. Everything else
    (image pickles, summary.json, clips outside `videos`) is copied byte for
    byte, so the result loads in TrackLab exactly like the original.
    We write to a temporary file first so a crash never leaves a half-written
    state behind under the final name.
    """
    tmp = dst + ".tmp"
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            is_det = name.endswith(".pkl") and not name.endswith("_image.pkl")
            vid = name[:-4] if is_det else None
            if is_det and (videos is None or video_key(vid) in videos):
                # Detections of a clip we want to edit: unpickle, edit, re-pickle.
                with zin.open(name, force_zip64=True) as f:
                    df = pd.read_pickle(f)
                df = edit_fn(vid, df)
                with zout.open(name, "w", force_zip64=True) as g:
                    pickle.dump(df, g, protocol=pickle.DEFAULT_PROTOCOL)
            else:
                # Anything else: stream-copy in 16 MB chunks without unpickling.
                with zin.open(name, force_zip64=True) as f, \
                        zout.open(name, "w", force_zip64=True) as g:
                    while True:
                        chunk = f.read(1 << 24)
                        if not chunk:
                            break
                        g.write(chunk)
    os.replace(tmp, dst)  # atomic rename: dst appears only when complete


# ==========================================================================
# Ground truth (Labels-GameState.json)
# ==========================================================================
class GroundTruth(object):
    """Ground-truth labels for every clip in a split folder (e.g. .../valid).

    Loads each clip's JSON lazily (only when first needed) and caches it.
    """

    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.split = os.path.basename(os.path.normpath(split_dir))  # e.g. "valid"
        self.folders = {}  # video key -> folder name, e.g. '021' -> 'SNGS-021'
        for path in sorted(glob.glob(os.path.join(split_dir, "*", "Labels-GameState.json"))):
            folder = os.path.basename(os.path.dirname(path))
            self.folders[video_key(folder)] = folder
        if not self.folders:
            raise SystemExit("No Labels-GameState.json found under %s" % split_dir)
        self._cache = {}

    def load(self, vkey):
        """Return (frame_of, gt) for one clip.

        frame_of: dict image_id (str) -> frame index 0..N-1
        gt:       DataFrame, one row per labeled person per frame, with columns
                  frame, gt_track_id, x, y (pitch meters), role, team, jersey
        """
        vkey = video_key(vkey)
        if vkey in self._cache:
            return self._cache[vkey]
        folder = self.folders[vkey]
        with open(os.path.join(self.split_dir, folder, "Labels-GameState.json")) as f:
            data = json.load(f)
        # Keep the same frames the evaluator keeps: only fully labeled ones.
        images = [im for im in data["images"]
                  if im.get("has_labeled_pitch") and im.get("has_labeled_camera")
                  and im.get("has_labeled_person")]
        images.sort(key=lambda im: int(str(im["image_id"]).split("_")[-1]))
        frame_of = {str(im["image_id"]): i for i, im in enumerate(images)}
        rows = []
        for a in data["annotations"]:
            if a.get("supercategory") != "object":  # skip pitch-line / camera entries
                continue
            attr = a.get("attributes") or {}
            if attr.get("role") == "ball":  # the evaluator ignores the ball
                continue
            iid = str(a["image_id"])
            if iid not in frame_of:  # frame not fully labeled
                continue
            xy = pitch_xy(a.get("bbox_pitch"))
            if xy is None:  # no pitch position, cannot be matched
                continue
            rows.append({
                "frame": frame_of[iid], "gt_track_id": int(a["track_id"]),
                "x": xy[0], "y": xy[1], "role": attr.get("role"),
                "team": attr.get("team"), "jersey": jersey_str(attr.get("jersey")),
            })
        gt = pd.DataFrame(rows, columns=["frame", "gt_track_id", "x", "y",
                                         "role", "team", "jersey"])
        self._cache[vkey] = (frame_of, gt)
        return self._cache[vkey]

    def track_table(self, vkey):
        """One row per GT player (track): their role, team and jersey number.

        GSR labels these per track, so they are constant; we take the most
        common non-null value just in case a frame disagrees.
        """
        _, gt = self.load(vkey)

        def mode(s):
            vals = [v for v in s if not is_null(v)]
            return Counter(vals).most_common(1)[0][0] if vals else None
        return gt.groupby("gt_track_id").agg(
            role=("role", mode), team=("team", mode), jersey=("jersey", mode),
            n_frames=("frame", "size"))


def map_pred_image_ids(image_ids, frame_of, split, vkey):
    """Convert predicted image ids to GT frame indices (-1 if no match).

    Normally the ids are identical. Older states used a different id format,
    so we apply the same conversion rule the official evaluator uses:
    "<split digit><clip id><frame number, 6 digits>".
    """
    split_id = SPLITS.index(split) + 1 if split in SPLITS else 2
    out = []
    for iid in image_ids:
        if is_null(iid):
            out.append(-1)
            continue
        s = str(int(iid)) if isinstance(iid, (int, np.integer, float, np.floating)) else str(iid)
        if s in frame_of:
            out.append(frame_of[s])
            continue
        if len(s) != 10 and len(s) >= 4 and s[-4:].isdigit():  # old-format id
            alt = "%d%s%06d" % (split_id, vkey, int(s[-4:]) + 1)
            if alt in frame_of:
                out.append(frame_of[alt])
                continue
        out.append(-1)
    return np.array(out, dtype=int)


# ==========================================================================
# Matching predictions to ground truth
#
# Shared by the oracle (needs the true number of each predicted tracklet) and
# the accuracy analysis (needs to know which player a tracklet really is).
# ==========================================================================
def match_video(dets, gt_obj, vkey, max_dist=5.0, purity_thr=0.8):
    """Match one clip's predicted tracklets to GT players.

    Step 1, per frame: pair predicted detections with GT players using the
    Hungarian algorithm on pitch distance (each prediction gets at most one
    player and vice versa, minimising total distance). Pairs further apart
    than `max_dist` meters are dropped; 5 m is GS-HOTA's own tolerance.

    Step 2, per tracklet: the tracklet is assigned to the GT player it was
    paired with most often across its frames ("majority vote").
    `purity` = share of its matched frames that went to that player. Low purity
    means the tracker switched between people mid-tracklet ("contaminated").

    Returns (det_match, trk):
      det_match: int array, det_match[i] = GT track id matched to dets.iloc[i]
                 in that frame, or -1. Positional on purpose, so it works even
                 if the DataFrame index has repeated labels.
      trk: DataFrame indexed by predicted track_id with columns
           gt_track_id (-1 = unmatched), n_dets, n_matched, purity, contaminated
    """
    frame_of, gt = gt_obj.load(vkey)
    n = len(dets)

    # --- Gather what we need from the predictions as plain arrays ----------
    frames = map_pred_image_ids(dets["image_id"].values, frame_of, gt_obj.split, vkey)
    xy = [pitch_xy(b) for b in dets["bbox_pitch"].values]
    tids = dets["track_id"].values
    has_tid = np.array([not is_null(t) for t in tids])
    # Usable for matching: has a pitch position, a track id and a labeled frame
    # (the evaluator also drops detections without track_id or bbox_pitch).
    ok = np.array([p is not None for p in xy]) & has_tid & (frames >= 0)
    px = np.array([p[0] if p is not None else np.nan for p in xy])
    py = np.array([p[1] if p is not None else np.nan for p in xy])

    # --- Step 1: frame-by-frame Hungarian matching --------------------------
    det_match = np.full(n, -1, dtype=int)
    gt_by_frame = {f: (g[["x", "y"]].values, g["gt_track_id"].values)
                   for f, g in gt.groupby("frame")}
    # Sort usable detections by frame and split them into one group per frame.
    ok_pos = np.flatnonzero(ok)
    order = ok_pos[np.argsort(frames[ok_pos], kind="stable")]
    if len(order):
        splits = np.flatnonzero(np.diff(frames[order])) + 1
        for grp in np.split(order, splits):
            f = frames[grp[0]]
            if f not in gt_by_frame:
                continue
            q, gids = gt_by_frame[f]                      # GT positions and ids
            p = np.stack([px[grp], py[grp]], axis=1)      # predicted positions
            dist = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))
            # Make too-distant pairs prohibitively expensive, then drop them.
            cost = np.where(dist <= max_dist, dist, 1e6)
            r, c = linear_sum_assignment(cost)
            keep = cost[r, c] <= max_dist
            det_match[grp[r[keep]]] = gids[c[keep]]

    # --- Step 2: majority vote per predicted tracklet ----------------------
    rows = []
    tid_pos = defaultdict(list)  # track_id -> row positions of its detections
    for i in np.flatnonzero(has_tid):
        tid_pos[tids[i]].append(i)
    for tid, pos in tid_pos.items():
        m = det_match[pos]
        m = m[m >= 0]  # frames where this tracklet matched someone
        if len(m):
            best, cnt = Counter(m.tolist()).most_common(1)[0]
            purity = cnt / float(len(m))
        else:
            best, purity = -1, np.nan  # never within max_dist of any GT player
        rows.append({"track_id": tid, "gt_track_id": int(best), "n_dets": len(pos),
                     "n_matched": int(len(m)), "purity": purity})
    trk = pd.DataFrame(rows, columns=["track_id", "gt_track_id", "n_dets", "n_matched",
                                      "purity"]).set_index("track_id")
    trk["contaminated"] = (trk["gt_track_id"] >= 0) & (trk["purity"] < purity_thr)
    return det_match, trk


# ==========================================================================
# Command: inspect
# ==========================================================================
def cmd_inspect(args):
    """Print the files, columns and example values of a state.

    Use it to check that the columns this script relies on exist:
    jersey_number, jersey_number_detection, bbox_pitch, track_id, image_id.
    """
    with zipfile.ZipFile(args.state) as zf:
        names = zf.namelist()
        print("%d files in %s" % (len(names), args.state))
        if "summary.json" in names:
            with zf.open("summary.json") as f:
                print("summary.json columns:", json.dumps(json.load(f)["columns"], indent=1))
    vids = list_state_videos(args.state)
    print("videos (%d): %s%s" % (len(vids), vids[:10], " ..." if len(vids) > 10 else ""))
    # Look at the first clip only; all clips share the same columns.
    df = next(iter_state_detections(args.state, videos={video_key(vids[0])}))[1]
    print("\nFirst video %s: %d detections" % (vids[0], len(df)))
    print(df.dtypes.to_string())
    for col in ["jersey_number", "jersey_number_detection", "jersey_number_confidence",
                "role", "team", "track_id", "image_id", "video_id", "bbox_pitch"]:
        if col in df.columns:
            vals = df[col].head(3).tolist()
            print("%-26s e.g. %s" % (col, vals))
        else:
            print("%-26s MISSING" % col)
    if "jersey_number" in df.columns:
        # Should be strings and None; numbers here would never match the GT strings.
        print("\njersey_number value types:",
              Counter(type(v).__name__ for v in df["jersey_number"]).most_common())


# ==========================================================================
# Command: edit (no-jersey / oracle configurations)
# ==========================================================================
def cmd_edit(args):
    """Write an edited copy of a state where only `jersey_number` changes.

    Everything else (positions, tracks, teams, roles) is untouched, so any
    change in GS-HOTA after re-scoring is caused by the jersey numbers alone.
    """
    videos = set(video_key(v) for v in args.clips) if args.clips else None
    gt_obj = GroundTruth(args.gt_dir) if args.mode.startswith("oracle") else None
    stats = Counter()  # counts printed at the end, as a sanity check

    def edit(vid, df):
        """Edit one clip's detections (called once per clip by write_edited_state)."""
        if "jersey_number" not in df.columns:
            raise SystemExit("State has no 'jersey_number' column; run `inspect` first.")
        df = df.copy()
        df["jersey_number"] = df["jersey_number"].astype(object)  # holds str or None

        # --- No-jersey: blank every prediction ------------------------------
        if args.mode == "nojersey":
            stats["detections"] += len(df)
            df["jersey_number"] = None
            return df

        # --- Oracle: replace predictions with the true numbers -------------
        vkey = video_key(vid)
        if vkey not in gt_obj.folders:
            stats["videos_without_gt"] += 1
            return df
        det_match, trk = match_video(df, gt_obj, vkey, args.max_dist)
        jersey_of = gt_obj.track_table(vkey)["jersey"].to_dict()  # GT track -> number
        old = df["jersey_number"].values
        new = old.copy()

        # Tracklet oracle: every detection of a matched tracklet gets the number
        # of the tracklet's majority GT player. This is the best a reader that
        # outputs one number per tracklet could do. Unmatched tracklets (false
        # detections) keep their predicted number.
        tid_gt = df["track_id"].map(trk["gt_track_id"]).values  # NaN where no track_id
        for i in np.flatnonzero(pd.notna(tid_gt) & (np.nan_to_num(tid_gt, nan=-1) >= 0)):
            new[i] = jersey_of.get(int(tid_gt[i]))

        # Frame oracle (optional): additionally give each detection the number
        # of the player it matches in that exact frame. Differs from the
        # tracklet oracle only where a tracklet switches between players, so
        # the score gap between the two measures what ID switches cost.
        if args.mode == "oracle-frame":
            for i in np.flatnonzero(det_match >= 0):
                new[i] = jersey_of.get(int(det_match[i]))

        stats["detections"] += len(df)
        stats["detections_changed"] += int(sum(jersey_str(a) != jersey_str(b)
                                               for a, b in zip(old, new)))
        df["jersey_number"] = new
        stats["tracklets"] += len(trk)
        stats["tracklets_matched"] += int((trk["gt_track_id"] >= 0).sum())
        return df

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_edited_state(args.state, args.out, edit, videos)
    print("Wrote %s (mode=%s)" % (args.out, args.mode))
    for k, v in sorted(stats.items()):
        print("  %s: %d" % (k, v))
    if args.mode.startswith("oracle") and stats["tracklets"]:
        print("  unmatched tracklets keep their predicted number (no GT player within %.1f m)"
              % args.max_dist)


# ==========================================================================
# Command: jersey-acc (item 2a: MMOCR tracklet accuracy + error breakdown)
# ==========================================================================
def classify_tracklet(pred, gt_j, reads):
    """Put one tracklet into an outcome bucket.

    pred:  the tracklet's final (voted) number, or None
    gt_j:  the true number, or None if the number is never visible
    reads: set of numbers read in individual frames, or None if the state
           doesn't store per-frame reads

    Buckets when the true number is labeled:
      correct            final number is right
      null_when_visible  no number output (in the baseline's vote this means
                         MMOCR never read a digit in any frame of the tracklet)
      wrong_outvoted     wrong, but the right number WAS read in some frame and
                         lost the vote -> better aggregation could fix it
      wrong_never_read   wrong, and the right number was never read in any
                         frame -> needs a better reader
      wrong_number       wrong (used when per-frame reads are unavailable)
    Buckets when the true number is not labeled:
      abstain_ok / number_when_not_visible
    """
    if gt_j is None:
        return "abstain_ok" if pred is None else "number_when_not_visible"
    if pred == gt_j:
        return "correct"
    if pred is None:
        return "null_when_visible"
    if reads is None:
        return "wrong_number"
    return "wrong_outvoted" if gt_j in reads else "wrong_never_read"


def cmd_jersey_acc(args):
    """Score the state's jersey predictions tracklet by tracklet.

    Accuracy = correct / (predicted tracklets matched to a GT *player* whose
    number is labeled). Goalkeepers and referees are reported separately
    because GS-HOTA does not score their jerseys.
    Outputs (in --out-dir): jersey_accuracy.txt (the printed report),
    jersey_error_breakdown.csv, jersey_accuracy.json, jersey_tracklets.csv
    (one row per tracklet, for digging into examples) and, with --plot,
    jersey_error_breakdown.png.
    """
    gt_obj = GroundTruth(args.gt_dir)
    videos = set(video_key(v) for v in args.clips) if args.clips else None
    rows = []
    has_reads = None  # whether the state stores per-frame reads

    # ---- One row per predicted tracklet, clip by clip ---------------------
    for vid, df in iter_state_detections(args.state, videos):
        vkey = video_key(vid)
        if vkey not in gt_obj.folders:
            print("skip %s: no ground truth" % vid)
            continue
        if has_reads is None:
            has_reads = "jersey_number_detection" in df.columns
        _, trk = match_video(df, gt_obj, vkey, args.max_dist, args.purity)
        gt_tracks = gt_obj.track_table(vkey)
        d = df[df["track_id"].notna()]
        for tid, g in d.groupby("track_id"):
            # Final number: identical on every row of a tracklet after voting.
            preds = [jersey_str(v) for v in g["jersey_number"]]
            pred = Counter(preds).most_common(1)[0][0]
            # Per-frame reads: what MMOCR read in each individual frame.
            reads = None
            n_reads = 0
            if has_reads:
                r = [jersey_str(v) for v in g["jersey_number_detection"]]
                r = [v for v in r if v is not None]
                reads, n_reads = set(r), len(r)
            t = trk.loc[tid] if tid in trk.index else None
            gid = int(t["gt_track_id"]) if t is not None else -1
            row = {"clip": gt_obj.folders[vkey], "track_id": tid, "n_dets": len(g),
                   "pred": pred, "n_frame_reads": n_reads,
                   "distinct_reads": "|".join(sorted(reads)) if reads else "",
                   "gt_track_id": gid,
                   "purity": float(t["purity"]) if t is not None else np.nan,
                   "contaminated": bool(t["contaminated"]) if t is not None else False}
            if gid < 0:
                # Not near any GT player: a false detection. Not scored.
                row.update(gt_role=None, gt_jersey=None, bucket="unmatched")
            else:
                info = gt_tracks.loc[gid]
                row.update(gt_role=info["role"], gt_jersey=info["jersey"])
                if info["role"] != "player":
                    row["bucket"] = "not_scored_role_%s" % info["role"]
                else:
                    row["bucket"] = classify_tracklet(pred, info["jersey"], reads)
            # Partial misreads, e.g. "1" for "17": one digit hidden or missed.
            pj, gj = row["pred"], row["gt_jersey"]
            row["partial_digit"] = bool(row["bucket"].startswith("wrong") and pj and gj
                                        and (pj in gj or gj in pj))
            rows.append(row)

    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("No tracklets found.")
    os.makedirs(args.out_dir, exist_ok=True)
    t.to_csv(os.path.join(args.out_dir, "jersey_tracklets.csv"), index=False)

    # ---- Summary numbers ---------------------------------------------------
    # "Scored" set: matched to a GT player whose number is labeled.
    vis = t[t["bucket"].isin(["correct", "null_when_visible", "wrong_outvoted",
                              "wrong_never_read", "wrong_number"])]
    notvis = t[t["bucket"].isin(["abstain_ok", "number_when_not_visible"])]
    order = ["correct", "null_when_visible", "wrong_outvoted", "wrong_never_read", "wrong_number"]
    counts = vis["bucket"].value_counts().reindex(order).dropna().astype(int)
    errors = counts.drop("correct", errors="ignore")
    n_vis = len(vis)
    acc = counts.get("correct", 0) / float(n_vis) if n_vis else np.nan
    # Weighted by tracklet length: long tracklets cover more frames, so they
    # matter more for GS-HOTA than short ones.
    w = vis["n_dets"]
    acc_w = w[vis["bucket"] == "correct"].sum() / float(w.sum()) if n_vis else np.nan

    tab = pd.DataFrame({"tracklets": counts,
                        "% of scored tracklets": (100.0 * counts / n_vis).round(1),
                        "% of errors": (100.0 * errors / max(errors.sum(), 1)).round(1)}
                       ).reindex(counts.index)
    tab["% of errors"] = tab["% of errors"].map(lambda v: "" if pd.isna(v) else v)

    # ---- Text report -------------------------------------------------------
    lines = []
    lines.append("Jersey tracklet accuracy (state: %s)" % os.path.basename(args.state))
    lines.append("Clips: %d | predicted tracklets: %d" % (t["clip"].nunique(), len(t)))
    lines.append("")
    lines.append("Scored set = predicted tracklets matched to a GT *player* whose number "
                 "is labeled (N = %d)" % n_vis)
    lines.append("  Tracklet accuracy:        %.1f%%" % (100 * acc))
    lines.append("  Detection-weighted acc.:  %.1f%%  (weights = tracklet length)" % (100 * acc_w))
    lines.append("")
    lines.append(tab.to_string())
    if not has_reads:
        lines.append("\n(No per-frame reads in this state, so wrong numbers are not split "
                     "into outvoted / never read.)")
    lines.append("")
    if len(vis):
        # Do ID switches hurt jersey accuracy? Compare clean vs contaminated.
        cont = vis.groupby("contaminated")["bucket"].apply(
            lambda s: (s == "correct").mean() * 100).round(1)
        lines.append("Accuracy by tracklet purity (contaminated = <%d%% of matched frames on "
                     "one GT player):" % int(100 * args.purity))
        for k, v in cont.items():
            lines.append("  %-14s %5.1f%%  (n=%d)" % ("contaminated" if k else "clean", v,
                                                     int((vis["contaminated"] == k).sum())))
        lines.append("Wrong numbers sharing a digit with the truth (e.g. '1' for '17'): %d"
                     % int(vis["partial_digit"].sum()))
    lines.append("")
    lines.append("Matched GT players with no labeled number (N = %d):" % len(notvis))
    for k, v in notvis["bucket"].value_counts().items():
        lines.append("  %-26s %d" % (k, v))
    other = t[~t.index.isin(vis.index) & ~t.index.isin(notvis.index)]["bucket"].value_counts()
    lines.append("Not scored for jerseys (GS-HOTA ignores them):")
    for k, v in other.items():
        lines.append("  %-26s %d" % (k, v))
    report = "\n".join(lines)
    print(report)

    # ---- Save everything ---------------------------------------------------
    with open(os.path.join(args.out_dir, "jersey_accuracy.txt"), "w") as f:
        f.write(report + "\n")
    tab.to_csv(os.path.join(args.out_dir, "jersey_error_breakdown.csv"))
    with open(os.path.join(args.out_dir, "jersey_accuracy.json"), "w") as f:
        json.dump({"n_scored": int(n_vis), "tracklet_accuracy": float(acc),
                   "detection_weighted_accuracy": float(acc_w),
                   "buckets": {k: int(v) for k, v in counts.items()}}, f, indent=2)
    if args.plot:
        plot_breakdown(counts, n_vis, os.path.join(args.out_dir, "jersey_error_breakdown.png"))


def plot_breakdown(counts, n, path):
    """Save a single stacked bar: share of scored tracklets in each bucket."""
    os.environ["MPLBACKEND"] = "Agg"  # Colab's inline backend breaks this older matplotlib
    import matplotlib
    matplotlib.use("Agg")             # draw to a file, no screen needed
    import matplotlib.pyplot as plt
    labels = {"correct": "Correct", "null_when_visible": "Null when visible",
              "wrong_outvoted": "Wrong (right number outvoted)",
              "wrong_never_read": "Wrong (right number never read)",
              "wrong_number": "Wrong number"}
    # Green for correct, warmer colours for worse outcomes.
    colors = {"correct": "#2a9d8f", "null_when_visible": "#e9c46a",
              "wrong_outvoted": "#f4a261", "wrong_never_read": "#e76f51",
              "wrong_number": "#e76f51"}
    fig, ax = plt.subplots(figsize=(8, 2.4))
    left = 0.0
    for k, v in counts.items():
        pct = 100.0 * v / n
        ax.barh([0], [pct], left=left, color=colors[k], label="%s (%.0f%%)" % (labels[k], pct))
        left += pct
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("%% of scored tracklets (N = %d)" % n)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved plot %s" % path)


# ==========================================================================
# Command: results (collect scores from TrackLab run folders)
# ==========================================================================
def read_run(run_dir):
    """Read one run's per-clip scores from TrackEval's detailed CSV.

    TrackLab's evaluator writes <run>/eval/results/tracklab/person_detailed.csv,
    with one row per clip plus a COMBINED row. GS-HOTA appears there under the
    name "HOTA". The ___AUC columns are the values averaged over the
    localization thresholds, which is how HOTA is reported. Values are 0-1,
    so we convert to percent.
    """
    hits = glob.glob(os.path.join(run_dir, "**", "person_detailed.csv"), recursive=True)
    if not hits:
        raise SystemExit("No person_detailed.csv under %s (did evaluation run?)" % run_dir)
    df = pd.read_csv(sorted(hits)[-1])
    keep = {"HOTA___AUC": "GS-HOTA", "DetA___AUC": "DetA", "AssA___AUC": "AssA"}
    out = df[["seq"] + list(keep)].rename(columns=keep)
    for c in keep.values():
        out[c] = (100 * out[c]).round(2)
    return out.set_index("seq")


def cmd_results(args):
    """Print and save an overall table (one row per run) and a per-clip table.

    Note: COMBINED is computed over each run's own clips, so only compare
    runs scored on the same clip set (e.g. all 58, or the same 5).
    """
    names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(names) != len(args.runs):
        raise SystemExit("--names must match the number of runs")
    tables = {n: read_run(r) for n, r in zip(names, args.runs)}
    overall = pd.DataFrame({n: t.loc["COMBINED"] for n, t in tables.items()}).T
    overall["clips"] = [len(t) - 1 for t in tables.values()]  # minus the COMBINED row
    print("Overall (COMBINED over each run's clips)")
    print(overall.to_string())
    per_clip = pd.concat({n: t.drop(index="COMBINED")["GS-HOTA"] for n, t in tables.items()},
                         axis=1)
    if args.clips:
        per_clip = per_clip.reindex(args.clips)
    print("\nGS-HOTA per clip")
    print(per_clip.to_string())
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        overall.to_csv(os.path.join(args.out_dir, args.prefix + "_overall.csv"))
        per_clip.to_csv(os.path.join(args.out_dir, args.prefix + "_per_clip.csv"))
        for n, t in tables.items():
            t.to_csv(os.path.join(args.out_dir, "%s_%s_detailed.csv" % (args.prefix, n)))
        print("\nSaved tables to %s" % args.out_dir)


# ==========================================================================
# Command-line interface
# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("inspect", help="show files, columns and example values of a state")
    s.add_argument("state", help="path to a .pklz tracker state")

    s = sub.add_parser("edit", help="write a no-jersey or oracle copy of a state")
    s.add_argument("state", help="input .pklz tracker state")
    s.add_argument("out", help="output .pklz path")
    s.add_argument("--mode", required=True, choices=["nojersey", "oracle", "oracle-frame"],
                   help="nojersey = all null; oracle = true number per tracklet; "
                        "oracle-frame = true number per detection")
    s.add_argument("--gt-dir", help="split folder with SNGS-*/Labels-GameState.json "
                                    "(required for oracle modes)")
    s.add_argument("--clips", nargs="*", help="only edit these clips (e.g. SNGS-021); "
                                              "others are copied unchanged")
    s.add_argument("--max-dist", type=float, default=5.0,
                   help="max pitch distance in meters for a prediction to match a GT player")

    s = sub.add_parser("jersey-acc", help="tracklet-level jersey accuracy + error breakdown")
    s.add_argument("state", help="input .pklz tracker state")
    s.add_argument("--gt-dir", required=True, help="split folder with SNGS-*/Labels-GameState.json")
    s.add_argument("--out-dir", required=True, help="folder for the report, CSVs and plot")
    s.add_argument("--clips", nargs="*", help="only analyse these clips (e.g. SNGS-021)")
    s.add_argument("--max-dist", type=float, default=5.0,
                   help="max pitch distance in meters for a prediction to match a GT player")
    s.add_argument("--purity", type=float, default=0.8,
                   help="tracklets with less than this share of frames on one player "
                        "are flagged as contaminated")
    s.add_argument("--plot", action="store_true", help="also save a stacked bar chart")

    s = sub.add_parser("results", help="collect GS-HOTA/DetA/AssA tables from run folders")
    s.add_argument("runs", nargs="+", help="TrackLab run directories")
    s.add_argument("--names", nargs="*", help="display name per run (same order as runs)")
    s.add_argument("--clips", nargs="*", help="restrict the per-clip table to these clips")
    s.add_argument("--out-dir", help="if given, save the tables here as CSV")
    s.add_argument("--prefix", default="gs_hota", help="file name prefix for saved tables")

    args = p.parse_args(argv)
    if args.cmd == "edit" and args.mode.startswith("oracle") and not args.gt_dir:
        p.error("--gt-dir is required for oracle modes")
    if args.cmd == "edit" and args.mode == "oracle":
        args.mode = "oracle-tracklet"  # internal name, to tell it apart from oracle-frame
    {"inspect": cmd_inspect, "edit": cmd_edit, "jersey-acc": cmd_jersey_acc,
     "results": cmd_results}.get(args.cmd, lambda a: p.print_help())(args)


if __name__ == "__main__":
    main()
