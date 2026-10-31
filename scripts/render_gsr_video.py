#!/usr/bin/env python3
"""
Render a broadcast + minimap video for one SoccerNet-GSR clip, plus an imputation demo.

    python scripts/render_gsr_video.py --clip-dir data/SNGS-021 \
        --results results --out results/videos/SNGS-021.mp4

Scenes (1920x1080, 25 fps, the clip's native frame rate):
  1. Title card.
  2. Full clip: broadcast frame with labeled boxes and pitch lines, the minimap with
     2 s trails, and a track timeline that shows when each player is visible (gaps =
     off screen or missed).
  3. Imputation demo: one player's 1 s gap is hidden; the observed context gets the
     same sigma = 0.25 m noise as the "noisy context" experiment; linear, Kalman and
     BiGRU fill the gap (half speed), then the ground truth is revealed with errors.
  4. End card with the test-set results.

Needs: numpy, opencv-python, pillow, torch (for the GRU), ffmpeg on PATH, and
src/imputation from this repo.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

W, H, FPS = 1920, 1080, 25
BG = (17, 20, 24)
PANEL = (26, 30, 36)
INK, MUTED, FAINT = (240, 239, 236), (170, 172, 176), (90, 94, 100)
PITCH_BG, PITCH_LINE = (28, 58, 42), (200, 214, 204)
TEAM = {"left": (57, 135, 229), "right": (230, 110, 60), None: (200, 200, 200)}
REF = (237, 180, 30)
METHOD = {  # fixed order, colour follows the method
    "linear": ((150, 150, 146), "Linear"),
    "kalman": ((230, 110, 60), "Kalman (CV + RTS)"),
    "gru": ((57, 135, 229), "BiGRU (ours)"),
}
FONT_DIR = "/usr/share/fonts/opentype/inter/"


def font(size, weight="Regular"):
    for name in (f"Inter-{weight}.otf", "Inter-Regular.otf"):
        try:
            return ImageFont.truetype(FONT_DIR + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {k: font(*v) for k, v in {
    "title": (64, "Bold"), "h1": (34, "SemiBold"), "h2": (24, "SemiBold"),
    "body": (20, "Regular"), "small": (16, "Regular"), "chip": (15, "SemiBold"), "tiny": (11, "SemiBold"),
    "big": (44, "Bold")}.items()}


# ------------------------------------------------------------------ data
def load_clip(clip_dir):
    lab = json.loads((clip_dir / "Labels-GameState.json").read_text())
    images = sorted(lab["images"], key=lambda im: im["file_name"])
    frame_of = {str(im["image_id"]): i for i, im in enumerate(images)}
    n = len(images)
    objs = [[] for _ in range(n)]
    lines = [None] * n
    for a in lab["annotations"]:
        f = frame_of.get(str(a["image_id"]))
        if f is None:
            continue
        if a.get("supercategory") == "pitch":
            lines[f] = a.get("lines", {})
        elif a.get("supercategory") == "object":
            at = a.get("attributes") or {}
            if at.get("role") in (None, "other"):
                continue
            bp = a.get("bbox_pitch") or {}
            objs[f].append({
                "tid": a["track_id"], "role": at.get("role"), "team": at.get("team"),
                "jersey": at.get("jersey"), "box": a["bbox_image"],
                "xy": (bp.get("x_bottom_middle"), bp.get("y_bottom_middle")),
            })
    return lab["info"], images, objs, lines


def tracks_xy(objs):
    """track_id -> (frames array, xy array) for objects with a pitch position."""
    tr = {}
    for f, os_ in enumerate(objs):
        for o in os_:
            x, y = o["xy"]
            if o["role"] == "ball" or x is None or y is None:
                continue
            tr.setdefault(o["tid"], ([], []))
            tr[o["tid"]][0].append(f)
            tr[o["tid"]][1].append((x, y))
    return {k: (np.array(v[0]), np.array(v[1], np.float32)) for k, v in tr.items()}


# ------------------------------------------------------------------ geometry
class Pitch:
    """Maps pitch metres (x in [-52.5, 52.5], y in [-34, 34]) into a pixel rectangle."""

    def __init__(self, x0, y0, w):
        self.s = w / 105.0
        self.x0, self.y0, self.w, self.h = x0, y0, w, 68 * self.s

    def p(self, x, y):
        return (int(round(self.x0 + (x + 52.5) * self.s)), int(round(self.y0 + (y + 34) * self.s)))

    def draw(self, img):
        x1, y1 = int(self.x0 + self.w), int(self.y0 + self.h)
        cv2.rectangle(img, (int(self.x0) - 10, int(self.y0) - 10), (x1 + 10, y1 + 10), PITCH_BG, -1)
        c, t = PITCH_LINE, 1
        L = lambda a, b: cv2.line(img, self.p(*a), self.p(*b), c, t, cv2.LINE_AA)
        cv2.rectangle(img, self.p(-52.5, -34), self.p(52.5, 34), c, t, cv2.LINE_AA)
        L((0, -34), (0, 34))
        cv2.circle(img, self.p(0, 0), int(9.15 * self.s), c, t, cv2.LINE_AA)
        for sgn in (-1, 1):
            gx = 52.5 * sgn
            for depth, half in ((16.5, 20.16), (5.5, 9.16)):
                ix = gx - sgn * depth
                L((gx, -half), (ix, -half)); L((ix, -half), (ix, half)); L((ix, half), (gx, half))
            cv2.circle(img, self.p(gx - sgn * 11, 0), 2, c, -1, cv2.LINE_AA)


def colour(o):
    if o["role"] == "referee":
        return REF
    return TEAM.get(o["team"], TEAM[None])


def label(o):
    if o["role"] == "referee":
        return "R"
    if o["role"] == "goalkeeper":
        return "GK" + (o["jersey"] or "")
    return o["jersey"] or ""


# ------------------------------------------------------------------ text layer
class Text:
    """Collect text, then draw it in one PIL pass per frame."""

    def __init__(self):
        self.items = []

    def __call__(self, xy, s, f="body", fill=INK, anchor="la"):
        self.items.append((xy, s, f, fill, anchor))

    def chip(self, xy, s, bg, f="chip"):
        self.items.append(("chip", xy, s, f, bg))

    def render(self, img_bgr):
        im = Image.fromarray(img_bgr[:, :, ::-1])
        d = ImageDraw.Draw(im)
        for it in self.items:
            if it[0] == "chip":
                _, (x, y), s, f, bg = it
                l, t, r, b = d.textbbox((x, y), s, font=F[f], anchor="mb")
                d.rounded_rectangle((l - 4, t - 2, r + 4, b + 3), 4, fill=bg)
                d.text((x, y), s, font=F[f], fill=(255, 255, 255), anchor="mb")
            else:
                xy, s, f, fill, anchor = it
                d.text(xy, s, font=F[f], fill=fill, anchor=anchor)
        return np.asarray(im)[:, :, ::-1].copy()


def rgb2bgr(c):
    return (c[2], c[1], c[0])


# ------------------------------------------------------------------ panels
def draw_broadcast(canvas, img, objs, lines, hidden_tid=None, dashed=False, scale=1280 / 1920):
    small = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    canvas[0:720, 0:1280] = small
    if lines:
        ov = canvas[0:720, 0:1280].copy()
        for pts in lines.values():
            p = np.array([[q["x"] * 1280, q["y"] * 720] for q in pts], np.int32)
            if len(p) > 1:
                cv2.polylines(ov, [p], False, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[0:720, 0:1280] = cv2.addWeighted(ov, 0.45, canvas[0:720, 0:1280], 0.55, 0)
    chips = []
    for o in objs:
        if o["role"] == "ball":
            continue
        b = o["box"]
        x0, y0 = int(b["x"] * scale), int(b["y"] * scale)
        x1, y1 = int((b["x"] + b["w"]) * scale), int((b["y"] + b["h"]) * scale)
        col = rgb2bgr(colour(o))
        if o["tid"] == hidden_tid and dashed:
            for i in range(x0, x1, 8):
                cv2.line(canvas, (i, y0), (min(i + 4, x1), y0), (235, 235, 235), 2)
                cv2.line(canvas, (i, y1), (min(i + 4, x1), y1), (235, 235, 235), 2)
            for j in range(y0, y1, 8):
                cv2.line(canvas, (x0, j), (x0, min(j + 4, y1)), (235, 235, 235), 2)
                cv2.line(canvas, (x1, j), (x1, min(j + 4, y1)), (235, 235, 235), 2)
            chips.append(((x0 + x1) // 2, y1 + 22, "hidden", (90, 94, 100)))
            continue
        if o["tid"] == hidden_tid:
            cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        if label(o):
            chips.append(((x0 + x1) // 2, y0 - 3, label(o), colour(o)))
    return chips


def draw_minimap(canvas, pitch, objs, trails, highlight=None):
    pitch.draw(canvas)
    for tid, pts in trails.items():
        if len(pts) > 1:
            col = rgb2bgr(pts[0][2])
            p = np.array([pitch.p(x, y) for x, y, _ in pts], np.int32)
            cv2.polylines(canvas, [p], False, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
    chips = []
    for o in objs:
        x, y = o["xy"]
        if x is None or o["role"] == "ball":
            continue
        c = pitch.p(x, y)
        col = rgb2bgr(colour(o))
        if o["role"] == "goalkeeper":
            cv2.rectangle(canvas, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), col, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, c, 7, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 8, (20, 20, 20), 1, cv2.LINE_AA)
        if o["tid"] == highlight:
            cv2.circle(canvas, c, 13, (255, 255, 255), 2, cv2.LINE_AA)
        if label(o) and c[1] - 10 > 70:
            chips.append((c[0], c[1] - 10, label(o), colour(o)))
    return chips


# ------------------------------------------------------------------ encoder
class Encoder:
    def __init__(self, path):
        self.tmp = Path(str(path)[:-4] + ".part.mp4")
        self.path = Path(path)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.tmp)],
            stdin=subprocess.PIPE)

    def write(self, frame, repeat=1):
        for _ in range(repeat):
            self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        self.tmp.rename(self.path)


def fade(frame, a):
    return (frame.astype(np.float32) * a + np.array(BG, np.float32) * (1 - a)).astype(np.uint8)


def card(lines, sub=None):
    img = np.full((H, W, 3), BG, np.uint8)
    t = Text()
    y = H // 2 - 40 * len(lines)
    for i, (s, f, c) in enumerate(lines):
        t((W // 2, y + i * 80), s, f, c, "mm")
    if sub:
        t((W // 2, H - 80), sub, "small", MUTED, "mm")
    return t.render(img)


def hold(enc, frame, seconds, fade_in=0.4, fade_out=0.4):
    n = int(seconds * FPS)
    fi, fo = int(fade_in * FPS), int(fade_out * FPS)
    for i in range(n):
        a = min(1.0, (i + 1) / fi if fi else 1, (n - i) / fo if fo else 1)
        enc.write(fade(frame, a) if a < 1 else frame)


# ------------------------------------------------------------------ imputation demo
def pick_window(tracks, objs, results_dir, window=75, gap=25, max_step=1.0, target_ratio=0.9):
    """A typical example, chosen by rule rather than by eye.

    Candidates: players clearly visible on the broadcast for the whole window (box inside
    the central part of the frame, at least 90 px tall, lower part of the frame) who move more than 2 m during the
    gap. Among them, pick the window whose Kalman error is near the median and whose
    BiGRU / Kalman error ratio is closest to the test-set average (about 0.9).
    """
    box = {(o["tid"], f): o["box"] for f, os_ in enumerate(objs) for o in os_}
    def visible(tid, f):
        b = box.get((tid, f))
        if b is None:
            return False
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        return 0.1 * 1920 < cx < 0.9 * 1920 and 0.3 * 1080 < cy < 0.92 * 1080 and b["h"] >= 90
    cands = []
    for tid, (fr, xy) in tracks.items():
        role = next((o["role"] for o in objs[fr[0]] if o["tid"] == tid), None)
        if role not in ("player",):
            continue
        brk = np.flatnonzero(np.diff(fr) != 1) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, len(fr)]):
            seg = xy[s:e]
            for k in range(0, len(seg) - window + 1, 10):
                w = seg[k:k + window]
                if np.linalg.norm(np.diff(w, axis=0), axis=1).max() > max_step:
                    continue
                g0 = (window - gap) // 2
                gg = w[g0:g0 + gap]
                chord = np.linspace(gg[0], gg[-1], gap)
                curv = np.linalg.norm(gg - chord, axis=1).max()
                disp = np.linalg.norm(gg[-1] - gg[0])
                f0 = int(fr[s + k])
                if disp > 2 and all(visible(tid, f0 + j) for j in range(window)):
                    cands.append((curv, tid, f0, w))
    if not cands:
        raise SystemExit("no clearly visible moving player window in this clip")
    scored = []
    for c in cands:
        _, _, _, errs = impute(c[3], results_dir)
        scored.append((errs["kalman"].mean(), errs.get("gru", errs["kalman"]).mean(), c))
    kal = np.array([x[0] for x in scored])
    rank = kal.argsort().argsort() / max(1, len(kal) - 1)
    score = [abs(g / k - target_ratio) + 0.5 * abs(r - 0.5) for (k, g, _), r in zip(scored, rank)]
    return scored[int(np.argmin(score))][2], len(cands)


def impute(gt, results_dir, sigma=0.25, gap=25, seed=0):
    from imputation.baselines import kalman, linear
    from imputation.data import centred_mask
    m = centred_mask(gap, len(gt))
    noisy = gt + np.random.default_rng(seed).normal(0, sigma, gt.shape).astype(np.float32)
    params = json.loads((results_dir / "baseline_params.json").read_text())[str(sigma)]
    Y = noisy[None]
    preds = {"linear": linear(Y, m)[0], "kalman": kalman(Y, m, **params["kalman"])[0]}
    ck = results_dir / "gru.pt"
    if ck.exists():
        import torch
        from imputation.gru import GRUImputer, predict_numpy
        st = torch.load(ck, map_location="cpu")
        model = GRUImputer(hidden=st["hidden"])
        model.load_state_dict(st["state_dict"])
        preds["gru"] = predict_numpy(model, Y, m)[0]
    errs = {k: np.linalg.norm(p[m] - gt[m], axis=1) for k, p in preds.items()}
    return m, noisy, preds, errs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True, type=Path)
    ap.add_argument("--results", default=REPO / "results", type=Path,
                    help="folder with gru.pt and baseline_params.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=None, help="for quick previews")
    ap.add_argument("--demo-track", type=int, default=None,
                    help="override the demo: track_id of the player (with --demo-start)")
    ap.add_argument("--demo-start", type=int, default=None,
                    help="override the demo: first frame (0-based) of the 75-frame window")
    ap.add_argument("--stills", type=Path, default=None,
                    help="only write preview PNGs of a few busy frames to this folder")
    args = ap.parse_args()

    info, images, objs, lines = load_clip(args.clip_dir)
    n = len(images) if args.max_frames is None else min(args.max_frames, len(images))
    tracks = tracks_xy(objs)
    name = info.get("name", args.clip_dir.name)

    # timeline rows: left team, right team, referees, sorted by jersey
    meta = {}
    for os_ in objs:
        for o in os_:
            if o["role"] != "ball":
                meta.setdefault(o["tid"], o)
    order = sorted(meta, key=lambda t: ({"left": 0, "right": 1}.get(meta[t]["team"], 2),
                                        meta[t]["role"] == "referee",
                                        int(meta[t]["jersey"]) if (meta[t]["jersey"] or "").isdigit() else 99))
    present = np.zeros((len(order), len(images)), bool)
    row = {t: i for i, t in enumerate(order)}
    for f, os_ in enumerate(objs):
        for o in os_:
            if o["tid"] in row:
                present[row[o["tid"]], f] = True

    pitch = Pitch(1304, 96, 592)

    def scene_clip_frame(f, trails_len=50):
        canvas = np.full((H, W, 3), BG, np.uint8)
        img = cv2.imread(str(args.clip_dir / "img1" / images[f]["file_name"]))
        chips = draw_broadcast(canvas, img, objs[f], lines[f])
        trails = {}
        for tid, (fr, xy) in tracks.items():
            sel = (fr <= f) & (fr > f - trails_len)
            if sel.sum() > 1 and fr[sel][-1] == f:
                o = next(o for o in objs[f] if o["tid"] == tid)
                trails[tid] = [(x, y, colour(o)) for x, y in xy[sel]]
        mchips = draw_minimap(canvas, pitch, objs[f], trails)
        # timeline
        x0, x1, y0 = 120, 1880, 768
        rh = min(13, (1060 - y0) // max(1, len(order)))
        cv2.rectangle(canvas, (0, 720), (W, H), PANEL, -1)
        for i, t in enumerate(order):
            y = y0 + i * rh
            on = present[i]
            col = rgb2bgr(colour(meta[t]))
            runs = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for a, b in zip(runs[::2], runs[1::2]):
                xa = x0 + (x1 - x0) * a / len(on)
                xb = x0 + (x1 - x0) * b / len(on)
                cv2.rectangle(canvas, (int(xa), y + 2), (int(xb), y + rh - 2),
                              col if a <= f else tuple(int(c * 0.35) for c in col), -1)
        px = int(x0 + (x1 - x0) * f / len(images))
        cv2.line(canvas, (px, y0 - 6), (px, y0 + rh * len(order) + 4), (255, 255, 255), 2)
        t = Text()
        for c in chips:
            t.chip((c[0], c[1]), c[2], c[3])
        for c in mchips:
            t.chip((c[0], c[1]), c[2], c[3], "chip")
        t((1304, 24), "Minimap (pitch coordinates, m)", "h2", INK)
        t((1304, 492), f"{name}  ·  {info.get('action_class', '')}  ·  {info.get('game_time_start', '')}", "body", MUTED)
        t((1304, 522), f"t = {f / FPS:5.1f} s   frame {f + 1}/{len(images)}", "body", INK)
        vis = sum(1 for o in objs[f] if o["role"] != "ball")
        t((1304, 552), f"{vis} people labeled in this frame", "body", MUTED)
        ly = 596
        for lab_, col in (("Left team", TEAM["left"]), ("Right team", TEAM["right"]), ("Referee", REF)):
            cv2.circle(canvas, (1316, ly + 12), 7, rgb2bgr(col), -1, cv2.LINE_AA)
            t((1334, ly), lab_, "body", INK)
            ly += 30
        t((1304, ly + 4), "Square = goalkeeper. Lines on video = labeled pitch markings.", "small", MUTED)
        t((1304, ly + 26), "Numbers shown only where the jersey is labeled.", "small", MUTED)
        t((40, 738), "Track timeline", "h2", INK)
        t((300, 742), "bars = player labeled on screen; breaks = off screen or not labeled", "small", MUTED)
        for i, tid in enumerate(order):
            t((x0 - 12, y0 + i * rh + rh // 2), label(meta[tid]) or "–", "tiny", colour(meta[tid]), "rm")
        return t.render(canvas)

    # ---------------- imputation demo setup
    if args.demo_track is not None and args.demo_start is not None:
        fr_, xy_ = tracks[args.demo_track]
        sel = (fr_ >= args.demo_start) & (fr_ < args.demo_start + 75)
        if sel.sum() != 75:
            raise SystemExit("that player is not labeled for all 75 frames of the window")
        tid, f0, gt, n_cands = args.demo_track, args.demo_start, xy_[sel], 1
    else:
        (curv, tid, f0, gt), n_cands = pick_window(tracks, objs, args.results)
    m, noisy, preds, errs = impute(gt, args.results)
    gidx = np.flatnonzero(m)

    inset = Pitch(52, 806, 392)

    def demo_frame(k, reveal):
        f = f0 + k
        canvas = np.full((H, W, 3), BG, np.uint8)
        img = cv2.imread(str(args.clip_dir / "img1" / images[f]["file_name"]))
        hidden_now = bool(m[k])
        chips = draw_broadcast(canvas, img, objs[f], lines[f], hidden_tid=tid, dashed=hidden_now)
        cv2.rectangle(canvas, (0, 720), (W, H), PANEL, -1)
        mchips = draw_minimap(canvas, inset, objs[f], {}, highlight=tid)
        # zoom view (right column, equal aspect)
        zx0, zy0, zw, zh = 1310, 78, 580, 580
        ctx = slice(max(0, gidx[0] - 12), min(len(gt), gidx[-1] + 13))  # gap +/- ~0.5 s
        pts = np.vstack([gt[ctx], noisy[ctx]] + [p_[gidx] for p_ in preds.values()])
        c = (pts.min(0) + pts.max(0)) / 2
        half = (pts.max(0) - pts.min(0)).max() / 2 + 0.5
        lo, hi = c - half, c + half
        s = zw / (hi[0] - lo[0])
        Z = lambda q: (int(zx0 + (q[0] - lo[0]) * s), int(zy0 + (q[1] - lo[1]) * s))
        cv2.rectangle(canvas, (zx0 - 10, zy0 - 10), (zx0 + zw + 10, zy0 + zh + 10), PITCH_BG, -1)
        # 1 m grid for scale
        for g in np.arange(np.ceil(lo[0]), hi[0], 1.0):
            cv2.line(canvas, Z((g, lo[1])), Z((g, hi[1])), (44, 78, 60), 1)
        for g in np.arange(np.ceil(lo[1]), hi[1], 1.0):
            cv2.line(canvas, Z((lo[0], g)), Z((hi[0], g)), (44, 78, 60), 1)
        upto = min(k, len(gt) - 1)
        for j in range(ctx.start, min(upto, ctx.stop - 1) + 1):
            if not m[j]:
                cv2.circle(canvas, Z(noisy[j]), 6, (230, 230, 230), -1, cv2.LINE_AA)
        shown = [j for j in gidx if j <= upto]
        if reveal:
            seq = gt[gidx[0] - 1:gidx[-1] + 2]
            cv2.polylines(canvas, [np.array([Z(q) for q in seq], np.int32)], False,
                          (255, 255, 255), 5, cv2.LINE_AA)
        if shown:
            for key in ("linear", "kalman", "gru"):
                if key not in preds:
                    continue
                seq = [preds[key][gidx[0] - 1]] + [preds[key][j] for j in shown]
                cv2.polylines(canvas, [np.array([Z(q) for q in seq], np.int32)], False,
                              rgb2bgr(METHOD[key][0]), 3, cv2.LINE_AA)
                cv2.circle(canvas, Z(seq[-1]), 6, rgb2bgr(METHOD[key][0]), -1, cv2.LINE_AA)
        a, b = inset.p(*lo), inset.p(*hi)
        cv2.rectangle(canvas, (a[0] - 2, a[1] - 2), (b[0] + 2, b[1] + 2), (255, 255, 255), 1, cv2.LINE_AA)

        t = Text()
        for ch in chips:
            t.chip((ch[0], ch[1]), ch[2], ch[3])
        for ch in mchips:
            t.chip((ch[0], ch[1]), ch[2], ch[3], "chip")
        jer = next((o["jersey"] for o in objs[f0] if o["tid"] == tid), None)
        who = f"player #{jer}" if jer else "the player in the white box"
        t((1310, 24), "Zoom on the hidden second (grid = 1 m)", "h2", INK)
        t((1310, 672), "dots: observed positions (+ σ = 0.25 m noise)   lines: model output", "small", MUTED)
        t((40, 734), f"Imputation demo: hide 1.0 s of {who}, then reconstruct it (half speed)", "h2", INK)
        t((40, 766), "Same setup as the noisy-context experiment: noise is added to the observed frames, "
          "error is measured against the clean labels.", "small", MUTED)
        status = "GAP: positions hidden from the models" if hidden_now else (
            "After the gap" if k > gidx[-1] else "Before the gap")
        t((500, 812), status, "h2", INK if hidden_now else MUTED)
        t((500, 848), f"t = {k / FPS:4.2f} s of a 3.00 s window", "body", MUTED)
        t((500, 1002), "Example chosen by rule to be typical of the test set:", "small", MUTED)
        t((500, 1024), "median Kalman error, BiGRU/Kalman ratio near the average.", "small", MUTED)
        ly = 812
        t((1260, ly), "Mean error over the hidden second", "body", MUTED)
        ly += 40
        for key in ("linear", "kalman", "gru"):
            if key not in preds:
                continue
            col, nm = METHOD[key]
            cv2.line(canvas, (1262, ly + 16), (1296, ly + 16), rgb2bgr(col), 4, cv2.LINE_AA)
            t((1312, ly), nm, "h2", INK)
            t((1880, ly), f"{errs[key].mean() * 100:5.1f} cm" if reveal else ("…" if shown else ""),
              "h2", INK if reveal else MUTED, "ra")
            ly += 46
        if reveal:
            cv2.line(canvas, (1262, ly + 16), (1296, ly + 16), (255, 255, 255), 5, cv2.LINE_AA)
            t((1312, ly), "Ground truth (revealed)", "h2", INK)
        return t.render(canvas)

    # ---------------- stills only
    if args.stills:
        args.stills.mkdir(parents=True, exist_ok=True)
        busiest = int(np.argmax([len(o) for o in objs[:n]]))
        for f in sorted({0, busiest, n // 2, n - 1}):
            cv2.imwrite(str(args.stills / f"clip_{f:04d}.png"), scene_clip_frame(f))
        cv2.imwrite(str(args.stills / "demo_gap.png"), demo_frame(int(gidx[len(gidx) // 2]), False))
        cv2.imwrite(str(args.stills / "demo_reveal.png"), demo_frame(len(gt) - 1, True))
        print("stills ->", args.stills, "| demo track", tid, "start frame", f0, "of", n_cands, "candidates", {k: round(float(v.mean()*100),1) for k, v in errs.items()})
        return

    # ---------------- render
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(args.out)
    hold(enc, card([("Soccer Game State Reconstruction", "title", INK),
                    ("Broadcast video  →  players on a pitch minimap", "h1", MUTED),
                    (f"{name}  ·  SoccerNet-GSR ground-truth labels", "h2", FAINT)],
                   "CMU 11-785 · Group 15 · Jin, Blackwell, Stewart"), 3.0)
    for f in range(n):
        fr = scene_clip_frame(f)
        a = min(1.0, (f + 1) / 10, (n - f) / 10)
        enc.write(fade(fr, a) if a < 1 else fr)
    hold(enc, card([("Filling short gaps in a player's path", "h1", INK),
                    ("Hide 1 second of one player's track and reconstruct it", "h2", MUTED),
                    ("from the positions before and after the gap", "h2", MUTED)]), 2.5)
    for k in range(len(gt)):
        enc.write(demo_frame(k, False), repeat=2)
    last = demo_frame(len(gt) - 1, True)
    hold(enc, last, 4.0, fade_in=0.2)
    hold(enc, card([("Test set: 5,556 windows, gaps of 0.2 to 1 s", "h1", INK),
                    ("Noisy context (σ = 0.25 m):  BiGRU 15.6 cm  ·  Kalman 17.3  ·  Spline 17.9  ·  Linear 28.5", "h2", MUTED),
                    ("Clean labels:  Spline 1.0 cm  ·  Kalman 2.7  ·  BiGRU 3.1  ·  Linear 9.9", "h2", MUTED)],
                   "Mean displacement error over the hidden frames"), 4.0)
    enc.close()
    print("wrote", args.out, "| demo track", tid, "start frame", f0,
          {k: round(float(v.mean() * 100), 1) for k, v in errs.items()})


if __name__ == "__main__":
    main()
