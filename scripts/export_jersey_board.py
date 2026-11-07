#!/usr/bin/env python3
"""
Data for the Jersey Board (jersey/data.json + jersey/crops.jpg).

    python scripts/export_jersey_board.py --states data/processed/game_states_valid.csv.gz \
        --clip-dir data/SNGS-021 --explorer explorer --out jersey

- Box heights of every labeled player in the validation split (58 clips): how many pixels
  a jersey number gets.
- Which tracks have a labeled number, per clip.
- Player crops from one clip (SNGS-021), 10 per track spread over the clip, with the true
  number, so you can see what the reader is given.
The midterm MMOCR numbers come from notebooks/gsr_baseline_colab_midterm.ipynb and are copied
into the page as they were printed there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CELL_W, CELL_H, PER_TRACK = 80, 128, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--clip-dir", required=True, type=Path)
    ap.add_argument("--explorer", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.states, dtype={"jersey": str})
    p = d[d.role == "player"]
    edges = list(range(0, 260, 10)) + [2000]
    hist, _ = np.histogram(p.bh, bins=edges)
    lab = p.groupby(["sequence", "track_id"]).jersey.agg(lambda s: s.notna().any())
    per_clip = lab.groupby(level=0).agg(["sum", "size"]).reset_index()
    out = {"heights": {"edges": edges, "counts": hist.tolist(), "median": float(p.bh.median()),
                       "p10": float(p.bh.quantile(.1)), "p90": float(p.bh.quantile(.9)), "n": int(len(p))},
           "labeled_tracks": {"labeled": int(lab.sum()), "tracks": int(len(lab)), "clips": int(p.sequence.nunique()),
                              "per_clip": [{"clip": r.sequence, "labeled": int(r["sum"]), "tracks": int(r["size"])} for _, r in per_clip.iterrows()]}}
    # crops
    C = json.loads((a.explorer / "clip.json").read_text())
    tracks = {int(t): m for t, m in C["tracks"].items() if m["role"] in ("player", "goalkeeper")}
    seen = {}
    for f, rows in enumerate(C["frames"]):
        for t, bx, by, bw, bh, x, y in rows:
            if t in tracks and bw > 4 and bh > 8:
                seen.setdefault(t, []).append((f, bx, by, bw, bh))
    order = sorted(seen, key=lambda t: (tracks[t]["team"], -len(seen[t])))
    rows_meta, cells = [], []
    for t in order:
        obs = seen[t]
        pick = [obs[int(i)] for i in np.linspace(0, len(obs) - 1, min(PER_TRACK, len(obs)))]
        row = []
        for f, bx, by, bw, bh in pick:
            img = Image.open(a.clip_dir / "img1" / f"{f + 1:06d}.jpg")
            pad = 0.08
            box = (max(0, bx - bw * pad), max(0, by - bh * pad), min(img.width, bx + bw * (1 + pad)), min(img.height, by + bh * (1 + pad)))
            cr = img.crop(tuple(int(v) for v in box))
            s = min(CELL_W / cr.width, CELL_H / cr.height)
            cr = cr.resize((max(1, int(cr.width * s)), max(1, int(cr.height * s))), Image.BICUBIC)
            cell = Image.new("RGB", (CELL_W, CELL_H), (20, 24, 22))
            cell.paste(cr, ((CELL_W - cr.width) // 2, CELL_H - cr.height))
            cells.append(cell)
            row.append({"f": f, "h": int(bh), "w": int(bw)})
        m = tracks[t]
        rows_meta.append({"track": t, "team": m["team"], "role": m["role"], "jersey": m.get("jersey") or None, "frames": len(obs), "crops": row})
    sheet = Image.new("RGB", (CELL_W * PER_TRACK, CELL_H * len(rows_meta)), (20, 24, 22))
    k = 0
    for r, meta in enumerate(rows_meta):
        for c in range(len(meta["crops"])):
            sheet.paste(cells[k], (c * CELL_W, r * CELL_H)); k += 1
    sheet.save(a.out / "crops.jpg", quality=82)
    out["crops"] = {"clip": C["info"]["name"], "cell": [CELL_W, CELL_H], "per_track": PER_TRACK, "fps": C["fps"], "tracks": rows_meta}
    (a.out / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    print(len(rows_meta), "tracks,", k, "crops;", (a.out / "crops.jpg").stat().st_size // 1024, "KB sprite")


if __name__ == "__main__":
    main()
