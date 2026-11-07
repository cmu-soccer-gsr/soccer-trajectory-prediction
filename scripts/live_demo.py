#!/usr/bin/env python3
"""
Replay a SkillCorner match as a live 10 Hz feed, run the possession model on every frame
as it arrives, and show it on the live board page.

Live (open http://localhost:8000 while it runs):
    python scripts/live_demo.py --match-dir opendata/data/matches/2013725 --start-min 10

Record a stretch for the static page instead (no server, runs as fast as it can):
    python scripts/live_demo.py --match-dir ... --start-min 10 --minutes 5 --record live/replay.json

The feed runs on its own clock. The engine only ever sees frames that have already
"arrived", and each frame's processing time (features + model) is measured and reported.
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from imputation.live import LiveEngine  # noqa: E402
from imputation.possession_model import PossessionModel  # noqa: E402
from imputation.skillcorner import FPS  # noqa: E402


def roster(meta):
    teams = {meta["home_team"]["id"]: {"name": meta["home_team"]["name"], "short": meta["home_team"].get("short_name"), "side": "home"},
             meta["away_team"]["id"]: {"name": meta["away_team"]["name"], "short": meta["away_team"].get("short_name"), "side": "away"}}
    players = {str(p["id"]): {"n": p.get("number"), "name": p.get("short_name"), "team": teams[p["team_id"]]["side"],
                              "gk": (p.get("player_role") or {}).get("acronym") == "GK"} for p in meta["players"]}
    return {"match": meta["id"], "date": meta.get("date_time"), "teams": {t["side"]: t for t in teams.values()},
            "score": [meta.get("home_team_score"), meta.get("away_team_score")],
            "pitch": [meta.get("pitch_length", 105), meta.get("pitch_width", 68)], "players": players}


def compact(fr, out):
    """What the page needs for one frame (positions to 0.1 m)."""
    b = fr.get("ball_data") or {}
    r1 = lambda v: None if v is None else round(v, 1)
    msg = {"f": fr["frame"], "t": fr.get("timestamp"), "p": fr.get("period"),
           "b": None if b.get("x") is None else [r1(b["x"]), r1(b["y"]), r1(b.get("z") or 0)],
           "pl": [[p["player_id"], r1(p["x"]), r1(p["y"]), int(bool(p.get("is_detected")))] for p in fr.get("player_data") or [] if p.get("x") is not None],
           "ms": round(out["ms"], 2)}
    cam = fr.get("image_corners_projection") or {}
    if cam.get("x_top_left") is not None:
        msg["cam"] = [r1(cam[k]) for k in ["x_top_left", "y_top_left", "x_top_right", "y_top_right",
                                           "x_bottom_right", "y_bottom_right", "x_bottom_left", "y_bottom_left"]]
    if out.get("carrier") is not None:
        msg["c"] = out["carrier"]
        msg["o"] = [round(v, 3) for v in out["outcome"]]
        msg["s"] = round(out["soon"], 3)
        msg["op"] = [[pid, round(p, 3), round(a, 3)] for pid, p, a in out["options"][:5]]
    return msg


def actual_events(md, mid, f0, f1):
    import pandas as pd
    ev = pd.read_csv(md / f"{mid}_dynamic_events.csv", low_memory=False)
    p = ev[(ev.event_type == "player_possession") & (ev.frame_end >= f0) & (ev.frame_start <= f1)]
    rows = []
    for _, r in p.iterrows():
        rows.append({"s": int(r.frame_start), "e": int(r.frame_end), "pid": int(r.player_id), "end": r.end_type,
                     "to": None if np.isnan(r.player_targeted_id) else int(r.player_targeted_id),
                     "ok": None if not isinstance(r.pass_outcome, str) else r.pass_outcome == "successful"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dir", required=True, type=Path)
    ap.add_argument("--model", default=REPO / "results" / "possession_model.pt", type=Path)
    ap.add_argument("--start-min", type=float, default=0)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--page-dir", default=REPO / "live", type=Path)
    ap.add_argument("--record", type=Path, default=None)
    a = ap.parse_args()
    mid = a.match_dir.name
    meta = json.loads((a.match_dir / f"{mid}_match.json").read_text())
    players = meta["players"]
    model = PossessionModel()
    model.load_state_dict(torch.load(a.model, map_location="cpu")["state_dict"])
    eng = LiveEngine(model, [p["id"] for p in players], [p["team_id"] for p in players],
                     [(p.get("player_role") or {}).get("acronym") == "GK" for p in players],
                     meta["home_team"]["id"], meta["home_team_side"])
    lines = (a.match_dir / f"{mid}_tracking_extrapolated.jsonl").read_bytes().splitlines()
    f0 = int(a.start_min * 60 * FPS)
    f1 = len(lines) if a.minutes is None else min(len(lines), f0 + int(a.minutes * 60 * FPS))
    # warm up the engine on the 2 s before the start so predictions begin right away
    for line in lines[max(0, f0 - 20):f0]:
        eng.step(json.loads(line))
    ros = roster(meta)

    if a.record:
        frames, ms = [], []
        for line in lines[f0:f1]:
            fr = json.loads(line)
            out = eng.step(fr)
            if out.get("carrier") is not None:
                ms.append(out["ms"])
            frames.append(compact(fr, out))
        lat = {"n": len(ms), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "max": float(np.max(ms)),
               "budget_ms": 1000 / FPS, "cpu": "1 thread"}
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps({"roster": ros, "fps": FPS, "latency": lat, "frames": frames,
                                        "events": actual_events(a.match_dir, mid, f0, f1)}, separators=(",", ":")))
        print(f"{len(frames)} frames -> {a.record}; per-frame time p50 {lat['p50']:.2f} ms, p95 {lat['p95']:.2f} ms, max {lat['max']:.2f} ms")
        return

    clients, lock, stats = [], threading.Lock(), []

    def feed():
        t_start = time.perf_counter()
        for i, line in enumerate(lines[f0:f1]):
            due = t_start + i / (FPS * a.speed)
            time.sleep(max(0.0, due - time.perf_counter()))
            fr = json.loads(line)
            out = eng.step(fr)
            if out.get("carrier") is not None:
                stats.append(out["ms"])
            msg = compact(fr, out)
            msg["lag"] = round((time.perf_counter() - due) * 1000, 1)
            data = ("data: " + json.dumps(msg, separators=(",", ":")) + "\n\n").encode()
            with lock:
                for q in list(clients):
                    q.put(data)
            if i % (FPS * 10) == 0 and stats:
                print(f"t={fr.get('timestamp')} per-frame p50 {np.percentile(stats, 50):.2f} ms, p95 {np.percentile(stats, 95):.2f} ms", flush=True)

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(a.page_dir), **kw)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/roster.json"):
                body = json.dumps({"roster": ros, "fps": FPS, "live": True}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
                return
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
                q = queue.Queue(maxsize=50)
                with lock:
                    clients.append(q)
                try:
                    while True:
                        self.wfile.write(q.get()); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with lock:
                        clients.remove(q)
                return
            super().do_GET()

    threading.Thread(target=feed, daemon=True).start()
    print(f"Live board on http://localhost:{a.port}  (match {mid}, from minute {a.start_min:g}, {a.speed:g}x)")
    ThreadingHTTPServer(("", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
