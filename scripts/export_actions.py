#!/usr/bin/env python3
"""
Export the action model's predictions for one clip to the explorer (adds action.json).

    python scripts/export_actions.py --states data/processed/game_states_valid.csv.gz \
        --sequence SNGS-021 --results results --out explorer

For every frame where someone has the ball (with 2 s of history): the carrier, the
dribble / pass / lost probabilities, the likely receivers (visible teammates), the
3 candidate carrier paths (from the forecaster) (cm offsets from the carrier, every 0.2 s up to 1 s) with their
probabilities, and what actually happened (label and receiver, if known).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--sequence", default="SNGS-021")
    ap.add_argument("--results", default=REPO / "results", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    import torch
    from imputation.action_model import ActionModel, predict
    from imputation.events import ACTIONS, canon, build_samples, load_states, label_frame

    seqs = [s for s in load_states(a.states) if s.name == a.sequence]
    if not seqs:
        raise SystemExit(f"{a.sequence} not in {a.states}")
    s = seqs[0]
    D = build_samples([s], require_label=False)
    model = ActionModel()
    model.load_state_dict(torch.load(a.results / "action_model.pt", map_location="cpu")["state_dict"])
    P = predict(model, D)
    # carrier path: the single-player forecaster is more accurate on carriers (test ADE 0.44 m
    # vs 0.71 m for the action model's path head), so the explorer draws its paths
    from imputation.forecast import Forecaster, predict as fc_predict
    fm = Forecaster()
    fm.load_state_dict(torch.load(a.results / "forecaster.pt", map_location="cpu")["state_dict"])
    P["paths"], P["pprob"] = fc_predict(fm, D["car"])
    idx = np.arange(4, 25, 5)
    frames = {}
    for i, (_, t, c, _) in enumerate(D["meta"]):
        k = canon(s.team.get(c, "left"))
        mates = [(int(D["tids"][i][j]), int(round(P["rec"][i][j] * 100)))
                 for j in range(len(D["tids"][i])) if D["mask"][i][j] and D["ag"][i][j][4] > 0.5]
        mates = sorted([m for m in mates if m[1] >= 3], key=lambda m: -m[1])[:4]
        paths = [np.round(P["paths"][i][m, idx] * k * 100).astype(int).ravel().tolist() for m in range(P["paths"].shape[1])]
        lab = label_frame(s, t)
        actual = [ACTIONS.index(lab[0]), int(lab[1]), int(lab[2])] if lab else None
        frames[str(t)] = [int(c), [int(round(v * 100)) for v in P["act"][i]], mates,
                          [int(round(v * 100)) for v in P["pprob"][i]], paths, actual]
    carrier = [int(c) for c in s.carrier]
    out = {"actions": ACTIONS, "step": 5, "horizon": 25, "carrier": carrier, "frames": frames}
    metrics = a.results / "action_metrics.json"
    if metrics.exists():
        out["metrics"] = json.loads(metrics.read_text())
    (a.out / "action.json").write_text(json.dumps(out, separators=(",", ":")))
    print(len(frames), "frames with predictions,", (a.out / "action.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
