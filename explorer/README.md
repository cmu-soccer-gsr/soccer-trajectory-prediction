# GSR Clip Explorer

Interactive viewer for one SoccerNet-GSR clip (SNGS-021): broadcast video with labeled boxes,
a synced minimap, the track timeline, 1 s path predictions for the ball carrier or any selected
player (drawn on the video through a per-frame homography), the gap lab for trajectory
imputation, and the test-set results.

## Run locally

```bash
cd explorer
python3 -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly (file://) does not work: the page loads its data with fetch.

## Files

| File | Made by |
|---|---|
| `clip.json`, `impute.json`, `results.json`, `broadcast.mp4` | `scripts/export_explorer_data.py` |
| `forecast.json` | `scripts/export_forecasts.py` (needs `results/forecaster.pt` from `scripts/train_forecaster.py`) |

`broadcast.mp4` is not committed (`*.mp4` is git-ignored). Recreate it with:

```bash
python scripts/export_explorer_data.py --clip-dir data/SNGS-021 --results results --out explorer
python scripts/export_forecasts.py --clip-dir data/SNGS-021 --results results --out explorer
```

`data/SNGS-021` is the clip folder from `valid.zip` (frames in `img1/` and `Labels-GameState.json`).
