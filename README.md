# soccer-trajectory-prediction

Improving SoccerNet Game State Reconstruction (GSR) with soccer-specific jersey number recognition and player trajectory imputation, and predicting what the ball carrier does next from a live tracking feed.

CMU 11-785 Introduction to Deep Learning, Fall 2026 team project.

## Overview

This project builds on the SoccerNet GSR baseline and focuses on three components:

1. **Jersey number recognition**: a soccer-specific model to improve player identification from broadcast video.
2. **Trajectory imputation**: filling gaps in player tracks caused by occlusion and missed detections.
3. **Possession prediction on a live feed**: for whoever has the ball, how the possession ends (pass, loss, shot), who a pass goes to and whether it arrives, updated every frame of a 10 Hz tracking feed.

## Repository structure

```
configs/        experiment configs (yaml)
data/           datasets (not tracked; see Data section)
docs/           proposal, reports, figures
explorer/       GSR Clip Explorer: one GSR clip with video, imputation and pass/dribble predictions
jersey/         Jersey Board: where the jersey reader fails and what fixing it is worth
live/           Live Board: possession model running on a replayed tracking feed
notebooks/      exploration and analysis
results/        metrics and logs (checkpoints are not tracked)
scripts/        training / evaluation / demo entry points
src/
  jersey/       jersey number recognition
  imputation/   trajectory imputation, forecasting and possession models
  utils/        shared helpers
tests/
```

## Setup

```bash
git clone https://github.com/cmu-soccer-gsr/soccer-trajectory-prediction.git
cd soccer-trajectory-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch numpy pandas orjson   # orjson is optional, it only speeds up loading tracking files
```

## Data

Data is not committed to this repo.

### SoccerNet GSR (jersey numbers, imputation, Clip Explorer)

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
d.downloadDataTask(task="gamestate-2024", split=["train", "valid", "test", "challenge"])
```

### SkillCorner open data (possession model, Live Board)

20 full A-League 2024/25 matches: 10 fps tracking of all 22 players and the ball (off-camera players estimated, with an `is_detected` flag) and SkillCorner's event data (possessions, passes, targeted player, pass outcome). Public, about 1.8 GB, tracking files are stored with Git LFS. Please credit SkillCorner when you use it.

```bash
brew install git-lfs && git lfs install          # once
git clone https://github.com/SkillCorner/opendata.git ~/data/skillcorner
cd ~/data/skillcorner
git checkout 4340d274572876239c154c90bc507a9b3250a656   # the version our results use
git lfs pull
```

Without `git lfs` the tracking files come down as 133-byte pointer files.

## Jersey number recognition

GS-HOTA when only the jersey numbers in the pipeline output change. "Correct per tracklet" gives each predicted tracklet the number of the labeled player it is matched to most often, the ceiling for any reader that outputs one number per tracklet.

58 validation clips, released baseline output:

| Jersey numbers | GS-HOTA | DetA | AssA |
|---|---|---|---|
| None (floor) | 13.14 | 4.30 | 40.30 |
| MMOCR (baseline) | 18.85 | 7.95 | 44.73 |
| Correct per tracklet | 29.31 | 20.38 | 42.18 |

MMOCR reads the correct number for 11.3% of the 3,017 tracklets whose player has a labeled number; for 76% it reads no number at all.

5 validation clips (SNGS-021 to 025), Koshkina and Elder reader (legibility classifier, ViTPose torso crop, PARSeq) in place of MMOCR:

| Jersey numbers | GS-HOTA | DetA | AssA | Tracklet accuracy | Share of possible gain |
|---|---|---|---|---|---|
| None | 15.15 | 6.68 | 34.39 | | |
| MMOCR | 23.54 | 12.77 | 43.45 | 13.9% | 46% |
| Koshkina and Elder | 29.60 | 20.35 | 43.16 | 38.3% | 79% |
| Correct per tracklet | 33.47 | 28.34 | 39.64 | | |

The Koshkina pipeline code (`src/jersey/koshkina_bridge.py`, `src/jersey/run_koshkina.py`) is on the `koshkina` branch and the run is in `notebooks/gsr_baseline_colab_midterm_Koshkina.ipynb`. The Jersey Board (`jersey/`) shows both analyses; see [jersey/README.md](jersey/README.md).

## Trajectory imputation

Fills gaps of up to 1 s inside on-screen player tracks (linear, cubic P-spline, Kalman smoother and a bidirectional GRU). The benchmark first used i.i.d. Gaussian context noise (σ = 0.25 m), where the GRU had 10% lower error than Kalman. We then measured the real noise of the GSR pipeline against the labels on 58 valid clips:

- 24% of frames are calibration failures (median player error 17 m), mostly in bursts longer than 1 s.
- On the other frames the median error is 0.72 m, correlated in time, with a +0.3 m bias in y.
- In the pipeline's own tracks, 95.5% of gaps are 1 s or shorter, and the track ID changes 4.8 times per labeled player.

With the measured noise (test, mean ADE over 0.2 to 1 s gaps): Kalman 0.771 m, the original GRU 0.782 m, a GRU retrained on measured noise 0.700 m, and a Kalman with the mean bias removed 0.715 m. Most of the retrained GRU's gain is bias correction.

Protocol, commands and full tables: [docs/imputation.md](docs/imputation.md). Measurement: `scripts/measure_pipeline_noise.py`, `results/pipeline_noise/`; re-run: `results/pipeline_eval/`.

## Possession model

```bash
python scripts/train_possession_model.py --data-dir ~/data/skillcorner/data/matches --out-dir results
```

Splits by match with a fixed seed (14 train, 3 valid, 3 test). The first run reads every tracking file once and caches it (about 530 MB in `results/skillcorner_cache`); later runs start in seconds. About 25 minutes on a laptop CPU for 10 epochs. Writes `results/possession_model.pt`, `possession_log.csv` and `possession_metrics.json`.

Test matches (1886347, 1996436, 2011166), never seen in training:

| | Model | Reference |
|---|---|---|
| Receiver, top 1, 0.1 s before the pass | 72.3% | SkillCorner passing-option model 63.7%, nearest teammate 26.6%, random 10% |
| Receiver, top 3, same moment | 93.5% | |
| Receiver, top 1, at the start of the possession | 48.5% | SkillCorner passing-option model 63.6% |
| Pass arrives, AUC | 0.880 | SkillCorner xPass 0.879 (same 1,972 passes) |
| Outcome (pass / loss / shot), balanced accuracy | 0.712 | always "pass" 0.333 |
| Loss AUC / shot AUC | 0.827 / 0.929 | |
| Releases the ball within 1 s, AUC | 0.717 | |
| Processing time per frame (features + model, 1 CPU thread) | 2.3 ms median, 8.7 ms max | a new frame every 100 ms |

SkillCorner's passing-option model makes one call per possession, so the comparison is not frame for frame: our model is stronger just before the pass and weaker at the start of the possession.

## Demos

### Live Board (`live/`)

Replays a match as a 10 Hz feed and runs the possession model on each frame as it arrives.

```bash
python scripts/live_demo.py --match-dir ~/data/skillcorner/data/matches/2011166 --start-min 10
# open http://localhost:8000   (--speed 2 for double speed)
```

Without the server, `live/replay.json` holds an 8-minute recording of the same match (made with `--record`); serve the folder with `python -m http.server` to watch it. See [live/README.md](live/README.md).

### Jersey Board (`jersey/`)

What correct jersey numbers are worth in GS-HOTA, where MMOCR fails (with player crops at real pixel size) and the Koshkina and Elder results on 5 clips. Static page: serve the folder with `python -m http.server`. See [jersey/README.md](jersey/README.md).

### GSR Clip Explorer (`explorer/`)

One SoccerNet GSR clip (SNGS-021) with the broadcast video, labeled boxes, the trajectory imputation gap lab, 1 s path forecasts and pass/dribble predictions drawn on the video. See [explorer/README.md](explorer/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: never push directly to `main`; open a PR and get one review.

## Team

- Hojun Jin
- Robert Neil Blackwell
- Reed C. Stewart

## References

- SoccerNet Game State Reconstruction (Somers et al., CVPRW 2024)
- SoccerNet-Tracking (Cioppa et al., CVPRW 2022)
- SkillCorner Open Data, https://github.com/SkillCorner/opendata
