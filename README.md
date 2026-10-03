# soccer-trajectory-prediction

Improving SoccerNet Game State Reconstruction (GSR) with soccer-specific jersey number recognition and player trajectory imputation.

CMU 11-785 Introduction to Deep Learning, Fall 2026 team project.

## Overview

This project builds on the SoccerNet GSR baseline and focuses on two components:

1. **Jersey number recognition**: a soccer-specific model to improve player identification from broadcast video.
2. **Trajectory imputation**: filling gaps in player tracks caused by occlusion and missed detections.

## Repository structure

```
configs/        experiment configs (yaml)
data/           datasets (not tracked; see Data section)
docs/           proposal, reports, figures
notebooks/      exploration and analysis
scripts/        training / evaluation entry points
src/
  jersey/       jersey number recognition
  imputation/   trajectory imputation
  utils/        shared helpers
tests/
```

## Setup

```bash
git clone https://github.com/cmu-soccer-gsr/soccer-trajectory-prediction.git
cd soccer-trajectory-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Data is not committed to this repo. Download SoccerNet GSR data into `data/`:

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
d.downloadDataTask(task="gamestate-2024", split=["train", "valid", "test", "challenge"])
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: never push directly to `main`; open a PR and get one review.

## Team

- Hojun Jin
- Robert Neil Blackwell
- Reed C. Stewart

## References

- SoccerNet Game State Reconstruction (Somers et al., CVPRW 2024)
- SoccerNet-Tracking (Cioppa et al., CVPRW 2022)
