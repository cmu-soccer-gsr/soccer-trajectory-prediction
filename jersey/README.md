# Jersey Board

Jersey number recognition in SoccerNet GSR on one page:

- **A better reader: Koshkina & Elder**: their released pipeline (legibility, ViTPose torso crop, PARSeq, weighted vote) on SNGS-021 to 025, against MMOCR, no jersey and the oracle: GS-HOTA overall and per clip, and tracklet accuracy on the same 209 tracklets.
- **What jersey numbers are worth**: GS-HOTA on the 58 validation clips with no numbers, the MMOCR baseline, and oracle numbers (per tracklet and per frame).
- **Where the reader fails**: the 3,017 scored tracklets split into right, no number, outvoted and never read, with what would fix each.
- **How many pixels a number gets**: box heights of all labeled players in the validation split.
- **What the reader is given**: 10 crops for each player in SNGS-021, enlarged or at their actual size in the frame.
- **Our own run on 5 clips**: GS-HOTA per clip for the cached state (TVCalib, NBJW) and our end-to-end run.

The MMOCR numbers are copied from `notebooks/gsr_baseline_colab_midterm.ipynb` and the Koshkina & Elder numbers from `notebooks/gsr_baseline_colab_midterm_Koshkina.ipynb` (cells K4 to K5b). Box heights and crops come from:

```bash
python scripts/export_jersey_board.py --states data/processed/game_states_valid.csv.gz \
    --clip-dir data/SNGS-021 --explorer explorer --out jersey
cd jersey && python -m http.server 8000
```

To add per-tracklet predictions, export `jersey_tracklets.csv` from the notebook (cell 15 writes it to Drive) and it can be plugged into the page.
