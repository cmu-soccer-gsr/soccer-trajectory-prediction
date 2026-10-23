# Trajectory imputation: masking protocol, baselines and first GRU

Scope: fill short gaps (up to 1 s) inside on-screen player tracks. Off-screen periods are
not modeled (tracks are cut at gaps > 1 s by `scripts/extract_trajectories.py`).

## How to reproduce

```bash
pip install -r requirements.txt
# trajectories_{train,valid,test}.csv from s3://cmu-soccer-gsr-data/processed/
python scripts/run_baselines.py --processed-dir data/processed --out-dir results   # ~1 min
python scripts/train_gru.py    --processed-dir data/processed --out-dir results    # ~35 min on CPU
python scripts/make_figures.py --processed-dir data/processed --results results    # figures + tables
```

## Masking protocol (`src/imputation/data.py`)

1. Split each on-screen segment into fully observed runs (no missing frames, finite pitch
   coordinates). Real short gaps therefore never enter evaluation, since they have no
   ground truth.
2. Cut runs into 3 s windows (75 frames). Drop windows with a jump > 1 m in one frame
   (25 m/s, annotation glitches).
3. Hide a **centred** block of 5, 10, 15, 20 or 25 frames (0.2 to 1 s). The same windows
   are used for every gap length, so the curves are paired. Eval windows do not overlap:
   train 7,438 / valid 7,586 / test 5,556.
4. Error = mean Euclidean distance (cm) between prediction and ground truth over hidden
   frames (ADE), with 95% CI over windows.

### Two conditions

The GSR labels are very smooth (median per-frame acceleration about 4 mm/frame²), so on
clean ground truth even a cubic spline fills a 1 s gap within ~2.5 cm. Real tracker +
calibration output is far noisier. We therefore also report a **noisy-context** condition:
Gaussian noise (σ = 0.25 m, fixed seed per split, identical for all methods) is added to
the observed frames, and error is still measured against the clean ground truth.
σ = 0.25 m is an assumption; it should be calibrated on baseline tracker outputs.

## Methods

| Method | Details | Tuned on train |
|---|---|---|
| Linear | straight line between the two frames next to the gap | none |
| Cubic P-spline | cubic B-spline, knot every 5 frames, 2nd-difference penalty λ, fit on all observed frames | λ = 1e-5 (clean), 1 (noisy) |
| Kalman (CV + RTS) | constant-velocity model per axis, forward filter + Rauch-Tung-Striebel smoother | q = 10, r = 1e-7 (clean); q = 1e-4, r = 0.03 (noisy) |
| BiGRU | 2-layer bidirectional GRU (hidden 64, 125k params). Input per frame: position, velocity, hidden flag; hidden frames pre-filled by linear interpolation, network predicts a residual (so epoch 0 = linear) | one model for both conditions |

GRU training: train windows with stride 5 (98,646), random gap 1 to 25 frames at a random
position (at least 5 observed frames each side), random context noise σ ~ U(0, 0.4 m),
random pitch flips. Loss: Smooth L1 (β = 5 cm) on hidden frames, in metres. AdamW,
one-cycle LR 2e-3, batch 256, 15 epochs. Checkpoint selected on valid ADE (best = last epoch).

## Results (test, ADE in cm)

### Clean context

| Method | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1 s | Mean |
|---|---|---|---|---|---|---|
| Linear | 1.7 | 4.9 | 9.2 | 14.2 | 19.3 | 9.9 |
| Cubic P-spline | 0.1 | **0.2** | **0.7** | **1.6** | **2.5** | **1.0** |
| Kalman (CV + RTS) | **0.1** | 0.6 | 1.9 | 4.1 | 6.9 | 2.7 |
| BiGRU | 0.6 | 1.0 | 2.0 | 4.2 | 7.4 | 3.1 |

### Noisy context (σ = 0.25 m)

| Method | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1 s | Mean |
|---|---|---|---|---|---|---|
| Linear | 24.4 | 25.4 | 27.9 | 30.8 | 34.2 | 28.5 |
| Cubic P-spline | 11.1 | 14.1 | 17.8 | 21.4 | 25.3 | 17.9 |
| Kalman (CV + RTS) | 10.9 | 13.8 | 17.2 | 20.5 | 24.1 | 17.3 |
| BiGRU | **9.9** | **12.3** | **15.3** | **18.5** | **22.1** | **15.6** |

With noisy context the GRU beats the best classical baseline (Kalman) at every gap length,
about 10% lower error on average; it is better on 58 to 60% of individual windows
(paired Wilcoxon test, p < 1e-10 at each gap length). On clean labels the cubic spline is
best by a wide margin and the GRU is slightly behind Kalman (3.1 vs 2.7 cm).

Figures: `results/figures/error_vs_gap.png`, `gru_loss.png`, `example.png`.

## Caveats and next steps

- The noisy condition uses synthetic i.i.d. noise. Next: measure the real jitter of
  baseline tracker pitch positions and re-run with that noise (or with actual tracker
  tracks as context).
- Single-agent model. Next: add teammates/opponents as context (graph attention, see
  project references) for the longer gaps.
- Plug the best imputer into the TrackLab pipeline after tracking and measure GS-HOTA.
