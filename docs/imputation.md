# Trajectory imputation: masking protocol, baselines, GRU and real pipeline noise

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

Re-run with the measured pipeline noise (section "Real pipeline noise" below):

```bash
# baseline-valid-slim.pklz from s3://cmu-soccer-gsr-data/states/, game_states_valid.csv.gz from the GSR labels
python scripts/measure_pipeline_noise.py --state baseline-valid-slim.pklz --gt game_states_valid.csv.gz --out results/pipeline_noise
python scripts/run_baselines.py --processed-dir data/processed --out-dir results/pipeline_eval/base --noise 0 0.25 pipeline:0.5 pipeline pipeline:1.5
python scripts/train_gru.py --processed-dir data/processed --out-dir results/pipeline_eval/gru_old --ckpt results/gru.pt --name gru --eval-noise 0 0.25 pipeline:0.5 pipeline pipeline:1.5
python scripts/train_gru.py --processed-dir data/processed --out-dir results/pipeline_eval/gru_pn --train-noise mix --name gru_pn --eval-noise 0 0.25 pipeline:0.5 pipeline pipeline:1.5
python scripts/summarize_pipeline_eval.py --processed-dir data/processed --eval-dir results/pipeline_eval
```

The last step reads the `metrics.csv` written by the three runs before it. Its outputs,
`results/pipeline_eval/metrics_test_all.csv`, `summary.csv` and `pipeline_noise_sweep.png`,
are committed. The checkpoint `gru_pn.pt` is not tracked; it is shared through
`s3://cmu-soccer-gsr-data/checkpoints/`.

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
σ = 0.25 m was an assumption. It has since been replaced by noise measured on the real
pipeline (next section), which is about three times larger and correlated in time.

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

## Real pipeline noise

`scripts/measure_pipeline_noise.py` compares the GSR baseline's pitch positions
(`bbox_pitch` bottom-middle in `baseline-valid-slim.pklz`) with the labels on all 58 valid
clips. Detections are matched to labeled boxes per frame in image space (Hungarian, IoU ≥ 0.5),
so association errors do not leak into the position error: 568,715 matched pairs. Full
numbers in `results/pipeline_noise/noise.json`.

| | Value |
|---|---|
| Frames with a calibration failure (median player error > 3 m) | 24% (median error 17 m inside them) |
| Failure frames in bursts longer than 1 s | 79% |
| Clips with more than half / fewer than 5% failed frames | 9 / 10 |
| Failures caught by "any player off the pitch" (no labels needed) | 49% recall, 96% precision |
| Median error on the other frames | 0.72 m |
| Mean bias on the other frames (x, y) | (+0.05, +0.30) m |
| Variance: shift shared by all players in a frame / constant per track / per player | 67% / 16% / 17% |
| Frame-to-frame jitter (x, y) | 0.30 / 0.45 m |
| Error, near touchline to far touchline | 0.63 m to 1.17 m |
| Detection recall / precision (IoU ≥ 0.5) | 82% / 94% |
| Pipeline track ID changes per labeled track | 4.8 |
| Gaps inside labeled tracks: median length, share up to 1 s | 1 frame, 95.5% |

The last two rows are the case for this task: in the pipeline's own output almost every gap
is short, and tracks break often.

### Noise model (`src/imputation/noise.py`)

Per axis, per window: `e(t) = offset + slow(t) + white(t)`. The offset is constant over the
window, `slow` is an AR(1) process and `white` is i.i.d. Parameters are fitted on runs of at
least 3 s in good frames and stored in `results/pipeline_noise/noise_model.json`:

| | x | y |
|---|---|---|
| Offset mean / std (m) | 0.06 / 0.45 | 0.30 / 0.44 |
| AR(1) std (m) / coefficient per frame | 0.33 / 0.936 | 0.41 / 0.923 |
| White std (m) | 0.20 | 0.31 |

Simulated frame-to-frame jitter (0.31 / 0.46 m) matches the measured one. Calibration
failures are not part of the model; they need to be detected and treated as missing.

`evaluate()` takes a float (i.i.d. Gaussian, as before) or a spec such as `pipeline` or
`pipeline:0.5` (measured noise scaled by 0.5). It reports `ade_m` against the clean labels
and `ade_track_m` against the labels plus the window's constant offset, i.e. how well the
gap continues the observed track. `train_gru.py --train-noise mix` trains on i.i.d. noise
for half the windows and measured noise at a random scale in [0, 1.5] for the other half.

### Results with measured noise (test, ADE in m, mean over 0.2 to 1 s gaps)

| Method | i.i.d. σ = 0.25 m | Measured ×0.5 | Measured ×1 | Measured ×1.5 |
|---|---|---|---|---|
| Linear | 0.285 | 0.423 | 0.808 | 1.200 |
| Cubic P-spline | 0.179 | 0.410 | 0.772 | 1.114 |
| Kalman (CV + RTS) | 0.173 | 0.410 | 0.771 | 1.116 |
| Kalman, bias-corrected | | 0.383 | 0.715 | 1.030 |
| BiGRU, trained on i.i.d. noise | **0.156** | 0.407 | 0.782 | 1.151 |
| BiGRU, trained on measured noise | 0.163 | **0.373** | **0.700** | **1.019** |

"Bias-corrected" subtracts the measured mean offset from the observations before filtering,
with the same tuned q and r.

- The first GRU's 10% lead over Kalman does not survive realistic noise: it is slightly
  worse than Kalman at scale 1 and 1.5.
- Trained on the measured noise, the GRU is best again, but most of its gain over Kalman
  (0.771 to 0.700) comes from removing the systematic offset; a bias-corrected Kalman
  reaches 0.715.
- On `ade_track_m` (offset removed from the target) Kalman and spline are best (0.437 and
  0.439 vs 0.477 for the retrained GRU), so the GRU corrects bias but does not continue
  tracks better.
- Error barely grows from 0.2 s to 1 s gaps: observation noise now dominates the gap.

Figure: `results/pipeline_eval/pipeline_noise_sweep.png`.

## Caveats and next steps

- Calibration failures (24% of frames) are the largest error source and no imputer fixes
  them. Next: detect them without labels and treat those frames as missing.
- Run imputation on the pipeline's own tracks (not simulated gaps), where the target is the
  labeled position of the matched player.
- Single-agent model. Next: add teammates/opponents as context (graph attention, see
  project references) for the longer gaps.
- Plug the best imputer into the TrackLab pipeline after tracking and measure GS-HOTA.
