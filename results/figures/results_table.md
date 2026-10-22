### Clean context

| Method | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1 s | Mean | vs linear |
|---|---|---|---|---|---|---|---|
| Linear | 1.7 | 4.9 | 9.2 | 14.2 | 19.3 | 9.9 |  |
| Cubic P-spline | 0.1 | **0.2** | **0.7** | **1.6** | **2.5** | **1.0** | -90% |
| Kalman (CV + RTS) | **0.1** | 0.6 | 1.9 | 4.1 | 6.9 | 2.7 | -72% |
| BiGRU (ours) | 0.6 | 1.0 | 2.0 | 4.2 | 7.4 | 3.1 | -69% |

### Noisy context (σ = 0.25 m)

| Method | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1 s | Mean | vs linear |
|---|---|---|---|---|---|---|---|
| Linear | 24.4 | 25.4 | 27.9 | 30.8 | 34.2 | 28.5 |  |
| Cubic P-spline | 11.1 | 14.1 | 17.8 | 21.4 | 25.3 | 17.9 | -37% |
| Kalman (CV + RTS) | 10.9 | 13.8 | 17.2 | 20.5 | 24.1 | 17.3 | -39% |
| BiGRU (ours) | **9.9** | **12.3** | **15.3** | **18.5** | **22.1** | **15.6** | -45% |

Test ADE in cm (mean displacement over hidden frames), 5556 windows of 3 s, centred gap. Bold = best per column.