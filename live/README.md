# Live Board

The possession model running on a tracking feed, one frame at a time.

For whoever has the ball the board shows how the possession is likely to end (pass, loss, shot), whether they release the ball within 1 s, and their pass options: for each teammate, the chance the pass goes to them and the chance it arrives. Players outside the broadcast camera's view are drawn dashed (SkillCorner estimates their position) and the camera's view is shaded.

## Live

```bash
python scripts/live_demo.py --match-dir ~/data/skillcorner/data/matches/2011166 --start-min 10
# open http://localhost:8000
```

`live_demo.py` reads the match's tracking file and releases one frame every 100 ms on its own clock, like a provider feed. The engine (`src/imputation/live.py`) only sees frames that have arrived: it keeps the last 2 s in a ring buffer, decides who has the ball (the feed's possession field when present, otherwise the nearest player within 1.5 m of the ball), builds the features and runs the model. The page gets each frame over Server-Sent Events. The terminal prints the per-frame processing time every 10 s.

Options: `--speed 2` (double speed), `--start-min 30` (start later), `--port 8001`.

Use a test match (1886347, 1996436 or 2011166) so the model has not seen the game in training.

## Recording

```bash
python scripts/live_demo.py --match-dir ~/data/skillcorner/data/matches/2011166 --start-min 16.5 --minutes 8 --record live/replay.json
cd live && python -m http.server 8000
```

With `--record` the same engine runs over the stretch as fast as it can and saves every frame's output, the measured processing times and SkillCorner's possession events for that stretch. The page then replays it and fills in "Calls vs what happened": for each pass, the model's top option 0.5 s before the ball was played next to the actual receiver.

## Views and sources

Pick the source above the stage and switch between **Board** (top-down) and **Broadcast**.

- **Full match** (SkillCorner): there is no video in the open data, so Broadcast is a virtual camera drawn from the tracking. It follows the real broadcast camera, whose field of view on the pitch comes with every frame of the feed.
- **GSR clip SNGS-021**: the real broadcast video, with the predictions drawn through a per-frame pitch-to-image homography fitted to the labeled boxes. Made with:

```bash
python scripts/record_gsr_clip.py --explorer explorer --out live/replay_gsr.json --video broadcast_gsr.mp4
```

The GSR clip only has the players on screen (the SkillCorner feed the model was trained on also estimates the ones off screen), and it has no event record, so there is no calls table for it.

## Sections

- **How the model works**: the five steps of one forward pass (read the feed, encode, compare players, predict, draw), filled with the values for the frame on screen.
- **Calls vs what happened**: for every possession in the recording, the model's call 0.5 s before it ended next to SkillCorner's event record, with running tallies.
- **Scorecard**: test-set results on the 3 held-out matches: headline numbers with baselines, receiver accuracy by time before the pass, calibration of "pass arrives" against SkillCorner xPass, the outcome confusion matrix and a per-match table. Made with:

```bash
python scripts/export_scorecard.py --cache-dir results/skillcorner_cache --out live/scorecard.json
```

## Files

| File | What it is |
|---|---|
| `index.html` | The board. Uses the live stream when served by `live_demo.py`, otherwise loads `replay.json`. |
| `replay.json` | 8 minutes of match 2011166 (from 16:30 in the feed), made with the command above. |
| `replay_gsr.json` | SNGS-021 run through the same engine at 10 Hz, with a homography per frame. |
| `scorecard.json` | Test-set scorecard from `export_scorecard.py`. |
| `broadcast_gsr.mp4` | The clip's video (git-ignored; recreate with `record_gsr_clip.py --video`). |

Data: SkillCorner open data (A-League 2024/25), used with credit to SkillCorner.
