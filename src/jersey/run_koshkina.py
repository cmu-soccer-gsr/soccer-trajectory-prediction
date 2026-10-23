#!/usr/bin/env python3
"""Run Koshkina & Elder's SoccerNet jersey pipeline on crops exported from a GSR state.

Their code uses paths relative to their repo, so run this from inside it, with
Colab's own Python (their "main" env) and conda on PATH for the other envs:

    export PATH=/opt/conda/bin:$PATH
    cd /content/jersey-number-pipeline
    python3 $REPO_DIR/src/jersey/run_koshkina.py --images /content/kosh/images --work /content/kosh/work

Their repo is not modified. We add a split called "gsr" to their config in
memory, then call their own pipeline function with these changes:

* soccer_ball_filter off. It flags small crops as the ball and labels them
  jersey 1, which would hit players far from the camera. GSR detections are
  people only, so we write an empty ball list instead.
* eval off. It needs their ground truth format; we score with GS-HOTA instead.
* Voting is theirs, unchanged. Its threshold (SUM_THRESHOLD = 1.0) needs
  enough confident frames, so export every frame (--every 1) to match their
  setup, or lower it with --sum-threshold if you subsample.
* ReID outlier filter (feat + filter) off by default, so the centroids env is
  not needed. We write a pass-through list that keeps every image. Turn it on
  with --reid-filter (needs WITH_REID=1 in the setup script).

Output: <work>/final_results.json, mapping tracklet name -> number (-1 = unreadable).
"""
import argparse
import json
import os
import sys

STEPS = ["feat", "filter", "legible", "pose", "crops", "str", "combine"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="exported crops, one folder per tracklet")
    ap.add_argument("--work", required=True, help="folder for all intermediate and final files")
    ap.add_argument("--reid-filter", action="store_true",
                    help="run their ReID + Gaussian outlier filter (needs the centroids env)")
    ap.add_argument("--from-step", choices=STEPS, default=STEPS[0],
                    help="resume from this step, reusing earlier outputs in --work")
    ap.add_argument("--sum-threshold", type=float, default=None,
                    help="override their vote threshold (default 1.0): a tracklet only gets a "
                         "number if its summed, bias-weighted confidence exceeds this. It was "
                         "tuned on full tracklets, so lower it if you export fewer frames")
    args = ap.parse_args()

    if not os.path.isfile("configuration.py") or not os.path.isfile("main.py"):
        raise SystemExit("Run this from inside the jersey-number-pipeline folder.")
    sys.path.insert(0, os.getcwd())
    import configuration as config   # their config module
    import main as kosh              # their pipeline
    import helpers                   # their voting code
    if args.sum_threshold is not None:
        helpers.SUM_THRESHOLD = args.sum_threshold
    print("Vote threshold (SUM_THRESHOLD):", helpers.SUM_THRESHOLD)

    images = os.path.abspath(args.images).rstrip("/")
    work = os.path.abspath(args.work)
    feats = os.path.join(work, "features")
    os.makedirs(feats, exist_ok=True)
    gauss_name = "main_subject_gauss_th=3.5_r=3.json"   # what their filter writes (3 rounds)

    sn = config.dataset["SoccerNet"]
    sn["root_dir"] = os.path.dirname(images)
    sn["working_dir"] = work
    sn["gsr"] = {
        "images": os.path.basename(images),
        "gt": "",
        "feature_output_folder": feats,
        "soccer_ball_list": "soccer_ball.json",
        "sim_filtered": "features/main_subject_0.4.json",
        "gauss_filtered": "features/" + gauss_name,
        "legible_result": "legible.json",
        "illegible_result": "illegible.json",
        "raw_legible_result": "raw_legible.json",
        "pose_input_json": "pose_input.json",
        "pose_output_json": "pose_results.json",
        "crops_folder": "crops",
        "jersey_id_result": "jersey_id_results.json",
        "final_result": "final_results.json",
    }

    # Empty ball list: their legibility and combine steps read this file.
    with open(os.path.join(work, "soccer_ball.json"), "w") as f:
        json.dump({"ball_tracks": []}, f)

    # Without the ReID filter, keep every image of every tracklet.
    if not args.reid_filter:
        keep = {t: sorted(os.listdir(os.path.join(images, t))) for t in os.listdir(images)}
        with open(os.path.join(feats, gauss_name), "w") as f:
            json.dump(keep, f)

    start = STEPS.index(args.from_step)
    on = {s: i >= start for i, s in enumerate(STEPS)}
    if not args.reid_filter:
        on["feat"] = on["filter"] = False
    actions = dict(on, soccer_ball_filter=False, legible_eval=False, eval=False)
    print("Steps:", ", ".join(s for s in STEPS if actions[s]))
    print("Tracklets:", len(os.listdir(images)))

    run = argparse.Namespace(dataset="SoccerNet", part="gsr", train_str=False, pipeline=actions)
    kosh.soccer_net_pipeline(run)

    out = os.path.join(work, "final_results.json")
    if not os.path.isfile(out):
        raise SystemExit("No %s: a step failed, see the log above. Fix it and rerun with "
                         "--from-step <that step>." % out)
    with open(out) as f:
        res = {k: v for k, v in json.load(f).items() if k != "imgs"}
    n_num = sum(1 for v in res.values() if 0 < int(v) < 100)
    print("Done: %s (%d tracklets, %d with a number, %d unreadable)"
          % (out, len(res), n_num, len(res) - n_num))


if __name__ == "__main__":
    main()
