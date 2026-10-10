# speed-model

Just the speed-prediction model: trajectory CSV in, trained regressor +
predictions out. No video staging, no Drive sync, no pipeline orchestration —
add that back later once this part works.

## Expected input CSV

One row per (track, frame). Column names are configurable in
`configs/default.yaml` under `columns:` — edit the right-hand side to match
your actual header, nothing else needs to change.

```
track_id,frame,x,y,w,h,speed
0,1,412,300,80,50,62.3
0,2,415,301,81,50,62.3
0,3,419,303,81,51,62.3
1,1,120,200,60,40,45.1
...
```

- `speed` is expected to repeat the same value across every row of a track
  (one ground-truth speed per vehicle, not per frame) — that's how
  VS13/I-24-style exports are typically structured. If your labels instead
  live in a separate file keyed by `track_id`, set `labels.source: file` in
  the config and point `labels.path` at it.
- `x, y` = bbox top-left corner, `w, h` = bbox width/height, in pixels.
  If your CSV instead has corner coordinates (`x1,y1,x2,y2`), convert to
  `w = x2-x1, h = y2-y1` before running this — not handled automatically.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_cli.py --csv data/your_tracks.csv --config configs/default.yaml
```

Splits by **track**, never by frame (frame-level splitting would leak the
same vehicle's trajectory across train/val/test and inflate accuracy).
Saves the best-on-validation checkpoint to `runs/default/best_model.pt`,
final held-out test MAE/RMSE printed at the end, full per-epoch history in
`runs/default/history.json`.

## Predict on new (unlabeled) trajectories

```bash
python scripts/predict_cli.py --csv data/new_tracks.csv \
    --checkpoint runs/default/best_model.pt --out predictions.csv
```

## Feature mode: `self_normalized` vs `raw`

Set in `configs/default.yaml` under `features.mode`.

- **`self_normalized`** (default): displacement and size-change are divided
  by the box's own current size (`Δx / w`, `Δw / w`, etc.), which cancels
  most of the near/far-from-camera scale difference. This is the
  improvement over the original paper's plain pixel-difference features,
  and should generalize better across cameras/distances.
- **`raw`**: plain pixel differences, matching the original paper exactly.
  Useful only if you want to A/B the two feature designs on the same data.

## 3D / metric mode (`configs/metric_3d.yaml`)

For trajectory data where position is already real-world coordinates
(e.g. I24-3D-style annotations: `x`/`y` in feet along/across the road,
plus `length`, `width`, `direction`, `timestamp`) instead of pixel bboxes.

```bash
python scripts/train_cli.py --csv data/your_3d_tracks.csv --config configs/metric_3d.yaml
```

**Read this before using it.** Because `x`/`y` here are already metric,
`Δx / Δt` is itself a usable (if noisy) speed estimate — no model required
to get *a* number. What this mode's model is actually for is **denoising**
that noisy per-frame signal: occlusion shrinks the reported box and jitters
position frame to frame, and the sequence model's job is to fuse a track's
whole noisy history into one stable estimate, better than a plain median
filter would. That's the comparison worth running (this model vs.
`median(Δx/Δt)` per track) — not "does it beat geometric calibration",
which isn't the right framing once position is already in real-world units.

If your 3D boxes instead come from your OWN detector (e.g. YOLOv6-3D) at
deployment time rather than from annotations, they were reconstructed
*using* a camera calibration in the first place — the calibration is where
the metric scale actually came from, so this mode is no longer
"calibration-free" in that setting, just a different feature representation
of the same geometry-derived information.

`data/sample_tracks_3d.csv` is synthetic (with injected timestamp jitter
and occasional occlusion-shrunk boxes) so you can see the expected schema
and sanity-check the pipeline runs before pointing it at real I24-3D data.

## Running on Colab (GPU)

No video files here, so no Drive-staging complexity needed — just upload
`data/your_tracks.csv` (or read it straight from Drive; it's small) and:

```python
!pip install -q -r requirements.txt
!python scripts/train_cli.py --csv data/your_tracks.csv --config configs/default.yaml
```

## v2: `speed_lstm/` (I-24 dataset, camera projection, speed-balanced split)

A second, more involved pipeline lives in `src/speed_lstm/`, built directly against
the raw I-24 dataset layout (`data/obj/sceneX_annotations.csv`, `data/ts/sceneX_ts.csv`,
`data/hg/sceneX_hg.json`) rather than a pre-flattened trajectory CSV. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design (feature engineering, the
speed-balanced vehicle split, model, training/normalization details). Tests covering
causality, translation invariance, split balance, and checkpoint round-tripping are in
`tests/` (`pytest`, needs `requirements-dev.txt`).

```bash
pip install -r requirements-dev.txt   # adds pytest on top of requirements.txt
python -m pytest tests/

python scripts/train_v2_cli.py --data-dir /path/to/i24_dataset --mode 3d --out runs/v2/3d
python scripts/predict_v2_cli.py --checkpoint runs/v2/3d/best.pt --input windows.json --out predictions.json
```

`--mode` is `2d`, `3d`, or `combined`. Windowing always applies the 2D-projection
filter regardless of mode, so `2d`/`3d`/`combined` runs on the same `--data-dir`,
`--scenes`, and `--seed` get the *identical* train/val/test split — required for their
test metrics to be comparable, and checked by `speed_lstm.balanced_report`.

### From raw video: detect → 3D box → v2 model

`speed_lstm/video.py` runs a trained v2 checkpoint directly on a video file:

1. **Detect + track** vehicles with Ultralytics YOLO + ByteTrack (car / motorcycle / bus / truck).
2. **Lift each 2D box to a 3D cuboid** (`speed_lstm/lift3d.py`) using the camera's 3×4
   projection matrix: the road-aligned cuboid (flat road, `z = height/2`) whose projected
   corners best match the detected box. Dimensions are regularized toward a per-class prior,
   then fixed to each track's median so only position varies frame to frame. Boxes clipped by
   the image edge are dropped. For an I-24 `hg.json`, each track's direction (EB/WB P matrix)
   is picked from its motion.
3. **Window + predict**: 16 consecutive frames per window (stride 8, split at tracking gaps),
   passed to `Predictor.predict(timestamps, boxes2d, boxes3d)` in the same format as training:
   `boxes3d = [cx, cy, cz, length, width, height]` in metres, `boxes2d` = the projected
   cuboid's xyxy box.

```bash
pip install -r requirements-video.txt   # adds ultralytics + opencv
python scripts/video_speed_cli.py --video p1c1.mp4 --checkpoint runs/v2/3d/best.pt \
    --calib /path/to/i24_dataset/hg/scene1_hg.json --camera p1c1 --render
```

Timestamps default to `frame / fps`; pass `--ts-csv data/ts/scene1_ts.csv` (with `--camera`) to
use I-24's corrected per-camera timestamps. `--calib` also accepts a single-camera
`{"P": [[...],[...],[...]]}` json; `--calib-units` says which world units P expects (`ft`
by default, since I-24 calibrations are in feet).

Outputs in `runs/video/<video name>/`: `detections.csv`, `boxes3d.csv` (per-frame 3D boxes +
reprojection error), `windows.json` (the exact model inputs, same format `predict_v2_cli.py`
reads), `window_predictions.csv`, `track_speeds.csv` (median per vehicle), and
`annotated.mp4` with `--render`. `geometric_mps` in the outputs is path length / time over
the same 3D centers (the training target's formula) — a useful sanity check against the model.

Caveat: the model was trained on annotation-derived cuboids, and these are monocular fits
from detector boxes. The metric scale comes entirely from the calibration, so a wrong P (or
wrong `--calib-units`) scales every speed.

#### BrnoCompSpeed

`scripts/brno_eval_cli.py` runs the same pipeline on a BrnoCompSpeed recording and scores it
with the dataset's official rules (`speed_lstm/brno.py`, ported from
[JakubSochor/BrnoCompSpeed](https://github.com/JakubSochor/BrnoCompSpeed)): the camera P is
built from the recording's calibration (vanishing points + scale, e.g.
`results/session4_center/system_dubska_optimal_calib.json`), tracks are matched to
ground-truth cars by their last-measurement-line crossing (±0.2 s, same lane), and errors
are in km/h over valid cars — for the model and for the official geometric speeds of the
same lifted trajectory (per-frame median, the official default, and line-to-line). The
calibration file's own tracks (the dataset's reference system) are scored the same way as a
reference. `speed_video_brno_test.ipynb` runs it in Colab.

```bash
python scripts/brno_eval_cli.py --session-dir .../dataset/session4_center \
    --calib .../results/session4_center/system_dubska_optimal_calib.json \
    --checkpoint runs/v2/3d/best.pt --max-seconds 600 --out-dir runs/brno/session4_center
```

Brno video is 50 fps; `--frame-step 2` (default) feeds the model 25 fps, closer to the
~30 fps it was trained on. The AVIs declare 100 fps (padded with empty packets OpenCV skips),
so time is decoded frame / the ground truth's fps, as in the official evaluation.

#### Fine-tuning on Brno (split C: train sessions 0–3, test sessions 4–6)

1. **Prepare** each recording once with `brno_eval_cli.py`: besides scoring the base model it
   caches `windows.json` (model-ready inputs) and `labeled_windows.json` (windows of matched cars,
   labeled with their measured speed).
2. **Experiment** with `scripts/brno_experiment.py --config '<json>'`: fine-tunes a checkpoint on
   the training recordings' labeled windows (`speed_lstm/finetune.py`; model selection on per-car
   validation error in km/h), re-predicts the test recordings' cached windows and scores them with
   the official matching — no detection or lifting is re-run, so an experiment takes seconds.
   Results go to `runs/experiments/<id>/results.json` and a shared `leaderboard.csv`.
3. **Automate** with `brno_finetune_worker.ipynb` in Colab: it runs `scripts/colab_worker.py`,
   which works through `experiments/queue.json` and pulls this branch between jobs, so pushing new
   jobs to the queue is all it takes to run more experiments.

Center cameras, first 10 min of each recording (train sessions 1–3, test sessions 4–6, 489 cars):

| | test MAE (km/h) |
|---|---|
| I-24 checkpoint, no fine-tuning | 6.75 |
| official geometric speed of the same lifted tracks | 1.86 |
| Brno reference system (Dubska, optimal calibration), same cameras | ≈1.35 |
| fine-tuned (all layers, I-24 normalization kept, lr 1e-3, batch 64), 3-seed ensemble | **1.18** |

What mattered: keep the I-24 feature/target normalization (refitting it, freezing the LSTM,
head-only, or training from scratch were all worse), and enough optimizer steps (batch 64 or
more epochs). Seed-to-seed spread is ~±0.1 km/h, so compare seed triples, not single runs.

#### Automatic calibration (no manual calibration)

`speed_lstm/autocalib.py` builds P from the traffic itself, so a new camera needs no `--calib`:

1. **VP1** (along the road) from the lines that tracked box centers move along, intersected on the
   Gaussian sphere with a robust (Cauchy) loss; boxes cut off by the image border are dropped first.
2. **VP1 / VP2 per vehicle** from a CNN on vehicle crops (`speed_lstm/vp_cnn.py`, after Kocur and
   Ftáčnik 2021, [deep_vp](https://github.com/kocurvik/deep_vp)), aggregated robustly on the Gaussian
   sphere. Track and CNN VP1 are averaged when they agree within 2°; otherwise the one under which
   cars keep a steady speed along the road wins. VP2 always comes from the CNN.
3. **Focal length and road plane** from VP1, VP2 and the image center (`brno.compute_camera_calibration`).
4. **Scale** from car size: the value at which the farther half of the car boxes is, in the median,
   as high relative to a 4.6 × 1.85 × 1.55 m cuboid placed at the same spot as on calibrated cameras
   (`BOX_HEIGHT_LOG_RATIO`, fitted on BrnoCompSpeed sessions 1-2; a detector box is about 15% lower
   than the cuboid's). It depends on the detector and tracker: refit it with
   `autocalib.fit_box_height_log_ratio` on training sessions if those change.

On sessions 1-3 with the dataset's VP2 in place of the CNN, along-road distances come out 2.8% off on
average with this (6.8% worst), against 0-2% for the dataset's own calibration; the constants were
fitted on the other sessions each time. VP2 is now the error that matters: 1° off gives about 5%.

The output is a BrnoCompSpeed-style system file, so it is scored exactly like the dataset's calibrations:

```bash
# train the VP CNN on training sessions only (labels: VP1 from each recording's annotated lane dividers,
# VP2 from its calibration file)
python scripts/train_vp_cnn.py --dataset-root .../2016-ITS-BrnoCompSpeed --prepared-root runs/brno \
    --train session0_center session1_center session2_center --val session3_center --out runs/vp_cnn
# calibrate a test recording automatically and compare with the dataset's calibration
python scripts/autocalib_cli.py --video .../dataset/session4_center/video.avi --vp-model runs/vp_cnn/best.pt \
    --detections runs/brno/session4_center/detections.csv --frame-step 2 \
    --mask .../dataset/session4_center/video_mask.png --out runs/autocalib/session4_center.json \
    --compare .../results/session4_center/system_dubska_optimal_calib.json
# score speeds with it
python scripts/brno_eval_cli.py --session-dir .../dataset/session4_center \
    --calib runs/autocalib/session4_center.json --checkpoint runs/v2/3d/best.pt \
    --detections runs/brno/session4_center/detections.csv --out-dir runs/brno_auto/session4_center
```

The CNN (ResNet-18 from ImageNet weights by default, or a small from-scratch net with `--arch small`) sees
only a handful of training cameras, so each training crop is also warped by a random homography with
its VP labels mapped by the same homography, which is exact because VPs are points. As in deep_vp, the
warp is applied to a larger context crop and the crop is then re-fitted to the warped vehicle box.

On Colab, `scripts/vp_cnn_job.py` does all of the above: it trains each `--archs` entry, keeps the best
on validation, then calibrates every `--val` (or `--eval`) recording and scores it with `brno_eval_cli.py`,
next to the dataset's own calibration scored the same way, writing `results_<tag>.md` to
`runs/vp_cnn/<name>/` in the project folder. `experiments/queue.json` runs it as three jobs writing to
`runs/vp_cnn/vp_cnn_v1`: `vp_cnn_v1_train` (sessions 0-2, validated on session 3), `vp_cnn_v1_val`
(session 3, `results_val.md`) and `vp_cnn_v1_test` (sessions 4-6, on hold until the setup is chosen).

`video_speed_cli.py --vp-model runs/vp_cnn/best.pt` (no `--calib`) does the same on any video and
writes `auto_calib.json` next to its outputs; the file's `reliable` flag and `quality` say whether
the calibration passed its checks (enough straight tracks from more than one lane, cars keeping their
speed along the road, enough different cars, scale stable across halves of the cars and between
near and far cars).

## What's NOT in this scaffold (on purpose)

- No detection/tracking for the v1 `speedmodel/` path — it assumes trajectory CSVs already
  exist (the v2 path has `video_speed_cli.py`, above)
- No Google Drive sync or checkpoint-across-sessions — training is a single
  run of at most a few minutes on this kind of data, not hours of video
  inference, so if Colab disconnects you just re-run `train_cli.py`
- No web app / dashboard integration
