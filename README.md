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

## What's NOT in this scaffold (on purpose)

- No detection/tracking for the v1 `speedmodel/` path — it assumes trajectory CSVs already
  exist (the v2 path has `video_speed_cli.py`, above)
- No Google Drive sync or checkpoint-across-sessions — training is a single
  run of at most a few minutes on this kind of data, not hours of video
  inference, so if Colab disconnects you just re-run `train_cli.py`
- No web app / dashboard integration
