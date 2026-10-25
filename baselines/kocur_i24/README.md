# Kocur 3D boxes on I-24 (baseline)

Runs the published Kocur 3D-box detector on I-24 3D videos and scores it against the
dataset's annotated cuboids. The method, weights and box construction are the authors'
(`gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D`: Macko, Gajdošech & Kocur 2025, the
YOLOv6 version of Kocur & Ftáčnik 2020). This folder only adapts I-24 inputs and scores
the output; it does not use our lift3d or LSTM.

- `run_kocur_i24.py`: one camera video → `detections.csv` (8 image corners per box,
  the authors' ground point, their IoU-tracker id). VPs from `hg.json`; road mask =
  hull of the annotated footprints for that camera.
- `eval_kocur_i24.py`: frame-offset search, road-plane matching, box errors (metres),
  bird's-eye IoU, per-vehicle speed error, overlays.
- `kocur.py`: imports the authors' modules, plus two compatibility shims
  (torch ≥ 2.6 checkpoint loading, numpy ≥ 2 `np.cross`).
- `i24.py`: I-24 geometry helpers.
- Notebook: `kocur_i24_baseline.ipynb` (repo root).

Notes:

- The warp needs the VP2-VP3 line outside the image. In scene1 that holds for the
  along-road cameras p2c1, p2c6, p3c1, p3c5 (and p1c1/p1c6, but their VP2 is at
  infinity); the side-looking cameras (p1c3, p1c4, p2c3, p2c4, p3c3, p3c4) and
  p1c2, p1c5, p2c2, p2c5, p3c2 fail it.
- The I-24 `hg.json` P matrices have **z pointing down**: camera centers come out at
  z ≈ −100 to −150 ft. Roofs are at z = −h. `i24.load_camera` returns the up sign.
- The earlier I-24 video test found the videos at 2× the `hg.json` resolution (1920×1080); `load_camera`
  scales by the actual video width.
- The weights were trained on BoxCars116k + BrnoCompSpeed only, so I-24 is zero-shot.
