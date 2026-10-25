# lift3d boxes vs Kocur boxes as LSTM features (BrnoCompSpeed)

Swaps only the 3D-box step of the Brno pipeline: the tracks (YOLO + ByteTrack), calibration
(`system_dubska_optimal_calib.json`), LSTM, fine-tuning and scoring stay those of the 1.18 km/h result.

- `run_kocur_brno.py`: the authors' YOLOv6-3D (`gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D`, Macko,
  Gajdošech & Kocur 2025) on the same frames as our YOLO run (ultralytics `vid_stride` keeps the last frame
  of each stride, so processed frame k is video frame k·step + step − 1). Saves the 8 image corners per box.
- `kocur_boxes3d.py`: each tracked box takes the Kocur box it overlaps most (IoU ≥ 0.5); its ground
  corners on the road plane give center / length / width, its vertical edges give height. Like lift3d,
  size is fixed per track (median) and centers are smoothed. Writes `boxes3d.csv` in lift3d's format.
- `scripts/brno_eval_cli.py --boxes3d` then prepares the run exactly like a lift3d run, and
  `scripts/brno_experiment.py` fine-tunes and scores it.
- Notebook: `kocur_brno_features.ipynb` (repo root).

The Kocur weights were trained on BrnoCompSpeed sessions 0-3, so its boxes on the training sessions (1-3)
are in-sample and on the test sessions (4-6) they are not.
