#!/usr/bin/env python3
"""
Video -> YOLO + ByteTrack -> 3D cuboids (via camera P) -> v2 speed model.
See speed_lstm/video.py for the pipeline and output files.

Usage:
    python scripts/video_speed_cli.py --video cam.mp4 --checkpoint runs/v2/3d/best.pt \
        --calib data/hg/scene1_hg.json --camera p1c1 --render
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm.video import main

if __name__ == "__main__":
    main()
