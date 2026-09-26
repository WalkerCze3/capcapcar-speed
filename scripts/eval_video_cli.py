#!/usr/bin/env python3
"""
Score a video_speed_cli.py run against I-24 annotations (calibration check,
per-window speed error). See speed_lstm/video_eval.py.

Usage:
    python scripts/eval_video_cli.py --run-dir runs/video/p1c1 --data-dir /path/to/i24 --scene scene1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm.video_eval import main

if __name__ == "__main__":
    main()
