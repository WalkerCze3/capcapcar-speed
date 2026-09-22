#!/usr/bin/env python3
"""
Predict speed from a JSON file of windows using a trained v2 checkpoint.
See speed_lstm/predict.py's module docstring for the input JSON format.

Usage:
    python scripts/predict_v2_cli.py --checkpoint runs/v2/3d/best.pt --input windows.json --out predictions.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm.predict import main

if __name__ == "__main__":
    main()
