#!/usr/bin/env python3
"""
Train a v2 speed_lstm model (2d / 3d / combined).

Usage:
    python scripts/train_v2_cli.py --data-dir /path/to/i24_dataset --mode 3d --out runs/v2/3d
    python scripts/train_v2_cli.py --data-dir /path/to/i24_dataset --mode combined --out runs/v2/combined --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm.train import main

if __name__ == "__main__":
    main()
