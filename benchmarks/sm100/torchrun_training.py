#!/usr/bin/env python3
"""Blackwell WeftMoE and Megatron BF16/NVFP4 training benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.training import main


if __name__ == "__main__":
    main(blackwell=True)
