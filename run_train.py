"""
Run formality-based human vs machine text detection.
From project root: python run_train.py
"""

import sys
import os

# Run from project root so 'src' is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import run_pipeline

if __name__ == "__main__":
    # ahmadreza13: balanced human vs AI; 400 per class = 800 total (faster than full 3.6M)
    run_pipeline(max_samples=400, dataset_name="ahmadreza13")
