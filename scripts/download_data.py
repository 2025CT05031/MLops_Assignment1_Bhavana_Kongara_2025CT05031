"""Download raw UCI data and produce the cleaned CSV.

Usage: python scripts/download_data.py
Manual alternative: download processed.cleveland.data from
https://archive.ics.uci.edu/dataset/45/heart+disease into data/raw/
and run this script again.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import download_data, prepare_and_save  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_data()
    df = prepare_and_save()
    print(df.shape)
    print(df["target"].value_counts())
