"""Classify one or more URLs with the trained models.

Usage:
    python predict.py https://www.google.com http://paypal.verify-account.xyz/login
    python predict.py --file urls.txt
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

from features import build_feature_frame

MODELS = Path(__file__).parent / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--file", help="text file with one URL per line")
    args = ap.parse_args()

    urls = list(args.urls)
    if args.file:
        urls += [u.strip() for u in Path(args.file).read_text().splitlines() if u.strip()]
    if not urls:
        ap.error("give at least one URL (or --file)")

    rf = joblib.load(MODELS / "random.joblib")
    svm = joblib.load(MODELS / "svm.joblib")
    X = build_feature_frame(urls)
    p_rf = rf.predict_proba(X)[:, 1]
    p_svm = svm.predict_proba(X)[:, 1]

    for u, a, b in zip(urls, p_rf, p_svm):
        avg = (a + b) / 2
        verdict = "PHISHING  " if avg >= 0.5 else "legitimate"
        print(f"[{verdict}] RF={a:.2f} SVM={b:.2f}  {u}")


if __name__ == "__main__":
    main()
