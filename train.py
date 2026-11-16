"""Train and evaluate Random Forest and SVM phishing URL classifiers.

Usage:
    python train.py                      # uses data/urls.csv
    python train.py --data my_urls.csv   # any CSV with columns: url,label (1 = phishing)
"""
import argparse
import json
import warnings
from pathlib import Path

# scikit-learn 1.9+ warns that SVC(probability=True) will be deprecated; it still works.
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, build_feature_frame

BASE = Path(__file__).parent
OUT = BASE / "outputs"
MODELS = BASE / "models"
SEED = 42


def main(data_path: Path):
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = pd.read_csv(data_path).dropna(subset=["url", "label"])
    print(f"Loaded {len(df)} URLs | phishing share = {df.label.mean():.2%}")

    X = build_feature_frame(df["url"])
    y = df["label"].astype(int).values
    X.assign(label=y).to_csv(BASE / "data" / "features.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- Models -----------------------------------------------------------
    rf = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [None, 12], "min_samples_leaf": [1, 3]},
        cv=cv, scoring="f1", n_jobs=-1,
    )
    # SVM needs scaled features, so it lives in a Pipeline
    svm = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("svc", SVC(probability=True, random_state=SEED))]),
        {"svc__C": [1, 10], "svc__gamma": ["scale", 0.01], "svc__kernel": ["rbf"]},
        cv=cv, scoring="f1", n_jobs=-1,
    )

    results, probas = {}, {}
    models = {"Random Forest": rf, "SVM (RBF)": svm}
    for name, search in models.items():
        print(f"\nTraining {name} ...")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        probas[name] = proba
        cv_f1 = cross_val_score(best, X_train, y_train, cv=cv, scoring="f1")
        results[name] = {
            "best_params": search.best_params_,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
        }
        print(classification_report(y_test, pred, target_names=["legitimate", "phishing"]))
        print("Best params:", search.best_params_)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["legit", "phish"], cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion matrix - {name}")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_{name.split()[0].lower()}.png", dpi=150)
        plt.close(fig)

        joblib.dump(best, MODELS / f"{name.split()[0].lower()}.joblib")

    # ---- Comparison plots -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=150)
    plt.close(fig)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp = pd.DataFrame({n: [results[n][m] for m in metrics] for n in results}, index=metrics)
    ax = comp.plot.bar(figsize=(7, 4.5), ylim=(max(0, comp.values.min() - 0.1), 1.0), rot=0)
    ax.set_title("Model comparison (test set)")
    ax.set_ylabel("score")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close()

    # Random Forest feature importances
    rf_best = models["Random Forest"].best_estimator_
    imp = pd.Series(rf_best.feature_importances_, index=FEATURE_NAMES).sort_values().tail(15)
    ax = imp.plot.barh(figsize=(7, 5), color="#2a6f97")
    ax.set_title("Top 15 features - Random Forest")
    ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    comp.round(4).to_csv(OUT / "metrics.csv")
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print("\n=== Test-set summary ===")
    print(comp.round(4).to_string())
    print(f"\nSaved plots/metrics to {OUT} and models to {MODELS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data" / "urls.csv"))
    main(Path(ap.parse_args().data))
