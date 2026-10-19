"""Train, tune, evaluate and track models with MLflow."""
import argparse
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

log = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_zoo():
    """Candidate models and their hyper-parameter grids."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {"clf__n_estimators": [100, 300],
             "clf__max_depth": [3, 5, None],
             "clf__min_samples_leaf": [1, 3]},
        ),
    }


def make_pipeline(clf):
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def _plots(model, X_test, y_test, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC - {name}")
    p = out_dir / f"roc_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {name}")
    p = out_dir / f"cm_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def train(df=None, model_path=config.MODEL_PATH, tracking_uri=None,
          outer_folds=5, inner_folds=3):
    df = load_clean() if df is None else df
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=config.RANDOM_STATE)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=config.RANDOM_STATE)

    results = {}
    for name, (clf, grid) in model_zoo().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_type", name)
            mlflow.log_params({"test_size": 0.2, "outer_cv": outer_folds,
                               "inner_cv": inner_folds, "n_features": len(config.FEATURES)})

            search = GridSearchCV(make_pipeline(clf), grid, cv=inner,
                                  scoring="roc_auc", n_jobs=-1)

            # Nested CV -> unbiased estimate of generalisation performance
            cv = cross_validate(search, X_tr, y_tr, cv=outer, scoring=SCORING, n_jobs=1)
            cv_metrics = {}
            for m in SCORING:
                cv_metrics[f"cv_{m}_mean"] = float(np.mean(cv[f"test_{m}"]))
                cv_metrics[f"cv_{m}_std"] = float(np.std(cv[f"test_{m}"]))
            mlflow.log_metrics(cv_metrics)

            # Fit on full train split with tuning, evaluate on held-out test
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_metric("best_inner_cv_roc_auc", float(search.best_score_))

            proba = best.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            test_metrics = {
                "test_accuracy": accuracy_score(y_te, pred),
                "test_precision": precision_score(y_te, pred),
                "test_recall": recall_score(y_te, pred),
                "test_f1": f1_score(y_te, pred),
                "test_roc_auc": roc_auc_score(y_te, proba),
            }
            mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})

            for p in _plots(best, X_te, y_te, name, config.REPORTS_DIR / "models"):
                mlflow.log_artifact(str(p), artifact_path="plots")

            sig = infer_signature(X_tr, best.predict_proba(X_tr)[:, 1])
            mlflow.sklearn.log_model(best, name="model", signature=sig,
                                     input_example=X_tr.head(3),
                                     serialization_format="cloudpickle")

            results[name] = {"run_id": run.info.run_id, "model": best,
                             **cv_metrics, **test_metrics}
            log.info("%s | CV ROC-AUC %.3f | test ROC-AUC %.3f", name,
                     cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"])

    # Select final model by nested-CV ROC-AUC
    best_name = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    final = results[best_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["model"], model_path)

    summary = {k: {m: v for m, v in r.items() if m != "model"} for k, r in results.items()}
    summary["selected_model"] = best_name
    config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(config.REPORTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with mlflow.start_run(run_id=final["run_id"]):
        mlflow.set_tag("selected_for_deployment", "true")
        mlflow.log_artifact(str(config.REPORTS_DIR / "metrics_summary.json"))
    log.info("Selected %s; saved to %s", best_name, model_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    args = ap.parse_args()
    train(tracking_uri=args.tracking_uri)
