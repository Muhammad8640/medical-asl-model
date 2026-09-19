"""Train and validation-select the two specified Medical-ASL classifiers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import (
    FEATURES_PER_FRAME,
    POSE_NAMES,
    SEQUENCE_LENGTH,
    feature_layout,
    flatten_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
EFFECTIVE_TIE_TOLERANCE = 1e-6


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() and default is not None:
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _load_features(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    expected_tail = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if values.ndim != 3 or values.shape[1:] != expected_tail:
        raise ValueError(f"{path} must have shape (N, {expected_tail[0]}, {expected_tail[1]}); got {values.shape}")
    return values.reshape(values.shape[0], -1, order="C")


def _evaluate(model: Any, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    model.predict(x_val[: min(len(x_val), 2)])  # exclude one-time setup from latency
    started = time.perf_counter()
    predictions = model.predict(x_val)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "prediction_latency_ms": float(elapsed_ms / len(y_val)),
    }


def _is_better(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    f1_difference = candidate["macro_f1"] - incumbent["macro_f1"]
    if abs(f1_difference) > EFFECTIVE_TIE_TOLERANCE:
        return f1_difference > 0
    accuracy_difference = candidate["accuracy"] - incumbent["accuracy"]
    if abs(accuracy_difference) > EFFECTIVE_TIE_TOLERANCE:
        return accuracy_difference > 0
    return candidate["prediction_latency_ms"] < incumbent["prediction_latency_ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "processed")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "logs/dataset_report.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.models.mkdir(parents=True, exist_ok=True)

    x_train = _load_features(args.processed / "X_train.npy")
    y_train = np.load(args.processed / "y_train.npy", allow_pickle=False)
    x_val = _load_features(args.processed / "X_val.npy")
    y_val = np.load(args.processed / "y_val.npy", allow_pickle=False)
    # The reserved test features are deliberately not loaded or evaluated.
    y_test = np.load(args.processed / "y_test.npy", allow_pickle=False)

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("Training and validation splits must both contain usable samples")
    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise RuntimeError("Feature/label counts do not match")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("At least two classes must be present in the training split")
    val_only_classes = sorted(set(y_val.tolist()) - set(y_train.tolist()))
    if val_only_classes:
        raise RuntimeError(f"Validation classes absent from training: {val_only_classes}")

    candidates = {
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}
    selected_name: str | None = None
    for name, model in candidates.items():
        logging.info("Training candidate: %s", name)
        model.fit(x_train, y_train)
        trained[name] = model
        metrics[name] = _evaluate(model, x_val, y_val)
        logging.info(
            "%s validation: accuracy=%.4f macro_f1=%.4f latency=%.3f ms/sample",
            name,
            metrics[name]["accuracy"],
            metrics[name]["macro_f1"],
            metrics[name]["prediction_latency_ms"],
        )
        if selected_name is None or _is_better(metrics[name], metrics[selected_name]):
            selected_name = name

    assert selected_name is not None
    selected_model = trained[selected_name]
    actual_classes = [str(value) for value in selected_model.classes_]
    report = _read_json(args.report, {})
    excluded_classes = list(report.get("excluded_classes", []))
    excluded_classes.extend(
        {"class": gloss, "reason": "missing from MS-ASL metadata"}
        for gloss in report.get("missing_classes", [])
    )

    model_path = args.models / "medical_asl_model.pkl"
    joblib.dump(selected_model, model_path)
    _write_json(args.models / "label_classes.json", actual_classes)
    metadata = {
        "model_version": "1.1-msasl",
        "training_dataset": "MS-ASL medical exact-match subset",
        "dataset_split_policy": "official MS-ASL train/val/test splits preserved",
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
        "feature_vector_length": SEQUENCE_LENGTH * FEATURES_PER_FRAME,
        "flattening_order": "C order: timestep-major, feature_layout-minor",
        "random_state": RANDOM_STATE,
        "landmarks": {
            "left_hand": 21,
            "right_hand": 21,
            "pose": list(POSE_NAMES),
        },
        "feature_layout": feature_layout(),
        "normalization": {
            "origin": "3D midpoint between left and right shoulders",
            "scale": "3D Euclidean shoulder distance",
            "fallback": "centroid and maximum 3D radius of available required pose points; then available hand points; scale 1.0 if no distinct available points",
            "missing_landmarks": "all-zero slots retained after normalization",
        },
        "temporal_processing": {
            "sequence_length": SEQUENCE_LENGTH,
            "long_sequence_strategy": "30 deterministic evenly spaced frame indices including endpoints",
            "short_sequence_strategy": "per-feature linear interpolation including endpoints",
        },
        "classes": actual_classes,
        "excluded_classes": excluded_classes,
        "model_type": selected_name,
        "probability_support": True,
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "reserved_test_samples": int(len(y_test)),
        "validation_metrics": metrics[selected_name],
        "candidate_validation_metrics": metrics,
        "model_selection": {
            "primary": "macro_f1",
            "secondary": ["accuracy", "prediction_latency_ms"],
            "effective_tie_tolerance": EFFECTIVE_TIE_TOLERANCE,
            "data_used": "MS-ASL validation split only",
            "test_split_evaluated": False,
        },
    }
    _write_json(args.models / "model_metadata.json", metadata)
    logging.info("Selected model: %s", selected_name)
    logging.info("Saved model: %s", model_path)
    logging.info("Saved metadata: %s", args.models / "model_metadata.json")


if __name__ == "__main__":
    main()
