"""Build iteration-1 CSV: only 3-fold -> 5-fold; keep the old recipe fixed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "leaderboard_loop"
OUTPUT = ROOT / "submission_iter1_5fold_same_recipe.csv"
TARGET = "track_genre"
ID_COLUMN = "track_id"
N_CLASSES = 112
N_SPLITS = 5
WEIGHTS = {"xgboost": 0.40, "lightgbm": 0.25, "extra_trees": 0.35}
PRIOR_ALPHA = 0.35
THRESHOLD_GAMMA = 0.80


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(N_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def fit_thresholds(
    y: np.ndarray, adjusted_probability: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    thresholds = np.empty(N_CLASSES, dtype=np.float64)
    for class_id in range(N_CLASSES):
        precision, recall, candidates = precision_recall_curve(
            y[mask] == class_id,
            adjusted_probability[mask, class_id],
        )
        binary_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
        best = int(np.nanargmax(binary_f1[:-1]))
        thresholds[class_id] = max(float(candidates[best]), 1e-8)
    return thresholds


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv", usecols=[TARGET])
    test = pd.read_csv(ROOT / "test.csv", usecols=[ID_COLUMN])
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    y = train[TARGET].to_numpy(dtype=np.int64)
    fold = np.load(ARTIFACT_DIR / "fold_assignment_5fold.npy")
    assert np.array_equal(y, np.load(ARTIFACT_DIR / "target.npy"))

    oof = {
        name: np.load(ARTIFACT_DIR / f"oof_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    }
    test_probability = {
        name: np.load(ARTIFACT_DIR / f"test_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    }
    oof_blend = sum(WEIGHTS[name] * oof[name] for name in WEIGHTS)
    test_blend = sum(
        WEIGHTS[name] * test_probability[name] for name in WEIGHTS
    )

    oof_adjusted = np.empty_like(oof_blend)
    uncalibrated_fold_scores = []
    for held_out in range(N_SPLITS):
        valid_mask = fold == held_out
        count = np.bincount(y[~valid_mask], minlength=N_CLASSES).astype(float)
        prior = count / count.sum()
        adjusted = oof_blend[valid_mask] / np.power(
            prior[None, :], PRIOR_ALPHA
        )
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        oof_adjusted[valid_mask] = adjusted
        uncalibrated_fold_scores.append(
            macro_f1(y[valid_mask], adjusted.argmax(axis=1))
        )

    calibrated_fold_scores = []
    for held_out in range(N_SPLITS):
        valid_mask = fold == held_out
        thresholds = fit_thresholds(y, oof_adjusted, ~valid_mask)
        prediction = (
            oof_adjusted[valid_mask]
            / np.power(thresholds[None, :], THRESHOLD_GAMMA)
        ).argmax(axis=1)
        calibrated_fold_scores.append(macro_f1(y[valid_mask], prediction))

    final_thresholds = fit_thresholds(
        y, oof_adjusted, np.ones(len(y), dtype=bool)
    )
    full_count = np.bincount(y, minlength=N_CLASSES).astype(float)
    full_prior = full_count / full_count.sum()
    test_adjusted = test_blend / np.power(
        full_prior[None, :], PRIOR_ALPHA
    )
    test_adjusted /= test_adjusted.sum(axis=1, keepdims=True)
    test_prediction = (
        test_adjusted / np.power(final_thresholds[None, :], THRESHOLD_GAMMA)
    ).argmax(axis=1)

    submission = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], TARGET: test_prediction.astype(np.int64)}
    )
    assert submission.shape == sample.shape
    assert list(submission.columns) == [ID_COLUMN, TARGET]
    assert submission[ID_COLUMN].equals(test[ID_COLUMN])
    assert submission[ID_COLUMN].equals(sample[ID_COLUMN])
    assert submission[ID_COLUMN].is_unique
    assert submission[TARGET].between(0, N_CLASSES - 1).all()
    submission.to_csv(OUTPUT, index=False)

    previous_path = ROOT / "submission_final_macro_f1_ensemble.csv"
    change_report = None
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        assert previous[ID_COLUMN].equals(submission[ID_COLUMN])
        changed = previous[TARGET].to_numpy() != submission[TARGET].to_numpy()
        change_report = {
            "rows_changed_vs_public_0.402": int(changed.sum()),
            "fraction_changed_vs_public_0.402": float(changed.mean()),
        }

    report = {
        "round": 1,
        "status": "awaiting_public_score",
        "baseline_public_score": 0.402,
        "baseline_public_test_fraction": 0.51,
        "single_change": "StratifiedGroupKFold n_splits: 3 -> 5",
        "weights": WEIGHTS,
        "prior_alpha": PRIOR_ALPHA,
        "threshold_gamma": THRESHOLD_GAMMA,
        "uncalibrated_fold_macro_f1": uncalibrated_fold_scores,
        "uncalibrated_mean": float(np.mean(uncalibrated_fold_scores)),
        "calibrated_fold_macro_f1": calibrated_fold_scores,
        "calibrated_mean": float(np.mean(calibrated_fold_scores)),
        "calibrated_std": float(np.std(calibrated_fold_scores, ddof=1)),
        "predicted_classes": int(submission[TARGET].nunique()),
        "missing_classes": sorted(
            set(range(N_CLASSES)) - set(submission[TARGET].unique())
        ),
        "output": OUTPUT.name,
        "change_report": change_report,
    }
    (ARTIFACT_DIR / "round1_submission_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    history = pd.DataFrame(
        [
            {
                "round": 0,
                "submission": "submission_final_macro_f1_ensemble.csv",
                "single_change": "baseline 3-fold recipe",
                "public_macro_f1": 0.402,
                "public_test_fraction": 0.51,
                "status": "scored",
            },
            {
                "round": 1,
                "submission": OUTPUT.name,
                "single_change": "n_splits 3 -> 5",
                "public_macro_f1": np.nan,
                "public_test_fraction": 0.51,
                "status": "awaiting_score",
            },
        ]
    )
    history.to_csv(ARTIFACT_DIR / "leaderboard_history.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
