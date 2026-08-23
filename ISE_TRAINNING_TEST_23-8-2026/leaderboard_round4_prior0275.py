"""Leaderboard iteration 4: nudge prior alpha only, 0.300 -> 0.275."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from leaderboard_round1_build_submission import (
    ARTIFACT_DIR,
    ID_COLUMN,
    N_CLASSES,
    ROOT,
    TARGET,
    WEIGHTS,
    fit_thresholds,
)
from leaderboard_round3_prior030 import adjusted_oof, crossfold_scores


OLD_ALPHA = 0.300
NEW_ALPHA = 0.275
GAMMA = 0.75
OUTPUT = ROOT / "submission_iter4_prior_alpha_0p275.csv"


def main() -> None:
    y = np.load(ARTIFACT_DIR / "target.npy")
    fold = np.load(ARTIFACT_DIR / "fold_assignment_5fold.npy")
    test_ids = pd.read_csv(ROOT / "test.csv", usecols=[ID_COLUMN])
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    oof_blend = sum(
        WEIGHTS[name]
        * np.load(ARTIFACT_DIR / f"oof_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    )
    test_blend = sum(
        WEIGHTS[name]
        * np.load(ARTIFACT_DIR / f"test_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    )

    old_oof = adjusted_oof(oof_blend, y, fold, OLD_ALPHA)
    new_oof = adjusted_oof(oof_blend, y, fold, NEW_ALPHA)
    old_scores = crossfold_scores(y, fold, old_oof)
    new_scores = crossfold_scores(y, fold, new_oof)

    thresholds = fit_thresholds(y, new_oof, np.ones(len(y), dtype=bool))
    count = np.bincount(y, minlength=N_CLASSES).astype(float)
    prior = count / count.sum()
    test_adjusted = test_blend / np.power(prior[None, :], NEW_ALPHA)
    test_adjusted /= test_adjusted.sum(axis=1, keepdims=True)
    prediction = (
        test_adjusted / np.power(thresholds[None, :], GAMMA)
    ).argmax(axis=1)

    submission = pd.DataFrame(
        {ID_COLUMN: test_ids[ID_COLUMN], TARGET: prediction.astype(np.int64)}
    )
    assert submission.shape == sample.shape
    assert submission[ID_COLUMN].equals(sample[ID_COLUMN])
    assert submission[ID_COLUMN].is_unique
    assert submission[TARGET].between(0, N_CLASSES - 1).all()
    submission.to_csv(OUTPUT, index=False)

    previous = pd.read_csv(ROOT / "submission_iter3_prior_alpha_0p30.csv")
    changed = previous[TARGET].to_numpy() != submission[TARGET].to_numpy()
    report = {
        "round": 4,
        "status": "awaiting_public_score",
        "round3_public_score": 0.403,
        "single_change": f"prior_alpha: {OLD_ALPHA} -> {NEW_ALPHA}",
        "weights": WEIGHTS,
        "gamma": GAMMA,
        "old_prior_alpha": OLD_ALPHA,
        "new_prior_alpha": NEW_ALPHA,
        "old_crossfold_scores": old_scores,
        "old_crossfold_mean": float(np.mean(old_scores)),
        "old_crossfold_std": float(np.std(old_scores, ddof=1)),
        "new_crossfold_scores": new_scores,
        "new_crossfold_mean": float(np.mean(new_scores)),
        "new_crossfold_std": float(np.std(new_scores, ddof=1)),
        "oof_mean_delta": float(np.mean(new_scores) - np.mean(old_scores)),
        "rows_changed_vs_round3": int(changed.sum()),
        "fraction_changed_vs_round3": float(changed.mean()),
        "predicted_classes": int(submission[TARGET].nunique()),
        "missing_classes": sorted(
            set(range(N_CLASSES)) - set(submission[TARGET].unique())
        ),
        "output": OUTPUT.name,
    }
    (ARTIFACT_DIR / "round4_submission_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    history_path = ARTIFACT_DIR / "leaderboard_history.csv"
    history = pd.read_csv(history_path)
    history.loc[history["round"] == 3, ["public_macro_f1", "status"]] = [
        0.403,
        "scored",
    ]
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "round": 4,
                        "submission": OUTPUT.name,
                        "single_change": "prior_alpha 0.300 -> 0.275",
                        "public_macro_f1": np.nan,
                        "public_test_fraction": 0.51,
                        "status": "awaiting_score",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    history.to_csv(ARTIFACT_DIR / "leaderboard_history_after_round3.csv", index=False)
    history.to_csv(history_path, index=False)
    print(json.dumps(report, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
