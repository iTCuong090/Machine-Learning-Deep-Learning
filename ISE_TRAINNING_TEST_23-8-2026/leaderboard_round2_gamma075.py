"""Leaderboard iteration 2: change threshold gamma only, 0.80 -> 0.75."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from leaderboard_round1_build_submission import (
    ARTIFACT_DIR,
    ID_COLUMN,
    N_CLASSES,
    N_SPLITS,
    PRIOR_ALPHA,
    ROOT,
    TARGET,
    WEIGHTS,
    fit_thresholds,
    macro_f1,
)


OLD_GAMMA = 0.80
NEW_GAMMA = 0.75
OUTPUT = ROOT / "submission_iter2_gamma_0p75.csv"


def main() -> None:
    y = np.load(ARTIFACT_DIR / "target.npy")
    fold = np.load(ARTIFACT_DIR / "fold_assignment_5fold.npy")
    test_ids = pd.read_csv(ROOT / "test.csv", usecols=[ID_COLUMN])
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    oof = {
        name: np.load(ARTIFACT_DIR / f"oof_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    }
    test_probability = {
        name: np.load(ARTIFACT_DIR / f"test_{name}_5fold.npy").astype(np.float64)
        for name in WEIGHTS
    }
    oof_blend = sum(WEIGHTS[name] * oof[name] for name in WEIGHTS)
    test_blend = sum(WEIGHTS[name] * test_probability[name] for name in WEIGHTS)

    oof_adjusted = np.empty_like(oof_blend)
    for held_out in range(N_SPLITS):
        valid = fold == held_out
        count = np.bincount(y[~valid], minlength=N_CLASSES).astype(float)
        prior = count / count.sum()
        adjusted = oof_blend[valid] / np.power(prior[None, :], PRIOR_ALPHA)
        oof_adjusted[valid] = adjusted / adjusted.sum(axis=1, keepdims=True)

    scores = {OLD_GAMMA: [], NEW_GAMMA: []}
    for held_out in range(N_SPLITS):
        valid = fold == held_out
        thresholds = fit_thresholds(y, oof_adjusted, ~valid)
        for gamma in scores:
            prediction = (
                oof_adjusted[valid] / np.power(thresholds[None, :], gamma)
            ).argmax(axis=1)
            scores[gamma].append(macro_f1(y[valid], prediction))

    thresholds = fit_thresholds(y, oof_adjusted, np.ones(len(y), dtype=bool))
    count = np.bincount(y, minlength=N_CLASSES).astype(float)
    prior = count / count.sum()
    test_adjusted = test_blend / np.power(prior[None, :], PRIOR_ALPHA)
    test_adjusted /= test_adjusted.sum(axis=1, keepdims=True)
    prediction = (
        test_adjusted / np.power(thresholds[None, :], NEW_GAMMA)
    ).argmax(axis=1)

    submission = pd.DataFrame(
        {ID_COLUMN: test_ids[ID_COLUMN], TARGET: prediction.astype(np.int64)}
    )
    assert submission.shape == sample.shape
    assert submission[ID_COLUMN].equals(sample[ID_COLUMN])
    assert submission[ID_COLUMN].is_unique
    assert submission[TARGET].between(0, N_CLASSES - 1).all()
    submission.to_csv(OUTPUT, index=False)

    previous = pd.read_csv(ROOT / "submission_iter1_5fold_same_recipe.csv")
    changed = previous[TARGET].to_numpy() != submission[TARGET].to_numpy()
    report = {
        "round": 2,
        "status": "awaiting_public_score",
        "round1_public_score": 0.402,
        "single_change": f"threshold_gamma: {OLD_GAMMA} -> {NEW_GAMMA}",
        "weights": WEIGHTS,
        "prior_alpha": PRIOR_ALPHA,
        "old_gamma": OLD_GAMMA,
        "new_gamma": NEW_GAMMA,
        "old_crossfold_scores": scores[OLD_GAMMA],
        "old_crossfold_mean": float(np.mean(scores[OLD_GAMMA])),
        "new_crossfold_scores": scores[NEW_GAMMA],
        "new_crossfold_mean": float(np.mean(scores[NEW_GAMMA])),
        "oof_mean_delta": float(
            np.mean(scores[NEW_GAMMA]) - np.mean(scores[OLD_GAMMA])
        ),
        "rows_changed_vs_round1": int(changed.sum()),
        "fraction_changed_vs_round1": float(changed.mean()),
        "predicted_classes": int(submission[TARGET].nunique()),
        "missing_classes": sorted(
            set(range(N_CLASSES)) - set(submission[TARGET].unique())
        ),
        "output": OUTPUT.name,
    }
    (ARTIFACT_DIR / "round2_submission_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    history_path = ARTIFACT_DIR / "leaderboard_history.csv"
    history = pd.read_csv(history_path)
    history.loc[history["round"] == 1, ["public_macro_f1", "status"]] = [
        0.402,
        "scored",
    ]
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "round": 2,
                        "submission": OUTPUT.name,
                        "single_change": "threshold_gamma 0.80 -> 0.75",
                        "public_macro_f1": np.nan,
                        "public_test_fraction": 0.51,
                        "status": "awaiting_score",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    history.to_csv(ARTIFACT_DIR / "leaderboard_history_after_round1.csv", index=False)
    history.to_csv(history_path, index=False)
    print(json.dumps(report, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
