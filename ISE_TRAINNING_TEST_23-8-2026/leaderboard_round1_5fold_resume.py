"""Resume only missing checkpoints from leaderboard iteration 1."""

from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from leaderboard_round1_5fold import (
    ARTIFACT_DIR,
    ID_COLUMN,
    MODEL_SEED,
    N_CLASSES,
    N_JOBS,
    N_SPLITS,
    ROOT,
    SEED,
    TARGET,
    engineer,
    macro_f1,
)


def main() -> None:
    resume_started = time.perf_counter()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    features = [column for column in test.columns if column != ID_COLUMN]
    y = train[TARGET].to_numpy(dtype=np.int64)
    raw_train, raw_test = train[features], test[features]
    xgb_train, xgb_test = engineer(train, features), engineer(test, features)

    categorical = ["explicit", "key", "mode", "time_signature"]
    domains = {
        "explicit": [False, True],
        "key": list(range(12)),
        "mode": [0, 1],
        "time_signature": list(range(6)),
    }
    lgb_train, lgb_test = raw_train.copy(), raw_test.copy()
    for column in categorical:
        dtype = pd.CategoricalDtype(domains[column])
        lgb_train[column] = lgb_train[column].astype(dtype)
        lgb_test[column] = lgb_test[column].astype(dtype)

    groups = pd.util.hash_pandas_object(raw_train, index=False)
    splits = list(
        StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        ).split(raw_train, y, groups=groups)
    )
    expected_folds = np.load(ARTIFACT_DIR / "fold_assignment_5fold.npy")
    recreated_folds = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splits):
        recreated_folds[valid_idx] = fold
    assert np.array_equal(expected_folds, recreated_folds)

    names = ("extra_trees", "lightgbm", "xgboost")
    oof = {
        name: np.load(ARTIFACT_DIR / f"oof_{name}_5fold.npy") for name in names
    }
    metric_rows: list[dict] = []
    best_iterations: dict[str, list[int | None]] = {
        "lightgbm": [None] * N_SPLITS,
        "xgboost": [None] * N_SPLITS,
    }

    def complete(name: str, fold: int, valid_idx: np.ndarray) -> bool:
        test_path = ARTIFACT_DIR / f"test_{name}_fold{fold}.npy"
        row_sums = oof[name][valid_idx].sum(axis=1)
        return test_path.exists() and np.allclose(row_sums, 1, atol=2e-4)

    def record(
        name: str,
        fold: int,
        valid_idx: np.ndarray,
        seconds: float | None,
        source: str,
    ) -> None:
        probability = oof[name][valid_idx]
        prediction = probability.argmax(axis=1)
        row = {
            "model": name,
            "fold": fold,
            "macro_f1": macro_f1(y[valid_idx], prediction),
            "accuracy": float(accuracy_score(y[valid_idx], prediction)),
            "log_loss": float(
                log_loss(y[valid_idx], probability, labels=np.arange(N_CLASSES))
            ),
            "seconds_in_resume": seconds,
            "source": source,
        }
        metric_rows.append(row)
        print(
            f"{name:11s} fold={fold} F1={row['macro_f1']:.6f} "
            f"source={source}",
            flush=True,
        )

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"\n===== RESUME CHECK FOLD {fold + 1}/{N_SPLITS} =====", flush=True)

        if complete("extra_trees", fold, valid_idx):
            record("extra_trees", fold, valid_idx, None, "checkpoint")
        else:
            started = time.perf_counter()
            model = ExtraTreesClassifier(
                n_estimators=120,
                max_features=1.0,
                min_samples_leaf=2,
                n_jobs=N_JOBS,
                random_state=MODEL_SEED + fold,
            )
            model.fit(raw_train.iloc[train_idx].astype(np.float32), y[train_idx])
            oof["extra_trees"][valid_idx] = model.predict_proba(
                raw_train.iloc[valid_idx].astype(np.float32)
            ).astype(np.float32)
            np.save(
                ARTIFACT_DIR / f"test_extra_trees_fold{fold}.npy",
                model.predict_proba(raw_test.astype(np.float32)).astype(np.float32),
            )
            np.save(ARTIFACT_DIR / "oof_extra_trees_5fold.npy", oof["extra_trees"])
            record("extra_trees", fold, valid_idx, time.perf_counter() - started, "resumed")
            del model
            gc.collect()

        if complete("lightgbm", fold, valid_idx):
            record("lightgbm", fold, valid_idx, None, "checkpoint")
        else:
            started = time.perf_counter()
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                n_estimators=1_200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                max_bin=255,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                reg_alpha=0.05,
                n_jobs=N_JOBS,
                random_state=MODEL_SEED + fold,
                verbosity=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    lgb_train.iloc[train_idx],
                    y[train_idx],
                    eval_set=[(lgb_train.iloc[valid_idx], y[valid_idx])],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(60, verbose=False)],
                    categorical_feature=categorical,
                )
            oof["lightgbm"][valid_idx] = model.predict_proba(
                lgb_train.iloc[valid_idx]
            ).astype(np.float32)
            np.save(
                ARTIFACT_DIR / f"test_lightgbm_fold{fold}.npy",
                model.predict_proba(lgb_test).astype(np.float32),
            )
            np.save(ARTIFACT_DIR / "oof_lightgbm_5fold.npy", oof["lightgbm"])
            best_iterations["lightgbm"][fold] = int(model.best_iteration_)
            record("lightgbm", fold, valid_idx, time.perf_counter() - started, "resumed")
            del model
            gc.collect()

        if complete("xgboost", fold, valid_idx):
            record("xgboost", fold, valid_idx, None, "checkpoint")
        else:
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=700,
                learning_rate=0.06,
                max_depth=7,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                reg_alpha=0.05,
                objective="multi:softprob",
                num_class=N_CLASSES,
                eval_metric="mlogloss",
                tree_method="hist",
                max_bin=256,
                n_jobs=N_JOBS,
                random_state=MODEL_SEED + fold,
                early_stopping_rounds=45,
            )
            model.fit(
                xgb_train.iloc[train_idx],
                y[train_idx],
                eval_set=[(xgb_train.iloc[valid_idx], y[valid_idx])],
                verbose=False,
            )
            oof["xgboost"][valid_idx] = model.predict_proba(
                xgb_train.iloc[valid_idx]
            ).astype(np.float32)
            np.save(
                ARTIFACT_DIR / f"test_xgboost_fold{fold}.npy",
                model.predict_proba(xgb_test).astype(np.float32),
            )
            np.save(ARTIFACT_DIR / "oof_xgboost_5fold.npy", oof["xgboost"])
            best_iterations["xgboost"][fold] = int(model.best_iteration + 1)
            record("xgboost", fold, valid_idx, time.perf_counter() - started, "resumed")
            del model
            gc.collect()

    for name in names:
        assert np.isfinite(oof[name]).all()
        assert np.allclose(oof[name].sum(axis=1), 1, atol=2e-4)
        test_folds = [
            np.load(ARTIFACT_DIR / f"test_{name}_fold{fold}.npy")
            for fold in range(N_SPLITS)
        ]
        test_mean = np.mean(test_folds, axis=0, dtype=np.float64).astype(np.float32)
        assert np.allclose(test_mean.sum(axis=1), 1, atol=2e-4)
        np.save(ARTIFACT_DIR / f"test_{name}_5fold.npy", test_mean)

    metrics = pd.DataFrame(metric_rows).sort_values(["fold", "model"])
    metrics.to_csv(ARTIFACT_DIR / "base_metrics_5fold_resumed.csv", index=False)
    summary = {
        "status": "complete_after_resume",
        "n_splits": N_SPLITS,
        "split_seed": SEED,
        "best_iterations_observed_during_resume": best_iterations,
        "resume_runtime_seconds": time.perf_counter() - resume_started,
    }
    (ARTIFACT_DIR / "training_summary_5fold_resumed.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n===== COMPLETE 5-FOLD BASE SUMMARY =====")
    print(metrics.groupby("model")["macro_f1"].agg(["mean", "std"]).to_string())
    print(f"resume runtime: {summary['resume_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
