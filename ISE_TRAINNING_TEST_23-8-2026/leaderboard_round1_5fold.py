"""Leaderboard iteration 1: train and persist a fresh 5-fold ensemble.

The final notebook/report is intentionally not modified during the five-round
leaderboard loop.  This script saves reusable OOF/test probabilities so later
rounds can explore changes without retraining these models.
"""

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


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "leaderboard_loop"
SEED = 20_260_823
MODEL_SEED = 42
N_SPLITS = 5
N_CLASSES = 112
TARGET = "track_genre"
ID_COLUMN = "track_id"
N_JOBS = 8


def engineer(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    key = result.pop("key").astype(int)
    time_signature = result.pop("time_signature").astype(int)
    for value in range(12):
        result[f"key_{value}"] = (key == value).astype(np.int8)
    for value in range(6):
        result[f"time_signature_{value}"] = (time_signature == value).astype(np.int8)
    angle = 2 * np.pi * key / 12
    result["key_sin"] = np.sin(angle)
    result["key_cos"] = np.cos(angle)
    fifth_angle = 2 * np.pi * ((key * 7) % 12) / 12
    result["circle_of_fifths_sin"] = np.sin(fifth_angle)
    result["circle_of_fifths_cos"] = np.cos(fifth_angle)
    key_mode = key + 12 * result["mode"].astype(int)
    for value in range(24):
        result[f"key_mode_{value}"] = (key_mode == value).astype(np.int8)
    result["acoustic_low_energy"] = result["acousticness"] * (1 - result["energy"])
    result["energy_loudness"] = result["energy"] * (result["loudness"] + 60)
    result["dance_energy"] = result["danceability"] * result["energy"]
    result["dance_valence"] = result["danceability"] * result["valence"]
    result["speech_explicit"] = result["speechiness"] * (
        1 + result["explicit"].astype(float)
    )
    result["instrumental_acoustic"] = (
        result["instrumentalness"] * result["acousticness"]
    )
    result["tempo_dance"] = result["tempo"] * result["danceability"]
    result["audio_missing"] = (
        result[["danceability", "speechiness", "valence", "tempo"]]
        .eq(0)
        .all(axis=1)
        .astype(np.int8)
    )
    return result.astype(np.float32)


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


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    features = [column for column in test.columns if column != ID_COLUMN]
    y = train[TARGET].to_numpy(dtype=np.int64)
    raw_train = train[features]
    raw_test = test[features]
    xgb_train = engineer(train, features)
    xgb_test = engineer(test, features)

    categorical = ["explicit", "key", "mode", "time_signature"]
    category_domains = {
        "explicit": [False, True],
        "key": list(range(12)),
        "mode": [0, 1],
        "time_signature": list(range(6)),
    }
    lgb_train = raw_train.copy()
    lgb_test = raw_test.copy()
    for column in categorical:
        dtype = pd.CategoricalDtype(category_domains[column])
        lgb_train[column] = lgb_train[column].astype(dtype)
        lgb_test[column] = lgb_test[column].astype(dtype)

    assert np.array_equal(np.unique(y), np.arange(N_CLASSES))
    assert not raw_train.isna().any().any() and not raw_test.isna().any().any()
    group_id = pd.util.hash_pandas_object(raw_train, index=False)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    splits = list(splitter.split(raw_train, y, groups=group_id))
    fold_assignment = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splits):
        fold_assignment[valid_idx] = fold
    assert np.all(fold_assignment >= 0)
    assert all(np.unique(y[valid_idx]).size == N_CLASSES for _, valid_idx in splits)
    assert (
        pd.DataFrame({"group": group_id, "fold": fold_assignment})
        .groupby("group")["fold"]
        .nunique()
        .max()
        == 1
    )
    np.save(ARTIFACT_DIR / "fold_assignment_5fold.npy", fold_assignment)
    np.save(ARTIFACT_DIR / "target.npy", y)
    (ARTIFACT_DIR / "test_track_id.csv").write_text(
        test[[ID_COLUMN]].to_csv(index=False), encoding="utf-8"
    )

    model_names = ("extra_trees", "lightgbm", "xgboost")
    oof = {
        name: np.zeros((len(train), N_CLASSES), dtype=np.float32)
        for name in model_names
    }
    metric_rows: list[dict] = []
    best_iterations = {"lightgbm": [], "xgboost": []}

    def record(name: str, fold: int, valid_idx: np.ndarray, probability: np.ndarray, seconds: float) -> None:
        prediction = probability.argmax(axis=1)
        row = {
            "model": name,
            "fold": fold,
            "macro_f1": macro_f1(y[valid_idx], prediction),
            "accuracy": float(accuracy_score(y[valid_idx], prediction)),
            "log_loss": float(
                log_loss(y[valid_idx], probability, labels=np.arange(N_CLASSES))
            ),
            "seconds": seconds,
        }
        metric_rows.append(row)
        print(
            f"{name:11s} fold={fold} F1={row['macro_f1']:.6f} "
            f"acc={row['accuracy']:.6f} time={seconds:.1f}s",
            flush=True,
        )

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(
            f"\n===== 5-FOLD {fold + 1}/{N_SPLITS}: "
            f"train={len(train_idx):,} valid={len(valid_idx):,} =====",
            flush=True,
        )

        started = time.perf_counter()
        model = ExtraTreesClassifier(
            n_estimators=120,
            max_features=1.0,
            min_samples_leaf=2,
            n_jobs=N_JOBS,
            random_state=MODEL_SEED + fold,
        )
        model.fit(raw_train.iloc[train_idx].astype(np.float32), y[train_idx])
        probability = model.predict_proba(raw_train.iloc[valid_idx].astype(np.float32))
        oof["extra_trees"][valid_idx] = probability.astype(np.float32)
        np.save(
            ARTIFACT_DIR / f"test_extra_trees_fold{fold}.npy",
            model.predict_proba(raw_test.astype(np.float32)).astype(np.float32),
        )
        record("extra_trees", fold, valid_idx, probability, time.perf_counter() - started)
        np.save(ARTIFACT_DIR / "oof_extra_trees_5fold.npy", oof["extra_trees"])
        del model, probability
        gc.collect()

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
        probability = model.predict_proba(lgb_train.iloc[valid_idx])
        oof["lightgbm"][valid_idx] = probability.astype(np.float32)
        np.save(
            ARTIFACT_DIR / f"test_lightgbm_fold{fold}.npy",
            model.predict_proba(lgb_test).astype(np.float32),
        )
        best_iterations["lightgbm"].append(int(model.best_iteration_))
        record("lightgbm", fold, valid_idx, probability, time.perf_counter() - started)
        np.save(ARTIFACT_DIR / "oof_lightgbm_5fold.npy", oof["lightgbm"])
        del model, probability
        gc.collect()

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
        probability = model.predict_proba(xgb_train.iloc[valid_idx])
        oof["xgboost"][valid_idx] = probability.astype(np.float32)
        np.save(
            ARTIFACT_DIR / f"test_xgboost_fold{fold}.npy",
            model.predict_proba(xgb_test).astype(np.float32),
        )
        best_iterations["xgboost"].append(int(model.best_iteration + 1))
        record("xgboost", fold, valid_idx, probability, time.perf_counter() - started)
        np.save(ARTIFACT_DIR / "oof_xgboost_5fold.npy", oof["xgboost"])
        del model, probability
        gc.collect()

    for name in model_names:
        assert np.isfinite(oof[name]).all()
        assert np.allclose(oof[name].sum(axis=1), 1, atol=2e-4)
        test_folds = [
            np.load(ARTIFACT_DIR / f"test_{name}_fold{fold}.npy")
            for fold in range(N_SPLITS)
        ]
        test_mean = np.mean(test_folds, axis=0, dtype=np.float64).astype(np.float32)
        assert np.allclose(test_mean.sum(axis=1), 1, atol=2e-4)
        np.save(ARTIFACT_DIR / f"test_{name}_5fold.npy", test_mean)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACT_DIR / "base_metrics_5fold.csv", index=False)
    summary = {
        "n_splits": N_SPLITS,
        "split_seed": SEED,
        "model_seed_base": MODEL_SEED,
        "best_iterations": best_iterations,
        "runtime_seconds": time.perf_counter() - run_started,
    }
    (ARTIFACT_DIR / "training_summary_5fold.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n===== 5-FOLD BASE SUMMARY =====")
    print(metrics.groupby("model")["macro_f1"].agg(["mean", "std"]).to_string())
    print("best iterations:", best_iterations)
    print(f"total runtime: {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
