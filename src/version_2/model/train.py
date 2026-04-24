# src/version_2/train_catboost.py
#
# Stage 3 — CatBoost Model Training with Hyperparameter Tuning
# Trains a CatBoost classifier on v2 data with Optuna tuning.
#
# What this file does:
#   1. Loads CatBoost-ready train + val splits
#   2. Loads cat_features list from artifacts
#   3. Computes scale_pos_weight for class imbalance
#   4. Runs Optuna hyperparameter search (50 trials)
#   5. Trains final model on best params
#   6. Evaluates on val set — AUC-ROC, AUPRC, F1, precision, recall
#   7. Saves model, metrics, feature importance
#
# Usage (from project ROOT):
#   python -m src.version_2.train_catboost

import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ── Project root on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.data_configs import (
    V2_TRAIN,
    V2_VAL,
    ARTIFACTS,
    V2_RESULT,
)

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────
ARTIFACTS_DIR      = ARTIFACTS
CAT_FEATURES_PATH  = ARTIFACTS_DIR / "catboost_cat_features.json"
RESULT_DIR         = V2_RESULT
MODEL_PATH         = RESULT_DIR / "model.cbm"
METRICS_PATH       = RESULT_DIR / "metrics.json"
FI_PATH            = RESULT_DIR / "feature_importance.json"
BEST_PARAMS_PATH   = RESULT_DIR / "best_params.json"

# ── Training constants ─────────────────────────────────────
RANDOM_SEED        = 42
OPTUNA_TRIALS      = 50        # number of hyperparameter search trials
EARLY_STOPPING     = 100       # stop if val AUC doesn't improve for N rounds
MAX_ITERATIONS     = 5000      # upper bound — early stopping controls actual count
OPTUNA_EVAL_METRIC = "AUC"     # metric Optuna optimises during search
TARGET_COL         = "isFraud"


# ══════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads CatBoost-ready train and val splits.
    Validates files exist, are non-empty, and have the target column.
    Returns (train_df, val_df).
    """
    for path in [V2_TRAIN, V2_VAL]:
        if not path.exists():
            raise FileNotFoundError(
                f"Split not found: {path}\n"
                f"Run catboost_prep.py first."
            )

    logger.info("Loading CatBoost-ready splits ...")
    train = pd.read_csv(V2_TRAIN)
    val   = pd.read_csv(V2_VAL)

    for name, df in [("train", train), ("val", val)]:
        if TARGET_COL not in df.columns:
            raise ValueError(
                f"'{TARGET_COL}' missing from {name} split. "
                f"Re-run catboost_prep.py."
            )
        logger.info(
            f"  {name}: {df.shape[0]:,} rows × {df.shape[1]:,} cols | "
            f"fraud rate: {df[TARGET_COL].mean()*100:.2f}%"
        )

    return train, val


# ══════════════════════════════════════════════════════════
#  STEP 2 — LOAD CAT FEATURES
# ══════════════════════════════════════════════════════════

def load_cat_features() -> list[str]:
    """
    Loads categorical feature list from artifacts/catboost_cat_features.json.
    This was saved by catboost_prep.py — single source of truth.
    Filters to only columns that exist in the current dataframe.
    """
    if not CAT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"cat_features file not found: {CAT_FEATURES_PATH}\n"
            f"Run catboost_prep.py first."
        )

    with open(CAT_FEATURES_PATH, "r") as f:
        data = json.load(f)

    cat_features = data["cat_features"]
    logger.info(
        f"Loaded {len(cat_features)} categorical features: "
        f"{cat_features}"
    )
    return cat_features


# ══════════════════════════════════════════════════════════
#  STEP 3 — PREPARE X, y + POOLS
# ══════════════════════════════════════════════════════════

def prepare_pools(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_features: list[str],
) -> tuple[Pool, Pool, np.ndarray, np.ndarray, float]:
    """
    Splits X/y, computes scale_pos_weight, builds CatBoost Pools.

    CatBoost Pool:
      A special data container that tells CatBoost:
        - Which columns are categorical (cat_features)
        - What the target column is
      This is how CatBoost knows to apply ordered target encoding
      to categorical columns internally — no manual encoding needed.

    scale_pos_weight:
      Handles class imbalance without SMOTE.
      Formula: negative_count / positive_count
      Effect:  each fraud row is weighted ~27x more than a legit row
      Why not SMOTE: CatBoost's ordered boosting uses sample order
                     for target statistics — synthetic rows break this.
    """
    # ── Separate features and target ──────────────────────
    feature_cols = [c for c in train.columns if c != TARGET_COL]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL].values

    X_val   = val[feature_cols]
    y_val   = val[TARGET_COL].values

    # ── Filter cat_features to only cols present in X_train ──
    available_cats = [c for c in cat_features if c in X_train.columns]
    missing_cats   = set(cat_features) - set(available_cats)
    if missing_cats:
        logger.warning(
            f"Cat features in JSON but not in data: {sorted(missing_cats)}"
        )

    # ── Compute scale_pos_weight ───────────────────────────
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    logger.info(
        f"Class distribution — legit: {neg_count:,} | "
        f"fraud: {pos_count:,} | "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ── Build CatBoost Pools ───────────────────────────────
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=available_cats,
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=available_cats,
    )

    logger.info("CatBoost Pools built ✓")
    return train_pool, val_pool, y_val, available_cats, scale_pos_weight


# ══════════════════════════════════════════════════════════
#  STEP 4 — OPTUNA HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════

def run_optuna_search(
    train_pool: Pool,
    val_pool:   Pool,
    scale_pos_weight: float,
) -> dict:
    """
    Runs Optuna hyperparameter search over OPTUNA_TRIALS trials.
    Each trial trains a CatBoost model with sampled params
    and evaluates AUC on the val pool.
    Returns the best params dict found across all trials.

    Search space covers the 6 most impactful CatBoost params:
      depth           → tree depth (controls model complexity)
      learning_rate   → step size per iteration
      l2_leaf_reg     → L2 regularisation (prevents overfitting)
      bagging_temperature → randomness in row sampling
      random_strength → randomness in split scoring
      border_count    → number of splits evaluated per feature
    """
    logger.info(
        f"\n[ Optuna ] Starting hyperparameter search "
        f"— {OPTUNA_TRIALS} trials ..."
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations":         500,       # fast eval during search
            "eval_metric":        OPTUNA_EVAL_METRIC,
            "random_seed":        RANDOM_SEED,
            "verbose":            0,
            "scale_pos_weight":   scale_pos_weight,
            "early_stopping_rounds": 50,

            # ── Search space ──────────────────────────────
            "depth": trial.suggest_int(
                "depth", 4, 10
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
            "border_count": trial.suggest_int(
                "border_count", 32, 255
            ),
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        val_preds = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(val_pool.get_label(), val_preds)
        return auc

    # ── Suppress Optuna's per-trial output ────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_params = study.best_params
    best_auc    = study.best_value

    logger.info(
        f"[ Optuna ] Search complete ✓ — "
        f"best val AUC: {best_auc:.5f}"
    )
    logger.info(f"[ Optuna ] Best params: {best_params}")

    return best_params


# ══════════════════════════════════════════════════════════
#  STEP 5 — TRAIN FINAL MODEL
# ══════════════════════════════════════════════════════════

def train_final_model(
    train_pool:       Pool,
    val_pool:         Pool,
    best_params:      dict,
    scale_pos_weight: float,
) -> CatBoostClassifier:
    """
    Trains the final CatBoost model using best params from Optuna.
    Uses full MAX_ITERATIONS with early stopping on val AUC.
    Logs training progress every 100 iterations.
    """
    logger.info(
        f"\n[ Training ] Final model — "
        f"up to {MAX_ITERATIONS:,} iterations "
        f"(early stopping: {EARLY_STOPPING} rounds) ..."
    )

    final_params = {
        **best_params,
        "iterations":            MAX_ITERATIONS,
        "eval_metric":           OPTUNA_EVAL_METRIC,
        "random_seed":           RANDOM_SEED,
        "scale_pos_weight":      scale_pos_weight,
        "early_stopping_rounds": EARLY_STOPPING,
        "verbose":               100,     # log every 100 iterations
    }

    model = CatBoostClassifier(**final_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    best_iter = model.get_best_iteration()
    best_score = model.get_best_score()

    logger.info(
        f"[ Training ] Complete ✓ — "
        f"best iteration: {best_iter:,} | "
        f"best val score: {best_score}"
    )
    return model


# ══════════════════════════════════════════════════════════
#  STEP 6 — EVALUATE ON VAL SET
# ══════════════════════════════════════════════════════════

def evaluate_model(
    model:   CatBoostClassifier,
    val_pool: Pool,
    y_val:   np.ndarray,
) -> dict:
    """
    Evaluates the trained model on the val set.

    Metrics:
      AUC-ROC  → overall ranking ability (threshold-independent)
      AUPRC    → precision-recall tradeoff (best for imbalanced data)
      F1       → harmonic mean of precision and recall
      Precision → of all predicted fraud, how many were actually fraud
      Recall   → of all actual fraud, how many did we catch
      Confusion matrix → TP, FP, TN, FN breakdown

    Threshold: default 0.5 for F1/precision/recall.
    AUC-ROC and AUPRC are threshold-independent.
    """
    logger.info("\n[ Evaluation ] Computing val set metrics ...")

    # ── Predict probabilities ──────────────────────────────
    y_proba = model.predict_proba(val_pool)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    # ── Compute metrics ────────────────────────────────────
    auc_roc    = roc_auc_score(y_val, y_proba)
    auprc      = average_precision_score(y_val, y_proba)
    f1         = f1_score(y_val, y_pred, zero_division=0)
    precision  = precision_score(y_val, y_pred, zero_division=0)
    recall     = recall_score(y_val, y_pred, zero_division=0)
    cm         = confusion_matrix(y_val, y_pred)

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc":   round(float(auc_roc),   5),
        "auprc":     round(float(auprc),      5),
        "f1":        round(float(f1),         5),
        "precision": round(float(precision),  5),
        "recall":    round(float(recall),     5),
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "threshold": 0.5,
        "val_rows":  int(len(y_val)),
        "fraud_rate_val": round(float(y_val.mean()), 5),
    }

    # ── Log results ────────────────────────────────────────
    logger.info("─" * 50)
    logger.info(f"  AUC-ROC    : {auc_roc:.5f}")
    logger.info(f"  AUPRC      : {auprc:.5f}  ← primary metric for imbalanced data")
    logger.info(f"  F1 Score   : {f1:.5f}")
    logger.info(f"  Precision  : {precision:.5f}")
    logger.info(f"  Recall     : {recall:.5f}")
    logger.info(f"  Confusion  : TP={tp:,} FP={fp:,} TN={tn:,} FN={fn:,}")
    logger.info("─" * 50)

    return metrics


# ══════════════════════════════════════════════════════════
#  STEP 7 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════

def save_outputs(
    model:       CatBoostClassifier,
    metrics:     dict,
    best_params: dict,
) -> None:
    """
    Saves all training outputs to src/version_2/result/:
      model.cbm              → trained CatBoost model (binary)
      metrics.json           → val set evaluation metrics
      best_params.json       → Optuna best hyperparameters
      feature_importance.json → top feature importances by gain
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────
    model.save_model(str(MODEL_PATH))
    logger.info(f"Model saved: {MODEL_PATH}")

    # ── Metrics ────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {METRICS_PATH}")

    # ── Best params ────────────────────────────────────────
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best params saved: {BEST_PARAMS_PATH}")

    # ── Feature importance ─────────────────────────────────
    fi = model.get_feature_importance(prettified=True)

    # prettified=True returns a DataFrame with Feature and Importances cols
    fi_dict = dict(
        zip(fi["Feature Id"].tolist(),
            fi["Importances"].tolist())
    )
    # Save top 40 features sorted by importance
    top_fi = dict(
        sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:40]
    )
    with open(FI_PATH, "w") as f:
        json.dump(top_fi, f, indent=2)
    logger.info(f"Feature importance saved: {FI_PATH}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def run_training() -> dict:
    """
    Orchestrates all training steps.
    Returns metrics dict for use by calling pipeline.
    """
    start = time.time()

    logger.info("=" * 60)
    logger.info("CATBOOST TRAINING — START")
    logger.info(f"Train data : {V2_TRAIN}")
    logger.info(f"Val data   : {V2_VAL}")
    logger.info(f"Results    : {RESULT_DIR}")
    logger.info(f"Optuna trials: {OPTUNA_TRIALS}")
    logger.info("=" * 60)

    try:
        # Step 1 — Load data
        logger.info("\n[ Step 1 ] Loading data ...")
        train, val = load_splits()

        # Step 2 — Load cat features
        logger.info("\n[ Step 2 ] Loading cat_features ...")
        cat_features = load_cat_features()

        # Step 3 — Prepare Pools
        logger.info("\n[ Step 3 ] Building CatBoost Pools ...")
        train_pool, val_pool, y_val, available_cats, scale_pos_weight = (
            prepare_pools(train, val, cat_features)
        )

        # Step 4 — Optuna search
        logger.info("\n[ Step 4 ] Hyperparameter search ...")
        best_params = run_optuna_search(
            train_pool, val_pool, scale_pos_weight
        )

        # Step 5 — Train final model
        logger.info("\n[ Step 5 ] Training final model ...")
        model = train_final_model(
            train_pool, val_pool, best_params, scale_pos_weight
        )

        # Step 6 — Evaluate
        logger.info("\n[ Step 6 ] Evaluating on val set ...")
        metrics = evaluate_model(model, val_pool, y_val)

        # Step 7 — Save
        logger.info("\n[ Step 7 ] Saving outputs ...")
        save_outputs(model, metrics, best_params)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"CATBOOST TRAINING — COMPLETE ({elapsed:.1f}s)")
        logger.info("=" * 60)

        return metrics

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_training()